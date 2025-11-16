# -*- coding: utf-8 -*-
# =========================================================================================
# Herschian Router v2 — STRICT Sequential Split Pipeline (Adaptive-N, Any # of Experts)
# -----------------------------------------------------------------------------------------
# 1) Split full_question into segments via splitter.derive_task_views(...)
#    -> segments with types: "other", "solve", "explain", "code".
#
# 2) Stage-1 (Global Loss):
#    - For each expert, compute CE loss on the FULL raw question string.
#    - Active experts = those within (1 + tau) * min_loss.
#
# 3) Stage-2 (Taskwise Loss with CONTEXT):
#    - context_text  = concat of all "other" segments (problem body etc.).
#    - For each task:
#         solve_for_loss   = context + solve_segment   (if any, else just context)
#         explain_for_loss = context + explain_segment (if any, else just context)
#         code_for_loss    = context + code_segment    (if any, else just context)
#
#      Loss prompts:
#         "Problem: <solve_for_loss>"
#         "Explain: <Explain <explain_for_loss> ...>"
#         "Code spec: <code_for_loss>"
#
#      Native expert rule:
#        - Each task has a native expert (math/chat/coder).
#        - Another expert can steal the role only if it's better by at least
#          cfg.min_relative_gain (e.g. 20% lower loss).
#
# 4) Stage-3 (GENERATION — STRICT SEQUENTIAL CHAIN):
#      let:
#         solve_seg_gen   = solve_for_loss              (same as loss text)
#         explain_seg_gen = raw_explain or explain_view
#         code_seg_gen    = raw_code or code_spec
#
#      - Step 1 (solve):
#           solve_prompt = [solve_instruction?, solve_seg_gen, "Answer:"]
#
#      - Step 2 (explain):
#           explain_prompt = [
#               explain_instruction?,
#               answer,
#               "Explain ...",
#               "Explanation:"
#           ]
#
#      - Step 3 (code):
#           code_prompt = [
#               code_instruction?,
#               explanation or answer,
#               code_seg_gen,
#               "Code:"
#           ]
#
#      => 2nd worker sees (output_1 + split_2),
#         3rd worker sees (output_2 + split_3).
#
# 5) Returns diagnostics +:
#       outputs = { "answer": answer, "explanation": explanation, "code": code }.
# =========================================================================================

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

# -----------------------------------------------------------------------------------------
# TorchVision shim (for environments where torchvision is missing, but transformers'
# image_utils still tries to import torchvision.transforms.InterpolationMode).
# -----------------------------------------------------------------------------------------
import sys
import types

# ===== TorchVision Shim (Safe for importlib.find_spec) =====
import sys, types, importlib.machinery

if "torchvision" not in sys.modules:
    tv = types.ModuleType("torchvision")
    tv.__file__ = "<shim>"
    tv.__path__ = []
    tv.__package__ = "torchvision"
    tv.__spec__ = importlib.machinery.ModuleSpec(
        name="torchvision",
        loader=None,
        is_package=True
    )
    sys.modules["torchvision"] = tv

    # ---- torchvision.transforms ----
    tr = types.ModuleType("torchvision.transforms")
    tr.__file__ = "<shim>"
    tr.__package__ = "torchvision"
    tr.__path__ = []
    tr.__spec__ = importlib.machinery.ModuleSpec(
        name="torchvision.transforms",
        loader=None,
        is_package=True,
    )

    class _InterpolationMode:
        NEAREST = 0
        NEAREST_EXACT = 0
        BILINEAR = 1
        BICUBIC = 2
        HAMMING = 3
        LANCZOS = 4
        BOX = 5

    tr.InterpolationMode = _InterpolationMode
    sys.modules["torchvision.transforms"] = tr

    # Empty stubs for modules transformers tries to import
    for m in [
        "torchvision.datasets",
        "torchvision.io",
        "torchvision.ops",
        "torchvision.utils",
        "torchvision.models",
        "torchvision._meta_registrations",
    ]:
        mod = types.ModuleType(m)
        mod.__file__ = "<shim>"
        mod.__path__ = []
        mod.__spec__ = importlib.machinery.ModuleSpec(
            name=m, loader=None, is_package=True
        )
        sys.modules[m] = mod


from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from splitter import derive_task_views

# ------------------------------ Debug knobs ------------------------------

DEBUG_ROUTER = True
PRINT_PREFIX_MAX = 800

# Make things stable on your rig
os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")
os.environ.setdefault("TORCHINDUCTOR_CUDAGRAPHS", "0")

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32
torch.set_grad_enabled(False)


# ============================== Path helpers ==============================

def _normalize_local_dir(path: str) -> Path:
    p = Path(path.strip()).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"[Router] Local model dir does not exist: {p}")
    if not p.is_dir():
        raise NotADirectoryError(f"[Router] Path is not a directory: {p}")
    return p


def _require_files(p: Path, needed: Tuple[str, ...]) -> None:
    missing = [f for f in needed if not (p / f).exists()]
    if missing:
        listing = "\n  - ".join(sorted(x.name for x in p.glob("*")))
        raise FileNotFoundError(
            f"[Router] Local model dir is missing required files: {missing}\n"
            f"[Router] Looked in: {p}\n"
            f"[Router] Found files:\n  - {listing or '(empty)'}\n"
            f"[Router] Ensure this is the *model root* (config.json, tokenizer, weights)."
        )


# ============================== Expert model ==============================

@dataclass
class Expert:
    name: str
    path: str
    tok: AutoTokenizer
    mdl: torch.nn.Module


def _max_len(mdl, fallback: int = 2048) -> int:
    try:
        m = getattr(mdl.config, "max_position_embeddings", None)
        return int(m) if isinstance(m, int) and m > 0 else fallback
    except Exception:
        return fallback


def load_expert(path: str, name: str) -> Expert:
    p = _normalize_local_dir(path)
    _require_files(p, ("config.json",))

    tok = AutoTokenizer.from_pretrained(
        str(p),
        use_fast=True,
        trust_remote_code=True,
        local_files_only=True,
    )
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token

    cfg = AutoConfig.from_pretrained(
        str(p),
        trust_remote_code=True,
        local_files_only=True,
    )
    model_type = getattr(cfg, "model_type", None)

    if model_type == "qwen2_5_vl":
        # Optional VL support if you ever plug Qwen2.5-VL here
        from transformers import Qwen2_5_VLForConditionalGeneration

        mdl = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            str(p),
            torch_dtype=_DTYPE,
            device_map="auto",
            trust_remote_code=True,
            local_files_only=True,
        ).eval()
    else:
        mdl = AutoModelForCausalLM.from_pretrained(
            str(p),
            torch_dtype=_DTYPE,
            device_map="auto",
            trust_remote_code=True,
            local_files_only=True,
        ).eval()

    return Expert(name=name, path=str(p), tok=tok, mdl=mdl)


def build_experts(configs: List[Dict]) -> List[Expert]:
    return [load_expert(c["path"], c["name"]) for c in configs]


# ============================== Loss helpers ==============================

def _enc_shift_self(tok, mdl, text: str) -> Dict[str, torch.Tensor]:
    maxlen = _max_len(mdl)
    enc = tok(
        text,
        add_special_tokens=True,
        return_tensors="pt",
        truncation=True,
        max_length=maxlen,
    )
    ids = enc["input_ids"]
    att = enc.get("attention_mask")
    if att is None:
        att = torch.ones_like(ids, dtype=torch.long)
    labels = ids.clone()
    return {"input_ids": ids, "attention_mask": att, "labels": labels}


@torch.inference_mode()
def ce_loss(mdl, batch: Dict[str, torch.Tensor]) -> float:
    try:
        dev = next(mdl.parameters()).device
        x = {k: v.to(dev) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        out = mdl(**x).loss
        val = float(out.detach().cpu().item())
        if not (val == val) or val in (float("inf"), float("-inf")):
            if DEBUG_ROUTER:
                print("[WARN] ce_loss non-finite; returning +inf")
            return float("inf")
        return val
    except Exception as e:
        if DEBUG_ROUTER:
            print(f"[WARN] ce_loss exception: {e}; returning +inf")
        return float("inf")


def _loss(E: Expert, task: str, text: str) -> float:
    batch = _enc_shift_self(E.tok, E.mdl, text)
    loss = ce_loss(E.mdl, batch)
    if DEBUG_ROUTER:
        preview = text[:PRINT_PREFIX_MAX].replace("\n", " \\n ")
        print(f"[LOSS/{task}] E={E.name} | loss={loss:.4f} | text='{preview}'")
    return loss


# ---------- Stage-1: Global loss on FULL raw question ----------

def stage1_global_losses(experts: List[Expert], full_question: str) -> Dict[str, float]:
    text = full_question.strip()
    losses: Dict[str, float] = {}
    for E in experts:
        losses[E.name] = ce_loss(E.mdl, _enc_shift_self(E.tok, E.mdl, text))
    if DEBUG_ROUTER:
        prev = text[:160].replace("\n", " \\n ")
        print(f"[S1] Prompt preview: '{prev}'")
        print("[S1] Losses:", {k: round(v, 4) for k, v in losses.items()})
    return losses


def select_active_experts(losses: Dict[str, float], tau: float) -> List[str]:
    if not losses:
        return []
    lmin = min(losses.values())
    cutoff = lmin * (1.0 + float(tau))
    actives = [k for k, v in losses.items() if v <= cutoff]
    if not actives:
        actives = [min(losses, key=losses.get)]
    return sorted(actives, key=lambda n: losses[n])


# ---------- Task-wise loss primitives ----------

def _solve_loss(E: Expert, text: str) -> float:
    t = (text or "").strip()
    if not t:
        return float("inf")
    if not t.lower().startswith(("problem", "solve", "find", "compute", "evaluate")):
        t = "Problem: " + t
    return _loss(E, "solve", t)


def _explain_loss(E: Expert, text: str) -> float:
    instr = (text or "").strip()
    if not instr:
        return float("inf")
    if not instr.lower().startswith("explain"):
        instr = "Explain " + instr
    t = f"Explain: {instr}"
    return _loss(E, "explain", t)


def _code_loss(E: Expert, text: str) -> float:
    spec = (text or "").strip()
    if not spec:
        return float("inf")
    t = f"Code spec: {spec}"
    return _loss(E, "code", t)


def _pick_best(mp: Dict[str, float]) -> str:
    return min(mp, key=mp.get)


def _native_for(task: str, cfg: "RouterConfig") -> Optional[str]:
    if task == "solve":
        return cfg.native_solve
    if task == "explain":
        return cfg.native_explain
    if task == "code":
        return cfg.native_code
    return None


def _pick_with_native(task: str, mp: Dict[str, float], cfg: "RouterConfig") -> str:
    """
    Native expert keeps the role unless someone beats it by >= cfg.min_relative_gain.
    """
    if not mp:
        raise ValueError(f"No losses for task={task!r}")

    native = _native_for(task, cfg)
    if not native or native not in mp:
        return _pick_best(mp)

    native_loss = float(mp[native])
    best = _pick_best(mp)
    best_loss = float(mp[best])

    if best == native or not (native_loss == native_loss) or native_loss == float("inf"):
        return native

    threshold = native_loss * (1.0 - float(cfg.min_relative_gain))

    if DEBUG_ROUTER:
        print(
            f"[NATIVE] task={task} | native={native} loss={native_loss:.4f} "
            f"| best={best} loss={best_loss:.4f} | threshold={threshold:.4f}"
        )

    if best_loss <= threshold:
        return best
    return native


# ---------- Exclusivity (optional) ----------

def _enforce_exclusive(
    assignments: Dict[str, str],
    losses: Dict[str, Dict[str, float]],
    cfg: "RouterConfig",
) -> Dict[str, str]:
    if not cfg.exclusive_roles:
        return assignments

    tasks = list(assignments.keys())
    winners = [assignments[t] for t in tasks]
    uniq = set(winners)

    # If everyone picked the same expert and we allow that, keep as is
    if len(uniq) == 1 and cfg.exclusive_allow_all_if_best_all:
        return assignments

    # Invert: expert -> list of tasks
    inv: Dict[str, List[str]] = {}
    for t in tasks:
        inv.setdefault(assignments[t], []).append(t)

    for expert, ts in inv.items():
        if len(ts) <= 1:
            continue

        # For this expert, we keep the task where it has the largest margin over next best
        diffs = []
        for t in ts:
            items = sorted(losses[t].items(), key=lambda kv: kv[1])
            idx = [i for i, (n, _) in enumerate(items) if n == expert][0]
            next_loss = None
            for n, l in items:
                if n != expert:
                    next_loss = l
                    break
            margin = (next_loss - items[idx][1]) if next_loss is not None else 1e9
            diffs.append((t, margin))

        diffs.sort(key=lambda x: x[1], reverse=True)
        keep_task = diffs[0][0]

        for t, _ in diffs[1:]:
            ranked = sorted(losses[t].items(), key=lambda kv: kv[1])
            for cand, _ in ranked:
                if cand not in assignments.values() or cand == expert:
                    continue
                assignments[t] = cand
                break

    return assignments


# ============================== Generation ==============================

def _supports_chat(tok) -> bool:
    return hasattr(tok, "apply_chat_template") and callable(
        getattr(tok, "apply_chat_template")
    )


def _gen_kwargs(cfg: "RouterConfig") -> dict:
    kw = dict(
        do_sample=cfg.do_sample,
        use_cache=True,
    )
    if cfg.do_sample:
        if cfg.temperature is not None:
            kw["temperature"] = cfg.temperature
        if cfg.top_p is not None:
            kw["top_p"] = cfg.top_p
        if cfg.top_k is not None:
            kw["top_k"] = cfg.top_k
    return kw


def _generate(E: Expert, prompt: str, max_new: int, cfg: "RouterConfig") -> str:
    dev = next(E.mdl.parameters()).device
    tok = E.tok

    if _supports_chat(tok):
        messages = [
            {"role": "system", "content": "Follow the instructions strictly."},
            {"role": "user", "content": str(prompt)},
        ]
        input_ids = tok.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(dev)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=dev)
    else:
        enc = tok(str(prompt), return_tensors="pt")
        input_ids = enc["input_ids"].to(dev)
        attention_mask = enc.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=dev)
        else:
            attention_mask = attention_mask.to(dev)

    in_len = int(input_ids.shape[1])
    kw = _gen_kwargs(cfg)

    out = E.mdl.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=int(max_new),
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id or tok.eos_token_id,
        **kw,
    )

    gen_ids = out[0, in_len:]
    text = tok.decode(gen_ids, skip_special_tokens=True)
    return text.rstrip()


def _gen_solve(E: Expert, seg: str, cfg: "RouterConfig") -> str:
    seg = (seg or "").strip()
    if not seg:
        return ""

    if not seg.lower().startswith(("problem", "solve", "find", "compute", "evaluate")):
        seg = "Problem: " + seg

    instr = (cfg.solve_instruction or "").strip()
    parts = []
    if instr:
        parts.append(instr)
    parts.append(seg)
    parts.append("Answer:")

    prompt = "\n\n".join(parts)
    return _generate(E, prompt, cfg.max_new_solve, cfg)


def _gen_explain(E: Expert, prev: str, seg: str, cfg: "RouterConfig") -> str:
    seg = (seg or "").strip()
    if not seg:
        return ""

    if not seg.lower().startswith("explain"):
        seg = "Explain " + seg

    instr = (cfg.explain_instruction or "").strip()
    parts = []
    if instr:
        parts.append(instr)
    if prev.strip():
        parts.append(prev.strip())
    parts.append(seg)
    parts.append("Explanation:")

    prompt = "\n\n".join(parts)
    return _generate(E, prompt, cfg.max_new_explain, cfg)


def _gen_code(E: Expert, prev: str, seg: str, cfg: "RouterConfig") -> str:
    seg = (seg or "").strip()
    if not seg:
        return ""

    instr = (cfg.code_instruction or "").strip()
    parts = []
    if instr:
        parts.append(instr)
    if prev.strip():
        parts.append(prev.strip())
    parts.append(seg)
    parts.append("Code:")

    prompt = "\n\n".join(parts)
    return _generate(E, prompt, cfg.max_new_code, cfg)


# ============================== RouterConfig ==============================

@dataclass
class RouterConfig:
    tau: float = 0.5

    # Exclusivity (for compatibility with older code / notebooks)
    exclusive_roles: bool = False
    exclusive_allow_all_if_best_all: bool = True

    # Native experts per role
    native_solve: Optional[str] = None
    native_explain: Optional[str] = None
    native_code: Optional[str] = None
    min_relative_gain: float = 0.20

    # Role-specific instructions
    solve_instruction: Optional[str] = None
    explain_instruction: Optional[str] = None
    code_instruction: Optional[str] = None

    # Generation knobs
    max_new_solve: int = 2048
    max_new_explain: int = 2048
    max_new_code: int = 2048
    do_sample: bool = False
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None


# ============================== Top-level routing ==============================

def route_and_execute(
    experts: List[Expert],
    full_question: str,
    global_instruction: str,
    cfg: RouterConfig,
    tasks_override: Optional[List[str]] = None,
) -> Dict[str, object]:
    """
    Main entry:
      - full_question: raw question text (your ValidationHard query).
      - global_instruction: currently unused (you embed role hints in cfg.*_instruction).
      - tasks_override: if not None, subset of {"solve","explain","code"}.
    """

    # ---- Stage 0: splitter (Tuple-2 from your splitter.py) ----
    views, segments = derive_task_views(full_question, prefer="auto")

    problem_view = (views.get("problem_view") or full_question).strip()
    explain_view = (views.get("explain_view") or "").strip()
    code_spec = (views.get("code_spec") or "").strip()

    # RAW segments by type (for loss + generation)
    raw_solve = next((s["text"] for s in segments if s.get("type") == "solve"), "").strip()
    raw_explain = next((s["text"] for s in segments if s.get("type") == "explain"), "").strip()
    raw_code = next((s["text"] for s in segments if s.get("type") == "code"), "").strip()

    other_contexts = [s.get("text", "").strip() for s in segments if s.get("type") == "other"]
    context_text = " ".join([t for t in other_contexts if t]).strip()

    def _with_context(core: str) -> str:
        core = (core or "").strip()
        if context_text and core:
            return context_text + "\n\n" + core
        return core or context_text

    # ---- Texts used for LOSS (always include context when available) ----
    solve_for_loss = _with_context(raw_solve or problem_view)
    explain_for_loss = _with_context(raw_explain or explain_view or problem_view)
    code_for_loss = _with_context(raw_code or code_spec)

    # ---- Texts used for GENERATION ----
    #   - solve_segment_for_gen must == solve_for_loss (your explicit requirement).
    #   - explain/code segments get the *instructional* text; context is provided
    #     by the chained prev=answer/explanation.
    solve_segment_for_gen = solve_for_loss
    explain_segment_for_gen = raw_explain or explain_view
    code_segment_for_gen = raw_code or code_spec

    # Decide tasks
    if tasks_override:
        tasks = [t for t in tasks_override if t in {"solve", "explain", "code"}]
    else:
        tasks = ["solve"]
        if (explain_segment_for_gen or explain_for_loss) and (raw_explain or explain_view):
            tasks.append("explain")
        if code_segment_for_gen or code_for_loss:
            tasks.append("code")

    # ---------- Stage 1: global losses ----------
    s1_losses = stage1_global_losses(experts, full_question)
    active_names = select_active_experts(s1_losses, cfg.tau)
    pool = {E.name: E for E in experts if E.name in active_names}

    # ---------- Stage 2: task-wise losses ----------
    per_task_losses: Dict[str, Dict[str, float]] = {}

    if "solve" in tasks and solve_for_loss:
        per_task_losses["solve"] = {
            name: _solve_loss(E, solve_for_loss) for name, E in pool.items()
        }
    if "explain" in tasks and explain_for_loss:
        per_task_losses["explain"] = {
            name: _explain_loss(E, explain_for_loss) for name, E in pool.items()
        }
    if "code" in tasks and code_for_loss:
        per_task_losses["code"] = {
            name: _code_loss(E, code_for_loss) for name, E in pool.items()
        }

    assignments: Dict[str, str] = {}
    for task, mp in per_task_losses.items():
        assignments[task] = _pick_with_native(task, mp, cfg)

    assignments = _enforce_exclusive(assignments, per_task_losses, cfg)

    # ---------- Stage 3: STRICT SEQUENTIAL GENERATION ----------
    answer = ""
    explanation = ""
    code = ""

    if "solve" in tasks and "solve" in assignments:
        chosen = assignments["solve"]
        answer = _gen_solve(pool[chosen], solve_segment_for_gen, cfg)

    if "explain" in tasks and "explain" in assignments:
        chosen = assignments["explain"]
        explanation = _gen_explain(
            pool[chosen],
            prev=answer,
            seg=explain_segment_for_gen or explain_for_loss,
            cfg=cfg,
        )

    if "code" in tasks and "code" in assignments:
        chosen = assignments["code"]
        code = _gen_code(
            pool[chosen],
            prev=explanation or answer,
            seg=code_segment_for_gen or code_for_loss,
            cfg=cfg,
        )

    return {
        "views": {
            "problem_view": problem_view,
            "explain_view": explain_view,
            "code_spec": code_spec,
        },
        "raw_segments": {
            "solve": raw_solve,
            "explain": raw_explain,
            "code": raw_code,
            "other": other_contexts,
        },
        "stage1_losses": s1_losses,
        "active_experts": active_names,
        "tasks": tasks,
        "per_task_losses": per_task_losses,
        "assignments": assignments,
        "outputs": {
            "answer": answer,
            "explanation": explanation,
            "code": code,
        },
    }


# ============================== Pretty printer ==============================

def pretty_print_run(
    full_question: str,
    global_instruction: str,
    result: Dict[str, object],
) -> None:
    print("=" * 120)
    print("Q:", full_question)
    print("Instruction:", global_instruction)
    print("-" * 120)

    views = result.get("views", {})
    raw_segments = result.get("raw_segments", {})
    s1 = result.get("stage1_losses", {})
    active = result.get("active_experts", [])
    tasks = result.get("tasks", [])
    per_task = result.get("per_task_losses", {})
    assignments = result.get("assignments", {})
    outs = result.get("outputs", {})

    print("Views:", views)
    print("Raw segments:", raw_segments)
    if s1:
        print("Stage-1 losses:", {k: round(float(v), 4) for k, v in s1.items()})
    print("Active experts:", active)
    print("Tasks:", tasks)
    for t, mp in per_task.items():
        print(f"{t.title()} losses:", {k: round(float(v), 4) for k, v in mp.items()})
    print("Assignments:", assignments)
    print("-" * 120)

    if outs.get("answer"):
        print("Answer:\n", outs["answer"], "\n")
    if outs.get("explanation"):
        print("Explanation:\n", outs["explanation"], "\n")
    if outs.get("code"):
        print("Code (python):\n", outs["code"], "\n")
    print("=" * 120)
