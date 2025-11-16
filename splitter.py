# splitter.py
# --------------------------------------------------------------------------------------
# Split a compound query into instruction-like segments AND (optionally) classify each
# segment as: solve / explain / code / other.
#
# Exposed API:
#   - segment_query(text: str, prefer: {"auto","sat","sentencex","regex"}, classify: bool)
#       -> List[{"type": <label or "segment">, "text": <segment>}]
#   - derive_task_views(question: str, prefer: {"auto","sat","sentencex","regex"})
#       -> (views, segments)
#          views = {"problem_view": str, "explain_view": str, "code_spec": str}
#          segments = list from segment_query(...) (may be [])
# --------------------------------------------------------------------------------------

from __future__ import annotations
import re
from typing import List, Dict, Tuple

__all__ = ["segment_query", "derive_task_views"]

# ------------------------------
# Optional imports (graceful)
# ------------------------------
_HAS_SAT = False
_HAS_SENTENCEX = False
try:
    from wtpsplit import SaT  # type: ignore
    _HAS_SAT = True
except Exception:
    _HAS_SAT = False

try:
    from sentencex import segment as sentencex_segment  # type: ignore
    _HAS_SENTENCEX = True
except Exception:
    _HAS_SENTENCEX = False


# ------------------------------
# Regex patterns (expanded + refined)
# ------------------------------
_SOLVE_RX = re.compile(r"""
\b(
    solve|find|calculate|compute|evaluate|approximate|estimate|
    derive|prove|show|demonstrate|establish|justify|
    simplify|expand|reduce|transform|rewrite|
    factor|factorise|factorize|
    integrate|differentiate|gradient|divergence|curl|laplacian|
    maximize|maximise|minimize|minimise|optimi[sz]e|
    determine|verify\s+that|check\s+whether|confirm\s+that|analy[sz]e|
    eigen(?:value|vector)s?|diagonaliz(?:e|e)|rank|trace|
    inverse|invert|null(?:\s*space)?|column\s*space|row\s*space|
    determinant|det|roots?|
    limit|series|sum|product|recurrence|expectation|variance|probabilit(?:y|ies)
)\b
""", re.I | re.VERBOSE)

# IMPORTANT: omit ambiguous tokens like "function", "method", "class" to avoid false code labeling
# NOTE: Removed "r\b" (R language) because it was matching "R^2" in math text and
# misclassifying solve segments as code.
_CODE_RX = re.compile(r"""
\b(
    # explicit mentions of programming
    code|snippet|script|program|implementation|module|package|library|api|cli|
    notebook|pipeline|

    # programming languages
    python|py|c\+\+|cpp|c#|java|javascript|typescript|ts|go|golang|rust|julia|
    matlab|octave|sql|sqlite|postgres(?:ql)?|mysql|swift|scala|kotlin|dart|fortran|
    bash|zsh|sh|powershell|ps1|shell|cmd|

    # programming/ML libraries
    numpy|sympy|scipy|pandas|polars|matplotlib|seaborn|plotly|bokeh|altair|
    sklearn|scikit-?learn|xgboost|lightgbm|catboost|
    pytorch|torch|tensorflow|keras|jax|flax|opencv|cv2|
    networkx|fastapi|flask|django|spark|pyspark|sqlalchemy
)\b
""", re.I | re.VERBOSE)

_EXPL_RX = re.compile(r"""
\b(
    # core explanation verbs (now with basic inflections)
    explain(?:s|ed|ing)?|
    describe|
    clarify|
    outline|
    summari[sz]e|

    # process markers
    steps?|
    step[-\s]*by[-\s]*step|
    procedure|
    walk\s+me\s+through|
    break\s+it\s+down|

    # why / how / intuition
    why|
    how|
    intuition|
    insight|
    motivat(?:e|ion)|
    reasoning|
    rationale|

    # justification (includes 'justification')
    justif(?:y|ication)|

    # commentary / annotation
    comment|
    annotate|

    # plain-language cues
    in\s+plain\s+(?:english|terms)|

    # ELI5-style cues
    like\s+i\s+am|
    like\s+i'm|
    eli5|
    as\s+if\s+to\s+a\s+beginner|
    for\s+(?:kids|a\s+child)
)\b
""", re.I | re.VERBOSE)


# Keep sequential cues STRICT (omit "first/second/third" etc. to avoid splitting "first few terms")
_SEQ_CUES = r"(?:then|and\s+then|next|after\s+that|afterwards|subsequently|finally)"

# Verbs that START a code intent for splitting (exclude bare 'code' to keep "write ... code" together)
_CODE_HEAD_VERBS = r"(?:code|snippet|script|program)"

# Verbs that START an explanation intent for splitting
_EXPL_VERBS = r"(?:explain|describe|clarify|outline|show|detail|justify|teach|summari(?:se|ze)|walk|break)"

_LEAD_CONNECTORS_RX = re.compile(r"""
^
(?:[,;.:]\s*)?
(?:please\s+|kindly\s+|now\s+)?                # polite/filler
(?:and\s+|&\s+|then\s+|next\s+|after\s+that\s+|afterwards\s+|subsequently\s+|finally\s+)?  # connectors
""", re.I | re.VERBOSE)

_TRIVIAL_OTHER = {
    "and", "&", "then", "next", "after that", "afterwards", "subsequently", "finally",
    "now", "please", "kindly", ",", ";", ":", "."
}


# ------------------------------
# Public: segment_query
# ------------------------------
def segment_query(
    text: str,
    prefer: str = "auto",  # "auto" | "sat" | "sentencex" | "regex"
    classify: bool = True
) -> List[Dict[str, str]]:
    text = text.strip()
    if not text:
        return []

    if prefer == "sat":
        segs = _split_with_sat(text)
    elif prefer == "sentencex":
        segs = _split_with_sentencex(text)
    elif prefer == "regex":
        segs = _split_with_regex(text)
    else:
        segs = _split_with_sat(text)
        if not segs:
            segs = _split_with_sentencex(text)
        if not segs:
            segs = _split_with_regex(text)

    # Refinement + cleanup
    segs = _refine_sequential_cues(segs)
    segs = [_strip_leading_connectors(s) for s in segs if s and s.strip()]

    if not classify:
        return [{"type": "segment", "text": s} for s in segs]

    items = [{"type": _classify_segment(s), "text": s} for s in segs]
    items = _coalesce_segments(items)  # drop trivial 'other' + merge adjacent same-class (preserve 'and' where natural)
    return items


# ------------------------------
# Public: derive_task_views
# ------------------------------
def derive_task_views(
    question: str,
    prefer: str = "auto"
) -> Tuple[Dict[str, str], List[Dict[str, str]]]:
    segs = segment_query(question, prefer=prefer, classify=True)

    problem_view = next((s["text"] for s in segs if s["type"] == "solve"), "").strip() or question.strip()
    explain_view = next((s["text"] for s in segs if s["type"] == "explain"), "").strip()
    code_spec    = next((s["text"] for s in segs if s["type"] == "code"), "").strip()

    # Clean small prefixes
    if explain_view:
        explain_view = re.sub(r"^(?:then\s+)?explain\b[:,]?\s*", "", explain_view, flags=re.I).strip()
        explain_view = _first_sentence(explain_view)
    if code_spec:
        code_spec = re.sub(
            r"^(?:and\s+)?(?:finally[,;:.]?\s+)?(?:please\s+)?(?:write|code|implement|generate|produce|provide|script|program)\b[:,]?\s*",
            "",
            code_spec,
            flags=re.I,
        ).strip()

    return (
        {
            "problem_view": problem_view,
            "explain_view": explain_view,
            "code_spec": code_spec,
        },
        segs
    )


# ======================================================================================
# Internals
# ======================================================================================

def _split_with_sat(text: str, model: str = "sat-3l") -> List[str]:
    if not _HAS_SAT:
        return []
    try:
        sat = SaT(model)
        segs = sat.split(text)
        return [_cleanup(s) for s in segs if _cleanup(s)]
    except Exception:
        return []

def _split_with_sentencex(text: str, lang: str = "en") -> List[str]:
    if not _HAS_SENTENCEX:
        return []
    try:
        segs = list(sentencex_segment(lang, text))
        return [_cleanup(s) for s in segs if _cleanup(s)]
    except Exception:
        return []

def _split_with_regex(text: str) -> List[str]:
    """
    Regex fallback that handles:
      - sentence boundaries (.?!)
      - sequential cues (then/next/after that/finally)
      - sub-intent splits like "… and write code", "… and explain the steps"
      - single-sentence chains such as "Find..., explain..., and write..."
    """
    t = _normalize_spaces(text)

    # Protect fenced code blocks
    code_blocks: Dict[str, str] = {}
    def _stash(m):
        key = f"__CODEBLOCK_{len(code_blocks)}__"
        code_blocks[key] = m.group(0)
        return key
    t = re.sub(r"```.*?```", _stash, t, flags=re.S)

    # Inject delimiters before sequential cues
    t = re.sub(rf"\b{_SEQ_CUES}\b", r"||| \g<0>", t, flags=re.I)

    # Inject delimiters around coordinators before verbs
    t = re.sub(rf"\b(?:and|&)\s+(?=(?:{_CODE_HEAD_VERBS})\b)", r"||| and ", t, flags=re.I)
    t = re.sub(rf"\b(?:and|&)\s+(?=(?:{_EXPL_VERBS})\b)", r"||| and ", t, flags=re.I)
    t = re.sub(rf",\s+(?=(?:{_EXPL_VERBS})\b)", r"||| ", t, flags=re.I)
    t = re.sub(rf",\s+(?=(?:{_CODE_HEAD_VERBS})\b)", r"||| ", t, flags=re.I)
    # Explicitly split on ", and write/implement/..." and ", and explain/..."
    t = re.sub(rf",\s+and\s+(?=(?:{_CODE_HEAD_VERBS})\b)", r"||| and ", t, flags=re.I)
    t = re.sub(rf",\s+and\s+(?=(?:{_EXPL_VERBS})\b)", r"||| and ", t, flags=re.I)

    # Split on sentence ends OR inserted delimiters
    parts = re.split(r"(?:\|\|\|\s*)|(?<=[.?!])\s+", t)
    parts = [_restore_codeblocks(_cleanup(p), code_blocks) for p in parts if _cleanup(p)]

    # Extra rule: break BEFORE explain verbs; for code, only before head verbs (not bare 'code')
    refined: List[str] = []
    for p in parts:
        p2 = re.sub(rf"\b(?=(?:{_EXPL_VERBS})\b)", "||| ", p, flags=re.I)
        p2 = re.sub(rf"(?<!\bto\s)\b(?=(?:{_CODE_HEAD_VERBS})\b)", "||| ", p2, flags=re.I)
        refined.extend([_cleanup(x) for x in re.split(r"\|\|\|\s*", p2) if _cleanup(x)])
    return refined

def _refine_sequential_cues(segments: List[str]) -> List[str]:
    out: List[str] = []
    for seg in segments:
        s = re.sub(rf"\b{_SEQ_CUES}\b", r"||| \g<0>", seg, flags=re.I)
        parts = [p for p in re.split(r"\|\|\|\s*", s) if _cleanup(p)]

        refined: List[str] = []
        for p in parts:
            p2 = re.sub(rf"\b(?:and|&)\s+(?=(?:write|implement|explain|describe|program|script)\b)", r"||| and ", p, flags=re.I)
            p2 = re.sub(r",\s+(?=(?:write|implement|explain|describe|program|script)\b)", r"||| ", p2, flags=re.I)
            # final guard: split before explain & code-head verbs
            p2 = re.sub(rf"\b(?=(?:{_EXPL_VERBS})\b)", "||| ", p2, flags=re.I)
            p2 = re.sub(rf"(?<!\bto\s)\b(?=(?:{_CODE_HEAD_VERBS})\b)", "||| ", p2, flags=re.I)
            refined.extend([_cleanup(x) for x in re.split(r"\|\|\|\s*", p2) if _cleanup(x)])
        out.extend(refined)
    return out

def _classify_segment(seg: str) -> str:
    t = seg.lower()
    # PRIORITY: code → explain → solve (prevents mislabeling code as solve)
    if _CODE_RX.search(t):
        return "code"
    if _EXPL_RX.search(t):
        return "explain"
    if _SOLVE_RX.search(t):
        return "solve"
    return "other"

def _first_sentence(s: str) -> str:
    parts = re.split(r"(?<=[.!?])\s+", s.strip())
    return parts[0].strip() if parts else s.strip()

def _strip_leading_connectors(s: str) -> str:
    s = s.strip()
    s = _LEAD_CONNECTORS_RX.sub("", s).strip()
    # Trim dangling trailing commas/spaces
    s = re.sub(r"[,\s]+$", "", s)
    return s

def _is_trivial_other(text: str) -> bool:
    t = text.strip().lower()
    if t in _TRIVIAL_OTHER:
        return True
    if re.fullmatch(r"[,;:.]+", t):
        return True
    return False

def _coalesce_segments(items: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Drop trivial connectors and merge adjacent segments of the same class (preserving natural 'and')."""
    merged: List[Dict[str, str]] = []
    for it in items:
        t, s = it["type"], it["text"].strip()
        if t == "other" and _is_trivial_other(s):
            continue
        if merged and merged[-1]["type"] == t:
            prev = merged[-1]["text"]

            # If the next chunk begins with another head verb, keep the natural 'and'
            andy = False
            if t in {"code", "explain"}:
                if re.match(rf"^(?:{_CODE_HEAD_VERBS})\b", s, flags=re.I) or re.match(rf"^(?:{_EXPL_VERBS})\b", s, flags=re.I):
                    andy = True

            # Detect "... to" ending to avoid "to and <verb>" (e.g., "code to and plot")
            to_inf = bool(re.search(r"\bto\s*$", prev, flags=re.I))

            if andy:
                if to_inf:
                    glue = " "  # → keeps "code to implement ..."
                elif not prev.endswith(("and", "and ", ",", ";", ":", " ")):
                    glue = " and "
                else:
                    glue = " "
            else:
                glue = "" if (prev.endswith((" ", "-")) or s.startswith((",", ".", ":", ";"))) else " "

            merged[-1]["text"] = (prev + glue + s).strip()
        else:
            merged.append({"type": t, "text": s})
    return merged


def _cleanup(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def _normalize_spaces(s: str) -> str:
    return re.sub(r"[ \t\r\f\v]+", " ", s).strip()

def _restore_codeblocks(text: str, code_blocks: Dict[str,str]) -> str:
    for k, v in code_blocks.items():
        text = text.replace(k, v)
    return text
