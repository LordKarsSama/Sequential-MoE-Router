# Sequential Specialist MoE Routing Architecture for Complex Multi-Stage Queries

### **Author**
- **M. G. Shree Harsha** (24BDS037)  
- Solo Project Submission – AI 3rd Semester  
- IIIT Dharwad  

---

## 📌 Project Overview

This project implements a **Sequential Specialist Mixture-of-Experts (MoE) Router** designed to handle complex multi-stage queries such as:

> **“Solve the problem → Explain the reasoning → Generate the code.”**

Instead of relying on a single large generalist model, this system coordinates **three domain-specialist LLMs**:

- **Qwen2.5-1.5B-Math** → mathematical reasoning  
- **Qwen2.5-0.5B-Instruct** → general explanation  
- **Qwen2.5-0.5B-Coder** → programming/coding tasks  

A **two-stage router** assigns segments of the query to the most suitable model based on **loss-based evaluation**, producing a structured pipeline:

1. **Stage 1:** Select relevant experts using LM loss on the full query  
2. **Stage 2:** Route individual segments (solve → explain → code)  
3. **Pipeline:** Execute selected experts sequentially and merge the outputs  

This architecture reduces compute cost, increases specialization, and achieves performance close to a 3B generalist model on difficult PhD-level tasks.

---
## Required Model Weights

Download these three experts:

- Qwen2.5-Math-1.5B-Instruct  
- Qwen2.5-0.5B-Instruct  
- Qwen2.5-Coder-0.5B-Instruct

Place each model in its own folder **directly inside the project directory**, like this:

```
YourProjectFolder/
├── Math/
│   └── Qwen2.5-Math-1.5B-Instruct
├── Qwen2.5-0.5B-Instruct/
│   └── (model files here)
├── Qwen2.5-Coder-0.5B-Instruct/
│   └── (model files here)
```
YourProjectFolder/
├── splitter.py
├── router_v2.py
├── Use.ipynb
├── pipeline.png
├── Lospicking.png
├── Report.pdf
├── Report.tex
├── Results.pdf
├── LICENSE
├── NOTICE
├── README.md
├── Math/
│   └── Qwen2.5-Math-1.5B-Instruct (model files)
├── Qwen2.5-0.5B-Instruct/
│   └── (model files)
└── Qwen2.5-Coder-0.5B-Instruct/
    └── (model files)


you do not need PNGs, Notes and all that for running them just take python and jupyter codes, all of them are necessary.
