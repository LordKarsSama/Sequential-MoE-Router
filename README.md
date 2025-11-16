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

## 📁 Repository Structure

```
.
├── splitter.py                 # Query segmentation logic (solve/explain/code)
├── router_v2.py                # Two-stage loss-based routing architecture
├── pipeline.png                # Architecture diagram
├── Lospicking.png              # Visualization of loss-based expert selection
├── Use.ipynb                   # Running the router end-to-end
├── Validationcontents/
│   ├── Validation3B.ipynb
│   ├── ValidationMoE.ipynb
│   ├── ValidationHard.jsonl
│   ├── validationHard.MoE_results.md
│   ├── validationHard.MoE_results.txt
│   ├── ValidationHard_Qwen2_5_VL_3B.md
│   └── ...
├── Report.pdf                  # IEEE-format project report
├── Results.pdf                 # Final graded comparison results
├── LICENSE                     # Apache 2.0 License
├── NOTICE                      # Model ownership + attribution statement
└── README.md
```
