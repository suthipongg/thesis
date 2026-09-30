# AI Workspace Guide & Context (System Instructions)

> **⚠️ FOR AI ASSISTANT:**
> When starting a new session in this workspace, ALWAYS read this file FIRST to understand the thesis context, directory structure, and specific formatting rules.
> If you create a new directory or change the workflow, you MUST update this file.

## 1. Project Context

- **Domain:** Systems for Machine Learning / High-Performance Computing (HPC)
- **Topic:** Proactive Memory-Aware Auto-Tuning Middleware for PyTorch DataLoader on Commodity Hardware
- **Core Problem:** PyTorch DataLoader requires manual tuning of `num_workers` and `prefetch_factor`. Too few → GPU Starvation. Too many → Out-of-Memory (OOM) crash.
- **Target Solution:** A Python Middleware (Drop-in Replacement) that proactively monitors RAM usage and jointly auto-tunes `num_workers` + `prefetch_factor` with a Safety Rollback mechanism.
- **Key SOTA Baseline:** MinatoLoader (EuroSys '26) — tested on 512GB HPC only; does NOT proactively tune for RAM and has NOT been tested on Unified Memory (Apple Silicon).
- **Research Positioning:** Local In-Node, Commodity HW (16-32GB), supporting both CUDA (discrete VRAM) and Apple MPS (Unified Memory).

## 2. Directory Structure & Rules

This workspace is designed to be fully compatible with **Obsidian (Zettelkasten method)**. All markdown files must use `[[Links]]` and YAML frontmatter (`tags`, `aliases`).

```
thesis/
├── 00_Master_Dashboard.md       ← MOC / Entry point (Obsidian)
├── 01_Research_Idea.md          ← Core thesis blueprint (Gap Analysis, Q&A, Final Direction)
├── 02_Research_Diary.md         ← Daily logbook + Advisor meeting notes
├── AI_WORKSPACE_GUIDE.md        ← This file
│
├── 01_Knowledge_Base/
│   ├── 00_Knowledge_Base_Index.md   ← Index / Table of contents for KB
│   ├── 01_Papers/                   ← Raw PDF papers (source of truth)
│   ├── 02_Notes/                    ← Markdown summaries, analyses, plans
│   │   ├── Data_Pipeline_Bottlenecks_and_PyTorch_Limitations.md
│   │   ├── MinatoLoader_Citation_and_Foundation_Analysis.md
│   │   ├── AutoTuning_Taxonomy_and_Advisor_Feedback.md
│   │   ├── Thesis_Roadmap.md        ← 4-Phase execution plan
│   │   └── Presentation_Script.md  ← Advisor proposal presentation script
│   └── 03_Papers_MD/               ← AI-readable Markdown copies of papers
│
├── 02_Experiments/
│   ├── requirements.txt
│   ├── 1_train_discover/           ← Experiment 1: initial training discovery
│   └── 2_experiment_scaling/       ← Experiment 2: worker/throughput scaling
│       └── experiment_summary.md   ← Analysis: OS Page Cache, GPU Starvation findings
│
└── 03_Thesis_Draft/               ← Draft chapters of the master's thesis
```

### Key Rules per Directory

- **`01_Knowledge_Base/01_Papers/`:** Raw PDF files only. Do NOT write notes here.
- **`01_Knowledge_Base/02_Notes/`:** All analytical outputs — paper summaries, research plans, meeting notes, presentations. Always add YAML frontmatter and `[[wiki-links]]`.
- **`02_Experiments/`:** Source code and Jupyter notebooks. Keep `experiment_summary.md` alongside the code. Reference from `02_Research_Diary.md`.
- **`03_Thesis_Draft/`:** Draft thesis chapters only.
- **Root files:** Only `00_Master_Dashboard.md`, `01_Research_Idea.md`, `02_Research_Diary.md`, and this guide.

## 3. Behavioral Guidelines for AI

1. **Language:** Respond in Thai using professional, academic, yet concise language unless instructed otherwise.
2. **Coding:** Focus on optimization, profiling, and hardware-level performance issues (I/O, Memory, Concurrency).
3. **Obsidian Integration:** Whenever generating a markdown file, include properties (tags/aliases) and use `[[wiki-links]]` to connect it to `00_Master_Dashboard` or other relevant files.
4. **Maintenance:** If you add a new major component or change the structural logic of this repository, update this `AI_WORKSPACE_GUIDE.md` immediately.
5. **Citations & References:** ALWAYS include direct links (e.g., `[Link Text](file:///...)` for local files or `[Link Text](https://...)` for web URLs) and exact quotes when summarizing papers, extracting statistics, or quoting claims. This ensures the user can easily verify and read the original sources without searching.
6. **Terminology:** Always use "Worker Processes" (NOT "Worker Threads") when referring to PyTorch DataLoader workers spawned via `torch.multiprocessing`. tf.data uses "Threads" internally (C++), but PyTorch uses Processes.
7. **Key Corrections to Remember:**
   - tf.data auto-tuning uses **Gradient Descent on M/M/1/k Queueing Model** (NOT Hill-Climbing).
   - **Synergy** (OSDI '22) is a Cluster Scheduler, NOT a DataLoader or CNN quantization tool. Do NOT cite it in the context of data loading.
   - MinatoLoader is tested on **512GB RAM HPC**, NOT commodity hardware. Its Section 5.5 shows it *tolerates* memory constraints but does NOT *proactively tune* based on available RAM.
