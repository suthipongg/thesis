# AI Workspace Guide & Context (System Instructions)

> **⚠️ FOR AI ASSISTANT:** 
> When starting a new session in this workspace, ALWAYS read this file FIRST to understand the thesis context, directory structure, and specific formatting rules.
> If you create a new directory or change the workflow, you MUST update this file.

## 1. Project Context
- **Domain:** Systems for Machine Learning / High-Performance Computing (HPC)
- **Topic:** Hardware-Aware Adaptive Data Loading Framework for Large-Scale Tensor Data Training.
- **Core Problem:** 
  1. OS Page Cache thrashing on Commodity Hardware (RAM constraints).
  2. Cold-Start Latency & Ingestion bottlenecks on HPC servers.
- **Target Solution:** A Python Middleware Framework for PyTorch DataLoader that performs auto-tuning (Workers, Prefetch, Batch Size) based on hardware profiling.

## 2. Directory Structure & Rules
This workspace is designed to be fully compatible with **Obsidian (Zettelkasten method)**. All markdown files must use `[[Links]]` and YAML frontmatter (`tags`, `aliases`).

- **`01_Knowledge_Base/`**
  - **Purpose:** Literature reviews, paper summaries, and theoretical concepts.
  - **Structure:** Divided into `01_Papers/` (for raw PDF files) and `02_Notes/` (for markdown summaries).
  - **Rule:** Link paper summaries back to `[[01_Research_Idea]]` (Thesis Plan).
- **`02_Research_Log/`**
  - **Purpose:** Store high-level analysis, empirical experiment conclusions, and analytical summaries.
  - **Rule:** When an experiment finishes, write an analysis here and link it to `02_Research_Diary.md`.
- **`03_Experiments/`**
  - **Purpose:** ALL executable code, Jupyter Notebooks, and local experiment results.
  - **Rule:** Keep local summaries (e.g., `experiment_1_summary.md`) next to the code, and reference them in the main Research Log.
- **`04_Thesis_Draft/`**
  - **Purpose:** Markdown files for drafting the actual chapters of the master's thesis.
- **Root Files:**
  - `00_Master_Dashboard.md`: The main Map of Content (MOC) for Obsidian.
  - `01_Research_Idea.md`: The core blueprint and architecture design.
  - `02_Research_Diary.md`: The daily logbook for tracking daily progress, bugs, and next steps.

## 3. Behavioral Guidelines for AI
1. **Language:** Respond in Thai using professional, academic, yet concise language unless instructed otherwise.
2. **Coding:** Focus on optimization, profiling, and hardware-level performance issues (I/O, Memory, Concurrency).
3. **Obsidian Integration:** Whenever generating a markdown file, include properties (tags/aliases) and use `[[wiki-links]]` to connect it to `00_Master_Dashboard` or other relevant files.
4. **Maintenance:** If you add a new major component or change the structural logic of this repository, update this `AI_WORKSPACE_GUIDE.md` immediately.
5. **Citations & References:** ALWAYS include direct links (e.g., `[Link Text](file:///...)` for local files or `[Link Text](https://...)` for web URLs) and exact quotes when summarizing papers, extracting statistics, or quoting claims. This ensures the user can easily verify and read the original sources without searching.
