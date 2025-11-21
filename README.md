## COS781 Group 1 – Apriori Acceleration

### Overview
This project investigates how to speed up Apriori-based association-rule mining for the Instacart dataset (~3M orders, ~50K products). We combine sparse CSR baskets, QR-driven dimensionality reduction, and Direct Hashing & Pruning (DHP) to cut runtime and memory while preserving rule quality. All experiments default to a stratified sample of 20 000 orders (weekday/hour/department balanced) because of workstation limits; the pipeline can ingest the full dataset when more RAM/CPU is available, so current metrics should be read as conservative lower bounds.

### Development & Execution Environment
- **Google Colab only**: Every stage—data download, preprocessing, experiments, plotting—was authored and executed inside Colab notebooks.
- **Local execution out of scope**: Reproducing the workflow locally would require replicating the Colab stack (CUDA/MKL versions, kagglehub paths, large-memory hardware). This was beyond the project’s scope, so no local instructions/support are provided.

### Repository Contents
- `COS781_Group1_Ruan.ipynb` – primary experiment pipeline (data prep → sampling → QR/PCA → Apriori/DHP → metrics/visualizations).
- `generate_rule_quality_graph.py` – optional helper for plotting lift/confidence distributions.
- Deliverable artifacts (`*.pdf`, `.tex`, etc.) – final report components and reflections.

### Running the Notebook (Colab Workflow)
1. Upload/open `COS781_Group1_Ruan.ipynb` in Google Colab.
2. Upload your `kaggle.json` via Colab’s file pane; the notebook copies it to `~/.kaggle/kaggle.json`.
3. Run cells sequentially:
   - Download & merge Instacart CSVs using `kagglehub`.
   - Build sparse CSR baskets, log density.
   - Apply stratified sampling (`TOP_N_CUSTOMERS = 20000` by default).
   - Execute QR/PCA dimensionality reduction and optional customer segmentation.
   - Run baseline Apriori, QR-only, and QR+DHP variants; capture runtime, RAM, rule counts, and rule-quality metrics.
   - Render comparison tables/plots (additional rule-quality graphs via the helper script if desired).

### Key Configuration Knobs
- `TOP_N_CUSTOMERS`: stratified sample size (default 20 000; set to `None` for full data if Colab hardware permits).
- `gamma`: QR column-retention threshold (0–1) controlling dimensionality reduction intensity.
- `n_jobs`: parallel worker count for TruncatedSVD, customer scoring, and sparse filtering.
- DHP settings: hash bucket size and minimum support thresholds for candidate pruning.

### Outputs
- Console logs summarizing dataset sizes, sparsity, sampling diagnostics.
- Pandas tables comparing runtime, memory, rule counts, and average lift/confidence across methods.
- Plots showing how QR thresholds affect performance and rule volumes.
- Optional lift/confidence histograms from `generate_rule_quality_graph.py`.

### Limitations & Future Work
- Current benchmarks use the 20 000-order slice; scaling further awaits more generous hardware or cloud execution.
- Local/CI automation, containerization, and GPU-specific tuning were intentionally excluded.
- Future extensions could explore FP-Growth/Eclat baselines, GPU-based association mining, or automated hyperparameter sweeps once resource constraints are lifted.
