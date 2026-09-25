# Amazon ML Challenge 2026 - Entity Resolution

A machine learning solution for entity resolution and record linkage across noisy multi-source datasets (Source 1, Source 2, and Source 3) to match business identities across diverse platforms.

---

## 📌 Project Overview

In large-scale business directories and e-commerce platforms, identical business entities are frequently registered under slightly different names, abbreviated legal structures (e.g., *Ltd*, *Pvt*, *Inc*), varied street address formats, or multilingual conventions.

The objective of this challenge is to resolve and link entities from **Source 1** to their corresponding matches in **Source 2** and **Source 3** using a blend of:
- **Text Normalization & Preprocessing** (regex cleaning, legal entity stripping, punctuation removal).
- **Fuzzy String Matching** (Levenshtein, token sort, Jaro-Winkler via `rapidfuzz`).
- **Semantic Text Embeddings** (`sentence-transformers` for dense name/address matching).
- **Gradient Boosted Ranking / Classification** (`LightGBM`, `XGBoost`).

---

## 📂 Repository Structure

```plaintext
Amazon-ML-Challenge/
├── dataset/                    # Dataset directory (gitignored due to file size)
│   ├── train/
│   │   ├── train_source1.tsv
│   │   ├── train_source2.tsv
│   │   ├── train_source3.tsv
│   │   └── train_ground_truth.tsv
│   └── test/
│       ├── test_source1.tsv
│       ├── test_source2.tsv
│       └── test_source3.tsv
├── notebooks/                  # Jupyter notebooks for EDA and modeling
│   └── 01_data_exploration.ipynb
├── src/                        # Modular source code (feature engineering, models, pipeline)
├── .gitignore                  # Excludes datasets, environments, caches, models
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation & setup instructions
```

---

## ⚙️ Prerequisites

- **OS**: Windows / Linux / macOS
- **Python**: `3.10` – `3.13` (*Python 3.11 or 3.12 recommended*)
- **Environment Tool**: [Miniconda](https://docs.anaconda.com/miniconda/) or Python `venv`
  > **Note for Windows Users**: Using Conda/Miniconda is highly recommended to prevent Windows Application Control / WDAC DLL execution errors with compiled scientific packages like `scipy` and `scikit-learn`.

---

## 🚀 Local Setup Instructions

Follow these steps to set up the repository locally on your machine.

### 1. Clone the Repository

```bash
git clone https://github.com/abhishekck31/Amazon-ML-Challenge.git
cd Amazon-ML-Challenge
```

---

### 2. Set Up the Python Environment

#### Option A: Using Conda / Miniconda (Recommended)

```bash
# Create a new environment
conda create -n amazon-ml python=3.13 -y

# Activate the environment
conda activate amazon-ml
```

#### Option B: Using standard Python `venv`

```bash
# On Windows PowerShell
python -m venv venv
.\venv\Scripts\Activate.ps1

# On Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

---

### 3. Install Dependencies

Install all required packages from `requirements.txt`:

```bash
pip install -r requirements.txt
```

Core libraries installed:
- `pandas`, `numpy`: High-performance data manipulation
- `scikit-learn`: Metrics, evaluation, cross-validation, and baseline models
- `rapidfuzz`: High-speed string similarity and token matching
- `sentence-transformers`: Dense neural semantic representations
- `lightgbm`, `xgboost`: Gradient boosting frameworks
- `tqdm`, `matplotlib`: Progress tracking and data visualization
- `ipykernel`: Jupyter notebook kernel integration

---

### 4. Register the Jupyter Kernel

To make sure your Jupyter notebooks run using the correct environment:

```bash
python -m ipykernel install --user --name=amazon-ml --display-name "Python (Amazon ML)"
```

When opening any notebook (e.g. in VS Code or JupyterLab), select the **`Python (Amazon ML)`** kernel from the top-right kernel picker.

---

### 5. Dataset Placement

Because competition dataset files are large (> 500 MB) and excluded from Git tracking, you must place your competition files in the `dataset/` directory:

1. Create the `dataset/` folder structure if it doesn't already exist:
   ```bash
   mkdir -p dataset/train dataset/test
   ```
2. Place the TSV files in their respective folders:
   - `dataset/train/train_source1.tsv`
   - `dataset/train/train_source2.tsv`
   - `dataset/train/train_source3.tsv`
   - `dataset/train/train_ground_truth.tsv`
   - `dataset/test/test_source1.tsv`
   - `dataset/test/test_source2.tsv`
   - `dataset/test/test_source3.tsv`

---

## 🧪 Running the Notebooks

1. Start Jupyter Lab / Notebook or open the project folder in VS Code:
   ```bash
   jupyter lab
   ```
2. Open [`notebooks/01_data_exploration.ipynb`](notebooks/01_data_exploration.ipynb).
3. Confirm that the kernel in the top right is set to **`Python (Amazon ML)`**.
4. Run cells sequentially to explore dataset dimensions, missing values, text cleaning, and baseline train/validation splits.

---

## 📝 Git Workflow & Best Practices

- Large files (`*.tsv`, `*.csv`, `*.parquet`, `*.pth`, `*.bin`, `models/`, `dataset/`) are automatically ignored via `.gitignore` to keep the repository lightweight and under GitHub's 100 MB file limit.
- Before committing new changes:
  ```bash
  git status
  git add .
  git commit -m "Description of changes"
  git push origin main
  ```

---

## 👥 Contributors

- **Abhishek** ([@abhishekck31](https://github.com/abhishekck31))
