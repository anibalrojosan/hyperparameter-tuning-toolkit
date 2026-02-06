# 🚀 Hyperparameter Tuning Toolkit

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Optuna](https://img.shields.io/badge/Optimization-Optuna-orange.svg)](https://optuna.org/)
[![DEAP](https://img.shields.io/badge/Framework-DEAP-green.svg)](https://deap.readthedocs.io/)

A high-performance benchmarking framework designed to evaluate and compare state-of-the-art hyperparameter optimization (HPO) strategies. This toolkit provides a modular environment for analyzing the trade-offs between computational efficiency and model accuracy, enabling data scientists to select the most effective tuning methodology for production-grade machine learning workflows.

## 📄 Table of Contents
1. [🎯 Project Overview & Objectives](#project-overview--objectives)
2. [🛠️ Tech Stack](#tech-stack)
3. [🏗️ Engineering Highlights](#engineering-highlights)
4. [📂 Project Structure](#project-structure)
5. [⚙️ Setup](#setup)
6. [🚀 Usage](#usage)
7. [🔍 Key Findings](#key-findings)
8. [💡 Conclusions](#conclusions)
9. [🔮 Future Work](#future-work)

---

## Project Overview & Objectives

This toolkit provides a robust environment for evaluating and implementing advanced hyperparameter optimization (HPO) strategies. It is designed to bridge the gap between theoretical optimization concepts and production-ready implementations by focusing on:

*   **Performance Benchmarking:** Comparative analysis of HPO techniques based on model accuracy (F1-Score, AUC) and computational overhead.
*   **Optimization Strategies:**
    *   **Grid Search:** Systematic exhaustive search for baseline establishment.
    *   **Random Search:** High-efficiency stochastic exploration of parameter spaces.
    *   **Bayesian Optimization (Optuna):** Sequential model-based optimization using surrogate probabilistic models.
    *   **Genetic Algorithms (DEAP):** Heuristic-based evolutionary search for complex, non-convex parameter landscapes.
*   **Engineering Best Practices:** Clean, modular implementation of data pipelines and visualization modules to ensure experiment reproducibility.

The primary objective is to provide a technical framework that quantifies the trade-offs between search time, resource consumption, and model performance across diverse real-world datasets.

## Tech Stack

*   **Machine Learning & Data Science:** `Scikit-Learn`, `Pandas`, `NumPy`, `SciPy` (for statistical distributions in Random Search).
*   **Optimization Frameworks:** 
    *   `Optuna`: For state-of-the-art Bayesian optimization.
    *   `DEAP`: For implementing custom Evolutionary Algorithms.
*   **Visualization:** `Matplotlib`, `Seaborn` (for performance and convergence analysis).
*   **Environment:** `Jupyter Notebooks`, `Python venv`.

## Engineering Highlights

This project is built with a modular architecture, ensuring that each component is independent and can be easily replaced or extended.

*   **Modular Architecture:** Clear separation between data ingestion (`src/data`), feature engineering (`src/features`), and model training.
*   **Reproducibility:** Use of fixed random seeds and strict dependency management via `requirements.txt`.
*   **Clean Code:** Implementation of docstrings and organized module structure for better maintainability.

## Project Structure

```text
hyperparameter-tuning-toolkit/
├── data/                 # 📊 Dataset storage (raw & processed)
│   ├── 01_raw/           # 📥 Raw, immutable data
│   └── 02_processed/     # 🧹 Cleaned data for modeling
├── notebooks/            # 📓 Jupyter notebooks for exploration and presentation
├── src/                  # 🛠️ Source code for use in this project
│   ├── data/             # 📥 Scripts to download or generate data
│   ├── features/         # 🏗️ Scripts to transform data for modeling
│   ├── models/           # 🧠 Scripts to train models
│   └── visualization/    # 📈 Scripts to create visualizations
├── tests/                # 🧪 Unit & Integration test suite
├── requirements.txt      # 📦 Project dependencies
└── README.md             # 📖 Project documentation
```

## Setup

### 1. Clone the Repository

```bash
git clone https://github.com/anibalrojosan/hyperparameter-tuning-toolkit
cd hyperparameter-tuning-toolkit
```

### 2. Create a Virtual Environment

It is highly recommended to use a virtual environment to manage dependencies.

```bash
# For Windows
python -m venv venv
.\venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Data Preparation (Mandatory)

Before running the notebooks, you must download and process the datasets:

```bash
# 1. Download raw data
python src/data/make_dataset.py

# 2. Process data for modeling
python src/features/build_features.py
```

**What happens here?**
*   `make_dataset.py`: Downloads the original datasets (`breast_cancer.csv` and `pima-indians-diabetes.csv`) into `data/01_raw/`.
*   `build_features.py`: Cleans, scales, and splits the data into training and testing sets, creating the final files in `data/02_processed/` (e.g., `cancer_X_train.csv`, `pima_y_test.csv`).

## Usage

The project is designed to be executed sequentially. You can run the experiments using **Jupyter Notebook** or directly within **VS Code**.

### Option A: Running in VS Code (Recommended)
VS Code has excellent built-in support for Jupyter Notebooks:
1.  Open any file in the `notebooks/` folder (e.g., `01_pima_diabetes_grid_random_bayesian.ipynb`).
2.  Click the **"Select Kernel"** button in the top right corner.
3.  Choose your virtual environment (`venv`).
4.  Run cells individually or use **"Run All"**.

### Option B: Running with Jupyter Notebook
1.  **Launch Jupyter:**
    ```bash
    jupyter notebook
    ```
2.  **Access in Browser:** Your browser will open a new tab. Navigate to the `notebooks/` folder and open the desired experiment.

### Experiment Workflow
*   **Notebook 01:** Compares traditional methods (`Grid`, `Random`) against `Bayesian Optimization`. It explores how traditional methods compare against sequential model-based optimization. Ideal for understanding the trade-off between time and precision.
*   **Notebook 02:** Explores `Genetic Algorithms`. A more advanced approach showing how to evolve a population of hyperparameters. It shows how evolutionary strategies can navigate complex parameter spaces.

### Internal Structure
Each notebook follows a standardized flow:
*   **Data Loading:** Automatic ingestion from `data/02_processed/`.
*   **Baseline:** Training a model with default parameters for benchmarking.
*   **Optimization:** Execution of the tuning technique (Optuna, DEAP, etc.).
*   **Evaluation:** Visual and metric comparison using the `src.visualization` module.


## Key Findings

The experiments with different hyperparameter tuning techniques across two datasets yielded several important insights:

### Notebook 01: Pima Indians Diabetes Dataset Benchmark

| Method | Time (s) | CV Score | Test F1-Score | Efficiency |
| :--- | :---: | :---: | :---: | :--- |
| **Baseline** | - | - | 0.6337 | 🟢 High |
| **Grid Search** | 115.59 | **0.772** | 0.6337 | 🔴 Low |
| **Random Search** | 36.01 | 0.763 | 0.6337 | 🟡 Medium |
| **Bayesian (Optuna)** | **28.49** | 0.761 | 0.6138 | 🟢 Ultra High |

### Notebook 02: Breast Cancer Wisconsin Dataset Benchmark

| Method | Performance (F1) | Improvement | Convergence |
| :--- | :---: | :---: | :--- |
| **Baseline** | 0.9488 | - | - |
| **Genetic Algorithm** | **0.9585** | +1.02% | 15 Generations |

## Conclusions

From the experiments, we can conclude that:
* **Always start simple**: Start with a baseline model before investing in complex tuning.
* **Choose the right method**: Choose tuning methods based on computational budget and parameter space complexity.

Each method has its own strengths and weaknesses, and the best method to use depends on the specific problem and dataset:
* **Random Search** offers excellent balance of performance and efficiency for many problems.
* **Genetic Algorithms** are valuable for complex parameter spaces where simpler methods may be inefficient.
* **Bayesian Optimization** is ideal for projects with limited time budgets requiring efficient search.

## Future Work

Some future work could include:

*   **Experiment Tracking:** Integrate **MLflow** or **Weights & Biases** to log and visualize all tuning trials and model artifacts.
*   **Advanced Models:** Extend the toolkit to include deep learning models using **PyTorch** or **TensorFlow**, exploring tuning for learning rates and architecture.
*   **Automated Pipeline:** Implement a full **CI/CD pipeline** for automated model retraining and deployment.
*   **Hyperband & BOHB:** Add support for more advanced algorithms like Hyperband or BOHB (Bayesian Optimization and Hyperband) for even more efficient search.

> This project was developed with ❤️ by [Anibal Rojo](https://github.com/anibalrojosan/).