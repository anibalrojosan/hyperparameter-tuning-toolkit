# Technical Roadmap: Hyperparameter Tuning Toolkit (MLOps Evolution)

This document outlines the evolution of the project from a experimentation-focused prototype to a production-ready hyperparameter tuning service. It is a living document that will be updated as the project evolves.

Last Updated: 2026-02-08

## Tech Stack

### Core ML & Optimization (Existing)
| Category | Tool | Purpose |
| --- | --- | --- |
| **Optimization** | `Optuna` | Bayesian optimization for efficient hyperparameter search. |
| **Optimization** | `DEAP` | Evolutionary algorithms for complex parameter landscapes. |
| **Machine Learning**| `Scikit-Learn` | Base models and evaluation metrics. |
| **Data Processing** | `Pandas` & `NumPy` | Data manipulation and numerical operations. |
| **Visualization** | `Matplotlib` & `Seaborn` | Performance and convergence analysis. |

### MLOps & Infrastructure (Evolution)
| Category | Tool | Purpose | Phase |
| --- | --- | --- | --- |
| **Package Manager** | `uv` | High-performance dependency management and Python isolation. | Phase 1 |
| **Core Framework** | `FastAPI` | Exposing the tuning engine as a production-ready service. | Phase 1 |
| **Validation** | `Pydantic v2` | Strict data validation for search spaces and model configs. | Phase 1 |
| **Linter/Formatter**| `Ruff` | Ultra-fast Python linting and formatting. | Phase 1 |
| **Tracking** | `MLflow` | Full experiment lifecycle and model registry. | Phase 2 |
| **Versioning** | `DVC` | Data versioning and pipeline reproducibility. | Phase 2 |
| **Observability** | `Evidently AI` | Data drift detection and model performance monitoring. | Phase 3 |
| **Testing** | `Pytest` + `Deepchecks` | Unit testing and ML-specific validation suites. | Phase 3 |
| **Deployment** | `Docker` | Containerization for cloud-native scalability. | Phase 4 |
| **CI/CD** | `GitHub Actions` | Automated quality gates and deployment pipelines. | Phase 4 |

---

## Phase 1: Foundations & Industrialization
**Goal:** Transform the prototype into a structured, validated, and high-performance Python package.
### Requirements
*   Migrate from `requirements.txt` to `uv` and `pyproject.toml`.
*   Refactor notebook logic into a modular core library (`src/core`).
*   Implement Pydantic schemas for hyperparameter search space definitions.
*   Setup `Ruff` and `MyPy` for strict code quality.

## Phase 2: Lifecycle & Reproducibility
**Goal:** Ensure every experiment is traceable and datasets are versioned.
### Requirements
*   Integrate `MLflow` for tracking trials, metrics, and artifacts.
*   Initialize `DVC` to manage data lineage.
*   Create a CLI/API entry point to trigger tuning jobs.

## Phase 3: Observability & Trustworthy AI
**Goal:** Add layers of monitoring and validation to ensure model reliability.
### Requirements
*   Implement `Evidently AI` reports for Data Drift and Model Health.
*   Add SHAP/LIME for model explainability in the tuning results.
*   Automate ML validation tests (performance thresholds).

## Phase 4: Intelligence & Cloud-Native
**Goal:** Scale the system and integrate modern AI-driven workflows.
### Requirements
*   Dockerize the application for production deployment.
*   Setup CI/CD pipelines for automated testing and linting.
*   (Optional) Integrate an LLM Agent to suggest search space optimizations based on MLflow history.
