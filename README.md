# System Software Reliability Modeling

This project explores various models for predicting and analyzing system software reliability, including traditional statistical models and modern machine learning approaches.

## Project Structure

```text
LBP_Proj_SystemSoftwareReliability/
├── data/                      # Raw and processed datasets
├── notebooks/                 # Interactive analysis & research
│   ├── NHPP_Models.ipynb      
│   ├── MixtureModels.ipynb    
│   └── MLbased.ipynb          
├── src/                       # Production-grade pipeline code
│   ├── pipeline/              # Core ML lifecycle modules
│   │   ├── data_loader.py     # Data ingestion logic
│   │   ├── trainer.py         # Model training wrappers
│   │   └── evaluator.py       # Metric calculations
│   ├── utils/                 # Helper functions
│   │   └── visualizer.py      # Plotting & reporting tools
│   └── main.py                # Pipeline entry point
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```

## Modular Pipeline (`src/`)

For production readiness, the logic from the notebooks has been modularized into a cleaner structure:

- **`data_loader.py`**: Handles loading of failure datasets and time-series sorting.
- **`trainer.py`**: Provides a unified interface to initialize NHPP and ML models.
- **`evaluator.py`**: Standardized metrics for both regression (RMSE, R2) and classification (Accuracy, F1).
- **`visualizer.py`**: Reusable plotting functions for reliability growth curves.

## Models Implemented

### 1. NHPP Models (`NHPP_Models.ipynb`)
This notebook focuses on **Non-Homogeneous Poisson Process (NHPP)** models, which are widely used for software reliability growth modeling (SRGM). It includes implementation and optimization of failure rate parameters.

### 2. Mixture Models (`MixtureModels.ipynb`)
This notebook implements **Survival Analysis** using various statistical distributions:
- **Weibull Distribution**: Captures varying failure rates over time.
- **Lognormal Distribution**: Useful for modeling repair times and certain failure patterns.
- **Gamma Distribution**: A flexible model for time-to-failure data.

### 3. Machine Learning Models (`MLbased.ipynb`)
This notebook compares several ML techniques for reliability prediction:
- **Regression Tasks**: Predicting future failure counts.
    - Artificial Neural Networks (ANN/Backpropagation)
    - Support Vector Regression (SVR)
    - Decision Tree Regressors
- **Classification Tasks**: Categorizing system risk levels (Low vs. High Risk).
    - Decision Tree Classifier
    - Bagging Classifier
    - Support Vector Classifier (SVC)

## Getting Started

### Prerequisites
Ensure you have Python installed. It is recommended to use a virtual environment.

### Installation
Install the required dependencies using pip:

```bash
pip install -r requirements.txt
```

### Usage
Launch Jupyter Notebook or JupyterLab to explore the implementations:

```bash
jupyter notebook
```

## Data
The models are tested on failure datasets such as the Blue Mountain supercomputer failure logs. Ensure the data paths in the notebooks are updated to point to your local data sources if necessary.
