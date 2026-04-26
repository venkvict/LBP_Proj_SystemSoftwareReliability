# Software Reliability Analysis & Prediction

This project provides a comprehensive toolkit for analyzing and predicting software reliability using both traditional **NHPP (Non-Homogeneous Poisson Process)** models and modern **Machine Learning** approaches.

## 🚀 Features

- **Standard NHPP Models**: Implementation of classic models like Goel-Okumoto, Delayed S-shaped, and Inflection S-shaped.
- **ML-Based Prediction**: Using MLP, SVR, and Tree-based regressors for flexible reliability forecasting.
- **Trend Analysis**: Integrated Laplace trend test to detect reliability growth or degradation.
- **Evaluation Metrics**: Support for AIC, BIC, MSE, and RMSE for model comparison.
- **Modular Structure**: Clean, production-ready code organized into reusable modules.

## 📂 Project Structure

```text
LBP_Proj_SystemSoftwareReliability/
├── data/                       # Datasets
│   ├── raw/                    # Original datasets
│   └── processed/              # Processed datasets for modeling
├── notebooks/                  # Original notebooks for reference
│   ├── 01_nhpp_analysis.ipynb
│   └── 02_ml_prediction.ipynb
├── src/                        # Source code
│   ├── models/                 # Model implementations (NHPP & ML)
│   ├── utils/                  # Helper functions (data loading, metrics)
│   └── main.py                 # Entry point to run experiments
├── tests/                      # Unit tests
├── requirements.txt            # Project dependencies
└── README.md                   # Project documentation
```

## 🛠️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/software-reliability.git
   cd software-reliability
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

## 📊 Datasets

This project is designed to work with software failure datasets. You can use the following Kaggle datasets:
1. [Blue Mountain Supercomputer Failures](https://www.kaggle.com/datasets/proshun/blue-mountain-supercomputer-monthly-failures)
2. [Software Failure Data](https://www.kaggle.com/datasets/atulanandjha/software-failure-data)

### Setup Datasets
1. Download the CSV files from Kaggle.
2. Place them in the `data/raw/` directory.
3. Update the path in your scripts or use the `load_failure_data` utility.

## 📈 Usage

### Running the Analysis
To run a complete demonstration using synthetic data:
```bash
python src/main.py
```

### Running on Kaggle Data
To run the analysis on a real dataset, modify `src/main.py` or create a new script:
```python
from src.utils.data_loader import load_failure_data
from src.models.nhpp_models import get_goel_okumoto

# Load real data from Kaggle
data = load_failure_data("data/raw/blue_mountain_failures.csv")

# Proceed with fitting and analysis
model = get_goel_okumoto()
model.fit(data)
```

## 📚 References
- Pham, H. (2006). *System Software Reliability*. Springer.
- Chapter 6: NHPP Software Reliability Models.

## 📄 License
MIT License
