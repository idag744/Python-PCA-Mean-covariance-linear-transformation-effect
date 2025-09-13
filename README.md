# Affine Transformations and Statistics in Machine Learning

This project explores how **mean** and **covariance** of a dataset change under **affine transformations**.  
It provides clean, vectorized implementations in Python using NumPy, along with simple tests and visualizations.

## 🔑 Key Features
- Compute dataset **mean** and **covariance** (vectorized).
- Apply **affine transformations** and verify theoretical identities:
  - `m' = A m + b`
  - `S' = A S Aᵀ`
- Example visualization on the **Olivetti Faces dataset**.
- Organized into a **modular structure** with unit tests.

## 📂 Project Structure
affine-transforms-ml/
│
├── src/
│   ├── __init__.py
│   ├── stats.py          # mean, covariance
│   ├── transforms.py     # affine transformations
│
├── tests/
│   ├── test_stats.py
│   ├── test_transforms.py
│
├── notebooks/
│   └── demo.ipynb        # visualization with Olivetti dataset
│
├── requirements.txt
├── README.md
└── .gitignore

## 🚀 Getting Started

### Install dependencies
```bash
pip install -r requirements.txt
pytest
