# DriftSense: Applied Materials Hackathon 2026

## Overview
DriftSense is a Python-based computer vision solution for detecting and localizing wafer feature drift, alignment shift, and defect anomalies in synthetic semiconductor manufacturing image data.

## Project Structure
```text
DriftSense/
├── .venv/                   # Python virtual environment
├── data/                    # Programmatically generated synthetic wafer images
├── src/                     # Core algorithms and helper modules
├── generate_dataset.py      # Programmatic synthetic dataset generator
├── localize.py              # Feature localization & drift detection module
├── evaluate.py              # Model / algorithm evaluation pipeline
├── requirements.txt         # Project dependencies
└── README.md                # Documentation
```

## Setup & Environment
1. Activate Virtual Environment:
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```
2. Install Dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
- **Generate Synthetic Dataset**:
  ```bash
  python generate_dataset.py
  ```
- **Run Localization / Drift Detection**:
  ```bash
  python localize.py
  ```
- **Evaluate Results**:
  ```bash
  python evaluate.py
  ```

## Principles
- All synthetic semiconductor data is generated programmatically without proprietary or external dataset downloads.
