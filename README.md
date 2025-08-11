# Model Evaluation Report

This repository hosts a Streamlit application for interactive analysis of object detection models in citrus groves. It aggregates metrics from multiple evaluation CSV files and provides dashboards for comparing precision, recall, and other performance metrics across models and contexts.

## Dependencies

- Python 3.11+
- Streamlit
- Pandas
- Plotly
- Openpyxl
- NumPy

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Launching the Streamlit App

1. Place your evaluation CSV files in the `data/` directory (see format below).
2. Install dependencies (see above).
3. Start the app:

```bash
streamlit run app.py
```

The app will open in your browser, offering filters, dashboards, and export options.

## Expected CSV File Format

Each model's evaluation should be stored in a separate file named `eval_model_<model_number>_Sheet1.csv` and located in the `data/` directory. The CSV must contain the following columns:

| Column | Description |
|--------|-------------|
| `compagnie` | Year of data collection (renamed to `year`) |
| `Domaine` | Domaine or region (renamed to `domaine`) |
| `Porte-greffe` | Rootstock type (renamed to `porte_greffe`) |
| `parcelle` | Plot identifier |
| `Filename` | Image filename |
| `True_count` | Number of ground truth objects |
| `detect_count` | Number of detected objects |
| `TP` | True positives |
| `FP` | False positives |
| `FN` | False negatives |
| `Precision` | Model precision value |
| `Recall` | Model recall value |

During loading, the app merges all model files and generates columns such as `precision_<model_number>`, `recall_<model_number>`, `tp_<model_number>`, `fp_<model_number>`, and `fn_<model_number>`.


## Purpose

Use this app to evaluate and compare object detection models, explore performance trends across years and domains, conduct error analysis, and export filtered data for further study.
