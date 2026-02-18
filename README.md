# NYSE Stock Forecasting with LSTM

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" />
  <img src="https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white" />
  <img src="https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white" />
  <img src="https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" />
</p>

---

## 🚀 Project Overview

This repository features a **comprehensive, end-to-end LSTM pipeline** designed for forecasting New York Stock Exchange (NYSE) data. It is specifically optimized for Kaggle environments, providing a seamless flow from raw data ingestion to production-ready model artifacts.

### 🎯 Key Highlights
- **End-to-End Workflow**: Covers data validation, EDA, preprocessing, training, and artifact export.
- **Interactive EDA**: Utilizes `Plotly` for high-quality, interactive financial visualizations.
- **Robust Preprocessing**: Implements sliding window sequence construction with automated MinMax normalization.
- **Deep Learning Architecture**: Features a stacked two-layer LSTM model with dropout regularization for time-series stability.
- **Production Ready**: Automatically exports model checkpoints, scalars, and metadata for immediate deployment.

---

## 📊 Implementation Flow

The pipeline follows a rigorous data science lifecycle to ensure model reliability and performance:

```mermaid
flowchart LR
    A["📥 Load Data"] --> B["🔍 Validate"]
    B --> C["📉 Visual EDA"]
    C --> D["⚙️ Preprocess"]
    D --> E["🧠 LSTM Train"]
    E --> F["🧪 Evaluate"]
    F --> G["📦 Artifacts"]

    style A fill:#1e293b,stroke:#3b82f6,color:#fff
    style G fill:#064e3b,stroke:#10b981,color:#fff
```

### 1. Exploratory Data Analysis (EDA)
The notebook generates deep insights using:
- **Candlestick Charts**: Visualizing price action and volume distribution.
- **Moving Averages**: Analyzing trends with 20-day and 60-day window averages.
- **Return Analysis**: Histograms of daily returns to understand market volatility.
- **Feature Correlation**: Heatmaps identifying relationships between OHLC (Open, High, Low, Close) features.

### 2. Neural Architecture
The model leverages a **Stacked LSTM** structure optimized for capturing both short-term noise and long-term temporal dependencies:

```mermaid
flowchart TB
    Input["Input Window (29 Days)"] --> LSTM1["LSTM Layer 1 (96 Units)"]
    LSTM1 --> Drop1["Dropout (15%)"]
    Drop1 --> LSTM2["LSTM Layer 2 (96 Units)"]
    LSTM2 --> Drop2["Dropout (15%)"]
    Drop2 --> Dense["Dense Output Layer (4 Neurons)"]
    Dense --> Output["Next-Step OHLC Forecast"]

    classDef layer fill:#f8fafc,stroke:#334155,stroke-width:2px;
    classDef io fill:#eff6ff,stroke:#2563eb,stroke-width:2px;
    class LSTM1,LSTM2,Drop1,Drop2,Dense layer;
    class Input,Output io;
```

| Hyperparameter | Value |
| :--- | :--- |
| **Lookback Window** | 29 Days |
| **LSTM Units** | 96 per layer |
| **Dropout Rate** | 0.15 |
| **Loss Function** | Mean Squared Error (MSE) |
| **Optimizer** | Adam (LR: 0.001) |
| **Batch Size** | 64 |
| **Features** | 4 (Open, High, Low, Close) |

---

## 🛠️ Getting Started

### Prerequisites
- Python 3.8+
- TensorFlow 2.x
- Pandas, NumPy, Scikit-learn, Plotly

### Dataset
This project uses the **Kaggle NYSE Dataset**:
- **Dataset ID**: `dgawlik/nyse`
- **Path**: `/kaggle/input/datasets/dgawlik/nyse/prices-split-adjusted.csv`

### How to Run
1.  Upload `nyse-stock-forecasting-fresh-flow.ipynb` to a Kaggle Notebook.
2.  Add the `dgawlik/nyse` dataset to your environment.
3.  (Optional) Modify the `TICKER` variable in the Configuration cell to forecast a specific stock (Default: `EQIX`).
4.  Execute all cells.

---

## 📦 Deployment & Artifacts

Upon completion, the pipeline exports all necessary components to `/kaggle/working/nyse_lstm_artifacts`:

| Artifact | Purpose |
| :--- | :--- |
| `nyse_lstm_ohlc.keras` | The final trained model file. |
| `best_lstm.keras` | The best performing checkpoint (lowest validation loss). |
| `config.json` | Stores hyperparameters and feature mapping. |
| `preprocess_meta.pkl` | Crucial metadata for inverse scaling and inference consistency. |

### Inference Snippet
```python
# Quick example of how to use the saved model
from tensorflow.keras.models import load_model
model = load_model('nyse_lstm_ohlc.keras')
prediction = model.predict(normalized_input_window)
```

---

## 🔬 Experimental Comparison (RNN vs GRU vs LSTM)

While this project defaults to LSTM for its stability with long-term dependencies, here is a conceptual breakdown of why it was chosen:

- **Vanilla RNN**: Simple and fast, but prone to **Vanishing Gradients** in longer sequences.
- **GRU (Gated Recurrent Unit)**: Efficient and often faster than LSTM; great for medium-length patterns.
- **LSTM (Long Short-Term Memory)**: Includes an explicit "Cell State" for fine-grained memory control, making it the most robust for volatile stock sequences.

---

<p align="center">
  <i>Developed for professional-grade stock forecasting and deployment research.</i>
</p>
