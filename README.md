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


### 2. Neural Architecture with Stacked LSTM
The model utilizes a **Stacked LSTM** architecture, designed to process sequential data with multiple levels of abstraction. 

The architecture consists of:
- **Input Channels**: 5 distinct market indicators (Open, High, Low, Close, Volume).
- **Layer Stacking**: Two sequential LSTM layers to extract both short-term volatility and long-term trends.
- **Output**: A single regression output predicting the next trading day's closing price.

```mermaid
graph TD
    subgraph Input ["Input Layer"]
        direction TB
        X1((x₁)) --- X2((x₂)) --- X3((x₃)) --- X4((x₄)) --- X5((x₅)) --- Xdots[...]
    end

    subgraph LSTM1 ["1st LSTM (96 Units)"]
        direction TB
        H11((h¹₁)) --- H12((h¹₂)) --- H13((h¹₃)) --- H14((h¹₄)) --- H15((h¹₅)) --- H1dots[...]
    end

    subgraph LSTM2 ["2nd LSTM (96 Units)"]
        direction TB
        H21((h²₁)) --- H22((h²₂)) --- H23((h²₃)) --- H24((h²₄)) --- H25((h²₅)) --- H2dots[...]
    end

    subgraph LSTM3 ["3rd LSTM (96 Units)"]
        direction TB
        H31((h³₁)) --- H32((h³₂)) --- H33((h³₃)) --- H34((h³₄)) --- H35((h³₅)) --- H3dots[...]
    end

    Target["🎯 Target:<br/>Close Price"]

    %% Recurrent connections (horizontal arrows within each layer)
    X1 -.-> X2 -.-> X3 -.-> X4 -.-> X5
    H11 -.-> H12 -.-> H13 -.-> H14 -.-> H15
    H21 -.-> H22 -.-> H23 -.-> H24 -.-> H25
    H31 -.-> H32 -.-> H33 -.-> H34 -.-> H35

    %% Vertical feedforward between layers (same timestep)
    X1 --> H11
    X2 --> H12
    X3 --> H13
    X4 --> H14
    X5 --> H15
    H11 --> H21
    H12 --> H22
    H13 --> H23
    H14 --> H24
    H15 --> H25
    H21 --> H31
    H22 --> H32
    H23 --> H33
    H24 --> H34
    H25 --> H35

    %% Final output connection
    H35 --> Target

    %% Exact color matching from your reference + white BG
    classDef inputStyle fill:#4fc3f7,stroke:#0277bd,stroke-width:3px,color:#fff
    classDef lstm1Style fill:#2196f3,stroke:#1976d2,stroke-width:3px,color:#fff
    classDef lstm2Style fill:#ffeb3b,stroke:#fbc02d,stroke-width:3px,color:#000
    classDef lstm3Style fill:#f44336,stroke:#d32f2f,stroke-width:3px,color:#fff
    classDef targetStyle fill:#66bb6a,stroke:#388e3c,stroke-width:3px,color:#fff

    class X1,X2,X3,X4,X5,Xdots inputStyle
    class H11,H12,H13,H14,H15,H1dots lstm1Style
    class H21,H22,H23,H24,H25,H2dots lstm2Style
    class H31,H32,H33,H34,H35,H3dots lstm3Style
    class Target targetStyle
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
