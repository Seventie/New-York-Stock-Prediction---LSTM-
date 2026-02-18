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
graph LR

    %% ── Input Layer Row ──────────────────────────────────────
    subgraph IL ["Input Layer  |  Shape (30, 5)"]
        direction LR
        X1["X₁\nOpen,High\nLow,Close\nVolume"]
        X2["X₂\nOpen,High\nLow,Close\nVolume"]
        X3["X₃\nOpen,High\nLow,Close\nVolume"]
        Xd["  ...  "]
        XN["X₃₀\nOpen,High\nLow,Close\nVolume"]
        X1 --> X2 --> X3 --> Xd --> XN
    end

    %% ── LSTM Layer 1 Row ─────────────────────────────────────
    subgraph L1 ["LSTM Layer 1  |  96 Units  |  return_sequences = True  |  Output: (30, 96)"]
        direction LR
        H11(["h¹₁"])
        H12(["h¹₂"])
        H13(["h¹₃"])
        H1d["  ...  "]
        H1N(["h¹₃₀"])
        H11 --> H12 --> H13 --> H1d --> H1N
    end

    %% ── LSTM Layer 2 Row ─────────────────────────────────────
    subgraph L2 ["LSTM Layer 2  |  96 Units  |  return_sequences = False  |  Output: (96,)"]
        direction LR
        H21(["h²₁"])
        H22(["h²₂"])
        H23(["h²₃"])
        H2d["  ...  "]
        H2N(["h²₃₀"])
        H21 --> H22 --> H23 --> H2d --> H2N
    end

    %% ── Dense Layer Row ──────────────────────────────────────
    subgraph D1 ["Dense Layer  |  25 Neurons  |  ReLU Activation  |  Output: (25,)"]
        direction LR
        DN1(["d₁"])
        DN2(["d₂"])
        DN3(["d₃"])
        DNd["  ...  "]
        DN25(["d₂₅"])
    end

    %% ── Output Layer ─────────────────────────────────────────
    subgraph OUT ["Output Layer  |  1 Neuron  |  Linear  |  Output: (1,)"]
        direction LR
        PRED["Predicted\nClose Price"]
    end

    %% ── Vertical column connections (same timestep) ──────────
    X1  --> H11
    X2  --> H12
    X3  --> H13
    XN  --> H1N

    H11 --> H21
    H12 --> H22
    H13 --> H23
    H1N --> H2N

    %% Only final hidden state feeds forward (many-to-one)
    H2N --> DN1
    H2N --> DN2
    H2N --> DN3
    H2N --> DN25

    DN1  --> PRED
    DN2  --> PRED
    DN3  --> PRED
    DN25 --> PRED

    %% ── Professional muted palette ───────────────────────────
    style IL  fill:#f5f5f0,stroke:#9e9e9e,stroke-width:2px,color:#212121
    style L1  fill:#e8edf2,stroke:#78909c,stroke-width:2px,color:#212121
    style L2  fill:#edf2f7,stroke:#607d8b,stroke-width:2px,color:#212121
    style D1  fill:#f7f7f2,stroke:#bdbdbd,stroke-width:2px,color:#212121
    style OUT fill:#f1f8e9,stroke:#aed581,stroke-width:2px,color:#1b5e20

    classDef inputBox  fill:#ececec,stroke:#9e9e9e,stroke-width:1.5px,color:#424242,font-size:13px
    classDef lstm1Cell fill:#cfd8dc,stroke:#78909c,stroke-width:1.5px,color:#263238,font-size:15px
    classDef lstm2Cell fill:#b0bec5,stroke:#546e7a,stroke-width:1.5px,color:#263238,font-size:15px
    classDef denseCell fill:#e0e0da,stroke:#9e9e9e,stroke-width:1.5px,color:#424242,font-size:15px
    classDef outCell   fill:#dcedc8,stroke:#7cb342,stroke-width:2px,color:#1b5e20,font-size:14px
    classDef dotCell   fill:#fafafa,stroke:#e0e0e0,stroke-width:1px,color:#9e9e9e,font-size:16px

    class X1,X2,X3,XN inputBox
    class H11,H12,H13,H1N lstm1Cell
    class H21,H22,H23,H2N lstm2Cell
    class DN1,DN2,DN3,DN25 denseCell
    class PRED outCell
    class Xd,H1d,H2d,DNd dotCell


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
