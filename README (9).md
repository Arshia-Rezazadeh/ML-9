# Bitcoin Price Prediction & Image Classification

This repository contains two independent machine learning experiments developed as part of a coursework assignment:

1. **Bitcoin Price Prediction** — comparing a gradient-boosted tree model (XGBoost) against a simple feed-forward neural network (MLP) on both a classification and a regression task.
2. **Image Classification** — a convolutional neural network (CNN) trained from scratch on the Caltech-101 dataset.

---

## Project Structure

```
.
├── btc_v1.ipynb         # Baseline: XGBoost classifier + regressor
├── btc_v2.ipynb         # Extended: XGBoost vs. MLP benchmark comparison
├── images.ipynb         # CNN trained on Caltech-101
├── bitcoin_dataset.csv  # OHLCV data (not included, see Data section)
└── README.md
```

---

## 1. Bitcoin Price Prediction

### Objective

Predict, using a given day's OHLCV (Open, High, Low, Close, Volume) values, two targets for the **next** trading day:

- **Classification** — whether the closing price will be higher than the opening price (binary: up/down).
- **Regression** — the actual closing price.

### Data

- Source: `bitcoin_dataset.csv`, daily OHLCV data spanning 1,461 days starting January 1, 2020.
- Features used: `Open`, `High`, `Low`, `Close`, `Volume` (same-day values).
- Targets are shifted by one day (`shift(-1)`) to represent next-day outcomes.
- Train/test split: 80/20, chronological (no shuffling), to avoid look-ahead bias.

### Notebooks

| Notebook | Description |
|---|---|
| `btc_v1.ipynb` | Initial baseline using only XGBoost (`XGBClassifier`, `XGBRegressor`) with default hyperparameters. |
| `btc_v2.ipynb` | Adds a Keras `Sequential` MLP (16 → 8 → output) for both tasks, with feature/target standardization (`StandardScaler`), and produces a side-by-side benchmark report against the XGBoost baseline. |

### Results

**Classification (next-day direction)**

| Model | Training Time (s) | Accuracy |
|---|---|---|
| XGBoost | ~0.34–0.43 | 52.74% |
| MLP | ~4.15 | 50.68% |

**Regression (next-day close price)**

| Model | Training Time (s) | RMSE |
|---|---|---|
| XGBoost | ~0.17 | $1,675.55 |
| MLP | ~3.88 | $785.52 |

### Interpretation

- For **direction classification**, both models perform close to random chance (~50%). This is expected: same-day OHLCV values carry very little predictive signal for next-day price direction in a highly efficient, noisy market like Bitcoin.
- For **price regression**, the MLP substantially outperforms XGBoost. This is largely attributable to feature/target scaling — XGBoost trained on unscaled data appears more sensitive to large price swings, while the standardized MLP generalizes better on this metric.
- Overall, this highlights a common trade-off: XGBoost trains orders of magnitude faster and requires no preprocessing, while the MLP needs more compute and scaling but yields better regression accuracy here.

### Possible Improvements

- Add technical indicators (RSI, moving averages, MACD, Bollinger Bands).
- Include multi-day lag features instead of only same-day values.
- Try walk-forward validation instead of a single chronological split.
- Tune hyperparameters for both models (currently using defaults).

---

## 2. Image Classification (Caltech-101)

### Objective

Classify images into their respective categories using a CNN trained from scratch (no transfer learning) on the Caltech-101 dataset.

### Architecture

A compact custom CNN (`StudentCNN`):

```
Conv2d(3→32) → ReLU → MaxPool
Conv2d(32→64) → ReLU → MaxPool
Conv2d(64→128) → ReLU → MaxPool
Flatten → Linear(128*16*16 → 256) → ReLU → Dropout(0.5) → Linear(256 → num_classes)
```

- Input size: 128×128 RGB
- Data augmentation (training set only): random horizontal flip, random rotation (±15°)
- Optimizer: Adam (lr = 0.001, weight decay = 1e-4)
- Loss: Cross-entropy
- Train/validation split: 80/20
- Epochs: 10

### Results

Training and validation loss/accuracy curves are plotted at the end of the notebook, along with the final validation accuracy after 10 epochs.

---

## Requirements

```bash
pip install pandas scikit-learn xgboost tensorflow torch torchvision matplotlib pillow
```

A CUDA-capable GPU is recommended for the image classification notebook but not required.

## Data Setup

- Place `bitcoin_dataset.csv` in the same directory as `btc_v1.ipynb` and `btc_v2.ipynb`.
- Place the Caltech-101 dataset in a folder named `caltech-101` or `101_ObjectCategories` in the same directory as `images.ipynb`.

## Usage

Open the notebooks in Jupyter and run all cells sequentially:

```bash
jupyter notebook
```

---

## Notes

This project was developed incrementally across two course weeks: `btc_v1.ipynb` represents the Week 8 baseline (classical ML), and `btc_v2.ipynb` extends it in Week 9 with a neural network comparison. `images.ipynb` is a separate, standalone deep learning exercise on image data.
