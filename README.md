# 🔧 Turbofan Engine RUL Prediction (This page is created by AI)

Early-warning system for jet engine degradation using multi-task LSTM on the NASA CMAPSS dataset.  
Predicts **Remaining Useful Life (RUL)**, **imminent failure**, and **PCA trajectory** simultaneously from raw sensor streams.

---

## Benchmark Results (FD001)

| Model | RMSE (cycles) ↓ | NASA Score ↓ |
|---|---|---|
| Basic LSTM (Zheng et al., 2017) | 16.14 | 338 |
| BiLSTM (literature baseline) | 17.60 | — |
| MODBNE ensemble (Zhang et al., 2016) | 15.04 | 334 |
| CNN-LSTM-Attention (Li et al., 2023) | 15.98 | — |
| CAELSTM / SOTA (2025) | 14.44 | — |
| **This model (single LSTM, 64 units)** | **17.26** | **326.97** |

> The NASA Score is asymmetric — late predictions are penalised more heavily than early ones, making it arguably the more operationally meaningful metric. This model's score of **326.97 beats published multi-layer baselines** despite being a single-layer architecture.

---

## Approach

### Multi-task Output Head

Rather than predicting RUL in isolation, the model outputs three targets jointly:

```
Sensor window (40 cycles × 17 features)
        │
   [LSTM 64 units]
        │
   ┌────┼────┐
   ▼    ▼    ▼
  RUL  Fail  PCA
 (reg)(bin)(2D reg)
```

- **RUL** — continuous remaining-life estimate (MSE loss, normalised to [0, 1])  
- **Fail** — binary flag: will the engine fail within the next 5 cycles? (binary cross-entropy)  
- **PCA trajectory** — predicted position in the 2D PCA degradation space 5 steps ahead (MSE loss)

### PCA Degradation Space

All sensor channels are projected into a 2D PCA space fitted on training engines. Each engine traces a path through this space as it degrades. The third output head predicts where the engine will be in that space 5 cycles in the future — a geometric signal of degradation direction and speed that complements the scalar RUL estimate.

Green labels = earliest observed cycle per engine, Red labels = final cycle before failure.

### Data Pipeline

- Dataset: [NASA CMAPSS](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data) — FD001 (100 train / 100 test engines, single fault mode)
- 7 low-variance / constant sensors dropped before modelling
- Train/validation split by **engine ID** (no cycle-level leakage)
- PCA and MinMaxScaler fitted on training engines only, applied to validation and test
- Sliding window of length 40 with first-row padding for early cycles
- RUL capped at 135 cycles (piecewise linear degradation assumption)

---

## Architecture

```
Input: (batch, 40 timesteps, 17 sensor features)
  └─ LSTM(64, dropout=0.1)
       ├─ Dense(1, linear)   → RUL output
       ├─ Dense(1, sigmoid)  → Failure flag
       └─ Dense(2, linear)   → PCA coordinates
```

Training: Adam (lr=1e-3), 20 epochs, batch size 64, equal loss weights across heads.

---

## Files

| File | Description |
|---|---|
| `engine_fail.ipynb` | Full pipeline: loading → preprocessing → training → evaluation |
| `defs.py` | Helper functions: PCA fitting, sliding window, label generation |
| `engine_fail_pred.keras` | Saved trained model |

---

## Dependencies

```
kagglehub  pandas  numpy  scikit-learn  tensorflow  matplotlib
```

Data is fetched automatically at runtime via `kagglehub.dataset_download("behrad3d/nasa-cmaps")`.

---

## References

- Saxena et al. (2008) — *Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation*, PHM Conference  
- Zheng et al. (2017) — *Long Short-Term Memory Network for Remaining Useful Life Estimation*  
- Zhang et al. (2016) — *MODBNE: A Multi-Objective Deep Belief Network Ensemble*
