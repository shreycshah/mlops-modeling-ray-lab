# Distributed Model Training with Ray

## Project Overview

This lab demonstrates how to accelerate machine learning model training by distributing work across multiple CPU cores using **Ray**. It compares sequential vs. parallel training of 20 `RandomForestRegressor` models across a hyperparameter sweep (`n_estimators` from 50 to 525 in steps of 25), measuring the wall-clock speedup from parallelization.

## Original Lab Reference

This lab is based on the [Original Lab Repository](https://github.com/raminmohammadi/MLOps/blob/main/Labs/Model_Development/Ray/Ray.ipynb) from the official Ray documentation, adapted for the IE 7374 MLOps course (Spring 2026).

## Changes Made

| | Original | This Lab |
|---|---|---|
| **Dataset** | California Housing (sklearn built-in) | Ames Housing via OpenML (`fetch_openml(name="house_prices", version=1)`) |
| **Features** | 8 numeric features, no preprocessing needed | 80 features (mixed numeric + categorical); label-encoded and NaN columns dropped — resulting in 77 features |
| **Model** | GradientBoostingRegressor | RandomForestRegressor (`n_jobs=1` to isolate Ray's benefit) |

## How to Run

1. Install dependencies:
   ```bash
   pip install scikit-learn pandas ray jupyter
   ```

2. Launch Jupyter and open the notebook:
   ```bash
   jupyter notebook src/Ray_Lab.ipynb
   ```

3. Run all cells top to bottom using **Kernel > Restart & Run All**, or execute each cell sequentially with `Shift+Enter`.

## Output

### Sequential Run (20 models, single-threaded)

```
n_estimators=50,  rmse=28944.34, took: 0.48 s
n_estimators=75,  rmse=28430.52, took: 0.61 s
...
n_estimators=525, rmse=28343.54, took: 4.23 s

Wall time: 47.2 s
Best model: rmse=28177.50, n_estimators=150
```

### Parallel Run (20 models, distributed via Ray)

```
Wall time: 14 s
Best model: rmse=28177.50, n_estimators=150
```

### Speedup

**~3.4x faster** with Ray parallel execution (47.2 s → 14 s). The exact speedup scales with the number of available CPU cores.