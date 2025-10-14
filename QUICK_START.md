# 🚀 Quick Start Guide - 39-Bus Power System Experiments

## 🎯 Run Everything in 3 Steps

### Step 1: Start MLflow UI (Optional but Recommended)
```bash
mlflow ui
```
Keep this terminal open and visit http://localhost:5000 in your browser.

### Step 2: Run All Experiments
```bash
python run_experiments.py
```
This will run all 24 experiments automatically. Expected time: 2-10 hours.


## ⚡ Quick Commands

| Command | What It Does |
|---------|--------------|
| `python run_experiments.py` | Run all 24 experiments |
| `python run_experiments.py --exp 5` | Run only experiment 5 |
| `python run_experiments.py --start 0 --end 5` | Run experiments 0-5 |
| `python run_experiments.py --list` | List all experiments |
| `python compare_results.py` | Generate comparison plots |
| `mlflow ui` | Open MLflow tracking UI |
| `python main.py` | Run single experiment (legacy) |

