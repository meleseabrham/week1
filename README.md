# Nova Financial Insights – Week 1

This repository houses the deliverables for the Nova Financial Insights Week 1 challenge.  
The goal is to explore how financial news sentiment influences equity price movements.

## Project Goals
- Build a reproducible analytics environment.
- Engineer sentiment scores from financial headlines.
- Link textual sentiment, publisher behavior, and publication timing to daily stock returns.

## Repo Layout
- `src/`: reusable Python modules (data loading, processing, analytics).
- `scripts/`: executable utilities for batch jobs or CLI tooling.
- `notebooks/`: exploratory data analysis and storytelling notebooks.
- `tests/`: automated tests to keep code quality high.
- `.github/workflows/`: CI configuration (linting, tests).

## Getting Started
1. Create a Python 3.11+ virtual environment.
2. Install dependencies: `pip install -r requirements.txt`.
3. Place the Financial News and Stock Price Integration Dataset (FNSPID) under `data/raw/`.
4. Run exploratory stats: `python scripts/run_eda.py`.
5. Run TA/LIB + PyNance indicators: `python scripts/run_technical_analysis.py`.

Outputs are written to `data/processed/eda/`, `data/processed/technical_metrics/`, and chart images under `reports/figures/`.

## Status
- [x] Data ingestion
- [x] Sentiment/EDA automation
- [x] Technical indicator pipeline
- [ ] Predictive modeling
- [ ] Reporting polish

Pull requests welcome!
