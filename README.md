# Nova Financial Insights – Week 1

This repository houses the deliverables for the Nova Financial Insights Week 1 challenge.  
The goal is to explore how financial news sentiment influences equity price movements.

## Project Goals
- Build a reproducible analytics environment.
- Engineer sentiment scores from financial headlines.
- Link textual sentiment, publisher behavior, and publication timing to daily stock returns.

## Repository Structure
- `src/`: reusable Python modules (data loading, processing, analytics).
- `scripts/`: executable utilities for batch jobs or CLI tooling.
  - `run_eda.py`: Complete EDA pipeline for financial news dataset
  - `run_technical_analysis.py`: Technical indicator computation and visualization
- `notebooks/`: exploratory data analysis and storytelling notebooks.
- `data/`: 
  - `raw/`: Original dataset files
  - `processed/eda/`: EDA outputs (statistics, counts, topic models)
  - `processed/prices/`: Stock price data (AAPL, AMZN, GOOG, META, MSFT, NVDA)
  - `processed/technical_metrics/`: Technical indicator time series
- `reports/`: 
  - `figures/`: Visualization charts for technical analysis
  - `interim_report.md`: Detailed progress report
- `tests/`: automated tests to keep code quality high.

## Implementation Details

### Task 1: Exploratory Data Analysis (Completed)
The EDA pipeline (`scripts/run_eda.py`) processes 1.4M+ financial news headlines through:

1. **Data Cleaning & Feature Engineering**:
   - UTC-aware date parsing with mixed format handling
   - Headline length metrics (character and word counts)
   - Publisher domain extraction from email formats
   - Temporal features (hour, day of week, date)

2. **Analytical Modules**:
   - Descriptive statistics for headline characteristics
   - Publisher distribution analysis (1,036 unique publishers)
   - Time series analysis (daily, hourly, weekday patterns)
   - Topic modeling using LDA (6 topics identified)

3. **Key Findings**:
   - Mean headline length: 73 characters (12.4 words)
   - Top publishers account for 65% of articles
   - Strong weekday patterns (mid-week peak, low weekend activity)
   - Six thematic clusters identified (ratings, earnings, announcements, market movers, ETFs, financial metrics)

### Task 2: Technical Analysis (Completed)
The technical analysis pipeline (`scripts/run_technical_analysis.py`) computes indicators for 6 technology stocks:

1. **TA-Lib Indicators**:
   - SMA (20-period): Short-term trend
   - EMA (50-period): Medium-term trend
   - RSI (14-period): Momentum oscillator
   - MACD (12, 26, 9): Trend-following momentum
   - ATR (14-period): Volatility measure

2. **PyNance Indicators**:
   - PN_GROWTH_5: 5-session growth rate
   - PN_VOL_20: 20-period volatility
   - PN_MOVAVE_10: 10-period moving average

3. **Visualization**:
   - Multi-panel candlestick charts with overlay indicators
   - RSI and MACD subplots
   - Last 180 trading days displayed

## Getting Started

### Prerequisites
- Python 3.11+
- Virtual environment (recommended)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/meleseabrham/week1.git
   cd week1
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Linux/Mac:
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Place the Financial News and Stock Price Integration Dataset (FNSPID) under `data/raw/`.

### Running the Analysis

1. **Run EDA Pipeline**:
   ```bash
   python scripts/run_eda.py
   ```
   Outputs: `data/processed/eda/*.csv` and `topic_keywords.json`

2. **Run Technical Analysis**:
   ```bash
   python scripts/run_technical_analysis.py
   ```
   Outputs: `data/processed/technical_metrics/*.csv` and `reports/figures/*.png`

## Output Files

### EDA Outputs (`data/processed/eda/`)
- `headline_length_stats.csv`: Descriptive statistics
- `publisher_article_counts.csv`: Article counts by publisher
- `publisher_domain_counts.csv`: Article counts by domain
- `daily_publication_counts.csv`: Daily time series
- `hourly_publication_counts.csv`: Hourly distribution (UTC)
- `weekday_publication_counts.csv`: Day-of-week distribution
- `topic_keywords.json`: LDA topic keywords

### Technical Analysis Outputs
- `data/processed/technical_metrics/{TICKER}_technicals.csv`: Full time series with indicators
- `data/processed/technical_metrics/technical_summary.csv`: Latest snapshot
- `reports/figures/{TICKER}_technicals.png`: Visualization charts

## Documentation

For detailed progress and findings, see:
- **[Interim Report](reports/interim_report.md)**: Comprehensive analysis of Task 1 and Task 2 progress, including methodology, findings, challenges, and next steps.

## Contributions

This repository represents original work completed for the Nova Financial Insights Week 1 challenge:

- **Data Processing**: Custom pipelines for financial news and stock price data
- **EDA Implementation**: Complete exploratory analysis with topic modeling
- **Technical Analysis**: Multi-indicator computation using TA-Lib and PyNance
- **Visualization**: Automated chart generation for technical indicators
- **Documentation**: Comprehensive reporting and code documentation

## Status
- [x] Data ingestion
- [x] EDA automation (Task 1)
- [x] Technical indicator pipeline (Task 2)
- [ ] Sentiment scoring integration
- [ ] Predictive modeling
- [ ] Final reporting

## License

This project is part of the Nova Financial Insights challenge.

## Repository Link

**GitHub Repository**: https://github.com/meleseabrham/week1.git
