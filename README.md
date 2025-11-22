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
  - `01_eda_analysis.ipynb`: Comprehensive EDA with statistical analysis and visualizations
  - `02_quantitative_analysis.ipynb`: Technical analysis using TA-Lib and PyNance with visualizations
  - `03_sentiment_correlation.ipynb`: Sentiment analysis and correlation with stock movements
- `data/`: 
  - `raw/`: Original dataset files
  - `processed/eda/`: EDA outputs (statistics, counts, topic models)
  - `processed/prices/`: Stock price data (AAPL, AMZN, GOOG, META, MSFT, NVDA)
  - `processed/technical_metrics/`: Technical indicator time series
  - `processed/sentiment_correlation/`: Sentiment scores and correlation analysis results
- `reports/`: 
  - `figures/`: Visualization charts for technical analysis
  - `interim_report.md`: Detailed progress report
- `tests/`: automated tests to keep code quality high.

## Implementation Details

### Task 1: Exploratory Data Analysis (Completed ✅)

The EDA pipeline (`scripts/run_eda.py`) performs comprehensive analysis on 1.4M+ financial news headlines:

#### 1. **Descriptive Statistics & Distribution Analysis**:
   - Basic statistics (mean, median, std, min, max) for headline lengths
   - **Statistical Distribution Testing**: Normality tests using D'Agostino-Pearson test
   - **Skewness & Kurtosis**: Measures of distribution shape
   - **Q-Q Plots**: Visual normality assessment
   - **Evidence**: Headline lengths follow a right-skewed distribution (not normal), indicating most headlines are concise with occasional long-form content

#### 2. **Publisher Analysis**:
   - Article counts per publisher (1,036 unique publishers)
   - Publisher domain extraction from email formats
   - **Concentration Metrics**: Gini coefficient calculation
   - **Power Law Distribution**: Publisher activity follows a power law (few publishers dominate)
   - **Evidence**: Top 10 publishers account for ~65% of articles, Gini coefficient > 0.7 indicates high concentration

#### 3. **Time Series Analysis**:
   - Daily publication volume trends over time
   - Hourly distribution (UTC) to identify peak publication times
   - Weekday patterns analysis
   - **Statistical Testing**: Chi-square test for weekday uniformity
   - **Evidence**: Significant weekday pattern (p < 0.05) - mid-week peak, low weekend activity

#### 4. **Topic Modeling (NLP)**:
   - Latent Dirichlet Allocation (LDA) with 6 topics
   - Keyword extraction and topic identification
   - **Evidence**: Six distinct thematic clusters identified:
     1. Analyst Ratings & Price Targets
     2. Earnings & Trading
     3. Corporate Announcements
     4. Market Movers
     5. ETF & Stock Movers
     6. Financial Metrics

#### 5. **Visualizations & Statistical Evidence**:
   - Distribution histograms with mean/median overlays
   - Q-Q plots for normality assessment
   - Time series plots showing publication trends
   - Weekday/hourly bar charts with statistical test results
   - Publisher concentration visualizations
   - All plots saved to `reports/figures/eda/`

**Output Files**: All statistical analyses, CSV data, and JSON metadata saved to `data/processed/eda/`

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

**Option 1: Using Jupyter Notebooks (Recommended)**
1. Start Jupyter Lab:
   ```bash
   jupyter lab
   ```
2. **For EDA Analysis**: Open `notebooks/01_eda_analysis.ipynb`
   - Run all cells to perform complete EDA with statistical analysis and visualizations
3. **For Technical Analysis**: Open `notebooks/02_quantitative_analysis.ipynb`
   - Run all cells to compute TA-Lib and PyNance indicators with visualizations
4. **For Sentiment Correlation**: Open `notebooks/03_sentiment_correlation.ipynb`
   - Run all cells to perform sentiment analysis and correlation with stock returns

**Option 2: Using Python Scripts**
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

3. **Run Sentiment Correlation Analysis**:
   ```bash
   jupyter lab notebooks/03_sentiment_correlation.ipynb
   ```
   Outputs: `data/processed/sentiment_correlation/*.csv` and correlation visualizations

## Output Files

### EDA Outputs (`data/processed/eda/`)
- `headline_length_stats.csv`: Descriptive statistics (mean, median, std, etc.)
- `statistical_analysis.json`: Distribution analysis (skewness, kurtosis, normality tests)
- `publisher_article_counts.csv`: Article counts by publisher
- `publisher_domain_counts.csv`: Article counts by domain
- `publisher_concentration_stats.json`: Gini coefficient and concentration metrics
- `daily_publication_counts.csv`: Daily time series
- `hourly_publication_counts.csv`: Hourly distribution (UTC)
- `weekday_publication_counts.csv`: Day-of-week distribution
- `time_series_statistics.json`: Chi-square test results and temporal patterns
- `topic_keywords.json`: LDA topic keywords

### EDA Visualizations (`reports/figures/eda/`)
- `headline_length_distributions.png`: Histograms and Q-Q plots for length distributions
- `publisher_analysis.png`: Top publishers bar chart and distribution plot
- `time_series_analysis.png`: Daily trends, weekday patterns, and hourly distributions

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

## Assignment Requirements Checklist

### Task 1: Git and GitHub ✅
- [x] GitHub repository created and properly structured
- [x] Branch `task-1` created for EDA work
- [x] Multiple commits with descriptive messages
- [x] CI/CD workflow configured (`.github/workflows/unittests.yml`)
- [x] Proper folder structure matching requirements

### Task 1: EDA & Statistical Analysis ✅
- [x] **Descriptive Statistics**: Headline length metrics (characters, words)
- [x] **Publisher Analysis**: Article counts per publisher, domain extraction
- [x] **Time Series Analysis**: Daily, hourly, weekday publication patterns
- [x] **Text Analysis/Topic Modeling**: LDA topic modeling with keyword extraction
- [x] **Statistical Distributions**: Normality tests, skewness, kurtosis analysis
- [x] **Statistical Plots**: Histograms, Q-Q plots, time series, distribution plots
- [x] **Evidence-Based Insights**: Chi-square tests, Gini coefficient, distribution analysis

### Task 2: Technical Analysis ✅
- [x] Branch `task-2` created
- [x] Pull Request merged to main branch
- [x] **TA-Lib Indicators**: SMA, EMA, RSI, MACD, ATR computed
- [x] **PyNance Metrics**: Growth, volatility, moving averages
- [x] **Data Visualization**: Multi-panel candlestick charts with indicators
- [x] Analysis completed for 6 stocks (AAPL, AMZN, GOOG, META, MSFT, NVDA)

### Task 3: Sentiment Analysis and Correlation ✅
- [x] Branch `task-3` created
- [x] **Sentiment Analysis**: TextBlob-based sentiment scoring on headlines
- [x] **Date Alignment**: Normalized timestamps between news and stock data
- [x] **Daily Returns**: Calculated percentage changes in stock prices
- [x] **Daily Aggregation**: Average sentiment scores when multiple articles per day
- [x] **Correlation Analysis**: Pearson correlation between sentiment and returns
- [x] **Visualizations**: Sentiment distribution, correlation charts, scatter plots, time series
- [x] **References**: Included TextBlob documentation and financial sentiment research papers

## Project Status
- [x] Data ingestion and preprocessing
- [x] EDA automation with statistical analysis (Task 1) ✅
- [x] Technical indicator pipeline (Task 2) ✅
- [x] Sentiment scoring integration (Task 3) ✅
- [ ] Predictive modeling (Future work)
- [ ] Final reporting (In progress)

## License

This project is part of the Nova Financial Insights challenge.

## Repository Link

**GitHub Repository**: https://github.com/meleseabrham/week1.git
