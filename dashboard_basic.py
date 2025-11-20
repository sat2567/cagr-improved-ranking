"""
Advanced Dashboard - Detailed Fund Selection with Custom Strategies
Adds Max Drawdown, Calmar Ratio, Beta, and Alpha to the metric set.
Includes a Custom Strategy Builder for weighted ranking.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# CONSTANTS
# ============================================================================

LOOKBACK = 252          # 1 year in trading days
HOLDING_PERIOD = 126    # 6 months in trading days
RISK_FREE_RATE = 0.05   # 5% annual
DEFAULT_TOP_N = 5
TRADING_DAYS_YEAR = 252

CATEGORY_MAP = {
    "Large Cap": "largecap",
    "Small Cap": "smallcap",
    "Mid Cap": "midcap",
    "Large & Mid Cap": "large_and_midcap",
    "Multi Cap": "multicap",
    "International": "international",
}

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="Advanced Fund Selection Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# NEW & EXISTING METRIC CALCULATION FUNCTIONS
# ============================================================================

def calculate_sharpe_ratio(returns, risk_free_rate=RISK_FREE_RATE):
    """Calculate annualized Sharpe ratio."""
    if len(returns) == 0 or returns.std() == 0:
        return np.nan
    excess_returns = returns - risk_free_rate / TRADING_DAYS_YEAR
    sharpe = (excess_returns.mean() / returns.std()) * np.sqrt(TRADING_DAYS_YEAR)
    return sharpe

def calculate_sortino_ratio(returns, risk_free_rate=RISK_FREE_RATE):
    """Calculate annualized Sortino ratio."""
    if len(returns) == 0:
        return np.nan
    excess_returns = returns - risk_free_rate / TRADING_DAYS_YEAR
    downside_returns = returns[returns < 0]
    if len(downside_returns) == 0 or downside_returns.std() == 0:
        return np.nan
    sortino = (excess_returns.mean() / downside_returns.std()) * np.sqrt(TRADING_DAYS_YEAR)
    return sortino

def calculate_max_drawdown(nav_series):
    """Calculate Max Drawdown over the period."""
    if len(nav_series) < 2: return np.nan
    nav_series = nav_series.astype(float)
    peak = nav_series.expanding(min_periods=1).max()
    drawdown = (nav_series - peak) / peak
    return drawdown.min()

def calculate_cagr(nav_series):
    """Calculate simple CAGR over the period."""
    if len(nav_series) < 2: return np.nan
    nav_series = nav_series.astype(float)
    days = (nav_series.index[-1] - nav_series.index[0]).days
    years = days / 365.25
    if years <= 0: return np.nan
    return (nav_series.iloc[-1] / nav_series.iloc[0]) ** (1 / years) - 1

def calculate_calmar_ratio(nav_series):
    """Calculate Calmar ratio (CAGR / |Max Drawdown|)."""
    cagr = calculate_cagr(nav_series)
    mdd = calculate_max_drawdown(nav_series)
    if np.isnan(cagr) or np.isnan(mdd) or mdd == 0: return np.nan
    return cagr / abs(mdd)

def calculate_beta(returns, benchmark_returns):
    """Calculate Beta against the benchmark."""
    if len(returns) < 2 or len(benchmark_returns) < 2: return np.nan
    
    # Align indices and drop NaNs
    common_index = returns.index.intersection(benchmark_returns.index)
    returns = returns.loc[common_index]
    benchmark_returns = benchmark_returns.loc[common_index]
    
    if len(returns) < 2: return np.nan
    
    # Covariance of fund and benchmark / Variance of benchmark
    fund_cov = returns.cov(benchmark_returns)
    bench_var = benchmark_returns.var()
    
    if bench_var == 0: return np.nan
    return fund_cov / bench_var

def calculate_alpha(returns, benchmark_returns, risk_free_rate=RISK_FREE_RATE):
    """Calculate annualized Jensen's Alpha."""
    beta = calculate_beta(returns, benchmark_returns)
    if np.isnan(beta): return np.nan
    
    # Calculate average daily excess returns
    fund_excess_mean = (returns - risk_free_rate / TRADING_DAYS_YEAR).mean()
    bench_excess_mean = (benchmark_returns - risk_free_rate / TRADING_DAYS_YEAR).mean()
    
    # Calculate daily Alpha
    daily_alpha = fund_excess_mean - beta * bench_excess_mean
    
    # Annualize Alpha
    return daily_alpha * TRADING_DAYS_YEAR

# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

@st.cache_data
def load_fund_data(category_key: str):
    """Load and prepare fund NAV data."""
    # ... (No change to load_fund_data)
    category_to_csv = {
        "largecap": "data/largecap_funds.csv",
        "smallcap": "data/smallcap_funds.csv",
        "midcap": "data/midcap_funds.csv",
        "large_and_midcap": "data/large_and_midcap_funds.csv",
        "multicap": "data/multicap_funds.csv",
        "international": "data/international_funds.csv",
    }
    
    funds_csv = category_to_csv.get(category_key, "data/largecap_funds.csv")
    
    if not os.path.exists(funds_csv):
        return None, None
    
    df = pd.read_csv(funds_csv)
    # Assuming standard fund data format
    df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y', errors='coerce')
    df['nav'] = pd.to_numeric(df['nav'], errors='coerce')
    df = df.drop_duplicates(subset=['scheme_code', 'date'], keep='last')
    
    nav_wide = df.pivot(index='date', columns='scheme_code', values='nav')
    nav_wide = nav_wide.sort_index().ffill(limit=5)
    
    scheme_names = df[['scheme_code', 'scheme_name']].drop_duplicates().set_index('scheme_code')['scheme_name'].to_dict()
    
    return nav_wide, scheme_names

@st.cache_data
def load_nifty_data():
    """Load Nifty 100 benchmark data (corrected for file structure)."""
    try:
        # Using the file previously inspected: nifty100_funds.csv
        nifty_df = pd.read_csv('nifty100_funds.csv')
        nifty_df.columns = [c.lower().strip() for c in nifty_df.columns]
        nifty_df['date'] = pd.to_datetime(nifty_df['date'], format='%d-%m-%Y', errors='coerce')
        nifty_df['nav'] = pd.to_numeric(nifty_df['nav'], errors='coerce')
        return nifty_df.dropna(subset=['date', 'nav']).sort_values('date').set_index('date')['nav']
    except Exception:
        return None

def calculate_simple_nifty_cagr(nifty_data, start_date, end_date):
    """Calculate simple CAGR from first to last Nifty price."""
    # ... (Existing function, no change)
    if nifty_data is None or len(nifty_data) == 0: return np.nan, None, None
    try:
        start_idx = nifty_data.index.get_indexer([start_date], method='nearest')[0]
        end_idx = nifty_data.index.get_indexer([end_date], method='nearest')[0]
        
        if start_idx < 0 or end_idx < 0 or start_idx >= len(nifty_data) or end_idx >= len(nifty_data):
            return np.nan, None, None
        
        initial_nifty = float(nifty_data.iloc[start_idx])
        final_nifty = float(nifty_data.iloc[end_idx])
        
        if initial_nifty <= 0 or final_nifty <= 0: return np.nan, None, None
        
        actual_start = nifty_data.index[start_idx]
        actual_end = nifty_data.index[end_idx]
        
        total_days = (actual_end - actual_start).days
        total_years = total_days / 365.25
        
        if total_years <= 0: return np.nan, actual_start, actual_end
        
        cagr = ((final_nifty / initial_nifty) ** (1 / total_years)) - 1
        return cagr, actual_start, actual_end
    except Exception:
        return np.nan, None, None

# ============================================================================
# SELECTION AND METRIC CALCULATION
# ============================================================================

def calculate_metrics_for_date(nav_wide, nifty_data, date_idx, lookback=LOOKBACK):
    """Calculate all metrics for all funds at a given date."""
    if date_idx < lookback:
        return None
    
    # 1. Get fund lookback data
    lookback_data = nav_wide.iloc[date_idx - lookback:date_idx]
    
    # 2. Get benchmark lookback data
    # Ensure benchmark data covers the same lookback period
    start_date = lookback_data.index[0]
    end_date = lookback_data.index[-1]
    nifty_slice = nifty_data[(nifty_data.index >= start_date) & (nifty_data.index <= end_date)].ffill().pct_change().dropna()
    
    metrics = {}
    
    for fund in nav_wide.columns:
        fund_prices = lookback_data[fund].astype(float).dropna()
        
        if len(fund_prices) < 2:
            metrics[fund] = {m: np.nan for m in ['sharpe', 'sortino', 'max_drawdown', 'calmar', 'beta', 'alpha']}
            continue
        
        fund_returns = fund_prices.pct_change().dropna()
        
        # Align fund returns and benchmark returns for relative metrics
        common_index = fund_returns.index.intersection(nifty_slice.index)
        fund_returns_aligned = fund_returns.loc[common_index]
        nifty_returns_aligned = nifty_slice.loc[common_index]
        
        # Calculate Absolute Metrics
        sharpe = calculate_sharpe_ratio(fund_returns)
        sortino = calculate_sortino_ratio(fund_returns)
        mdd = calculate_max_drawdown(fund_prices)
        calmar = calculate_calmar_ratio(fund_prices)
        
        # Calculate Relative Metrics
        beta = calculate_beta(fund_returns_aligned, nifty_returns_aligned)
        alpha = calculate_alpha(fund_returns_aligned, nifty_returns_aligned, RISK_FREE_RATE)
        
        metrics[fund] = {
            'sharpe': sharpe,
            'sortino': sortino,
            'max_drawdown': mdd,
            'calmar': calmar,
            'beta': beta,
            'alpha': alpha
        }
    
    return metrics

def calculate_custom_score(metrics_df, weights):
    """Calculates a composite score based on user-defined weights."""
    df = metrics_df.copy()
    
    # Normalize metrics using percentile rank (0 to 1)
    
    # High-is-better metrics (Sharpe, Sortino, Calmar, Alpha)
    for col in ['sharpe', 'sortino', 'calmar', 'alpha']:
        if weights.get(col, 0) > 0:
            df[f'{col}_rank'] = df[col].rank(pct=True, ascending=True)
    
    # Low-is-better metrics (Max Drawdown, Beta)
    # Max Drawdown is a negative number, so rank by ascending (less negative is better)
    # Beta ranking depends on the strategy (default: lower is safer, but here we rank for low volatility as a general filter)
    if weights.get('max_drawdown', 0) > 0:
        df['max_drawdown_rank'] = df['max_drawdown'].rank(pct=True, ascending=True) # less negative is better
    if weights.get('beta', 0) > 0:
        df['beta_rank'] = df['beta'].rank(pct=True, ascending=False) # Lower Beta is better/safer
        
    df['custom_score'] = 0.0
    
    for metric, weight in weights.items():
        if weight > 0:
            # Check if the required rank column exists
            rank_col = f'{metric}_rank'
            if rank_col in df.columns:
                df['custom_score'] += df[rank_col] * (weight / 100.0) # Sum weighted ranks
            
    # Normalize the final score by the sum of weights (if any weights are > 0)
    total_weight = sum(weights.values()) / 100.0
    if total_weight > 0:
        df['custom_score'] = df['custom_score'] / total_weight

    return df['custom_score']

def calculate_forward_returns(nav_wide, date_idx, holding_period=HOLDING_PERIOD):
    """Calculate forward returns and ranks for all funds."""
    # ... (Existing function, no change)
    if date_idx + holding_period >= len(nav_wide):
        return None, None
    
    current_prices = nav_wide.iloc[date_idx]
    future_prices = nav_wide.iloc[date_idx + holding_period]
    
    forward_returns = (future_prices / current_prices - 1).dropna()
    forward_ranks = forward_returns.rank(ascending=False, method='first')
    
    return forward_returns, forward_ranks

def process_all_selections(nav_wide, nifty_data, top_n=DEFAULT_TOP_N):
    """Process all selection dates and calculate metrics."""
    # This function is now updated to calculate ALL metrics
    start_idx = LOOKBACK
    end_idx = len(nav_wide) - HOLDING_PERIOD
    
    if start_idx >= end_idx:
        return []
    
    selection_indices = range(start_idx, end_idx, HOLDING_PERIOD)
    results = []
    
    for date_idx in selection_indices:
        selection_date = nav_wide.index[date_idx]
        
        # Calculate metrics (now including new metrics)
        metrics = calculate_metrics_for_date(nav_wide, nifty_data, date_idx, LOOKBACK)
        if metrics is None: continue
        
        # Calculate forward returns
        forward_returns, forward_ranks = calculate_forward_returns(nav_wide, date_idx, HOLDING_PERIOD)
        if forward_returns is None: continue
        
        # Create DataFrame with all metrics
        metrics_df = pd.DataFrame(metrics).T
        metrics_df['forward_return'] = metrics_df.index.map(forward_returns.to_dict())
        metrics_df['forward_rank'] = metrics_df.index.map(forward_ranks.to_dict())
        
        # Select top N by Sharpe
        sharpe_selected = metrics_df.nlargest(top_n, 'sharpe').index.tolist()
        
        # Select top N by Sortino
        sortino_selected = metrics_df.nlargest(top_n, 'sortino').index.tolist()
        
        results.append({
            'selection_date': selection_date,
            'date_idx': date_idx,
            'sharpe_selected': sharpe_selected,
            'sortino_selected': sortino_selected,
            'metrics_df': metrics_df,
            'forward_returns': forward_returns,
            'forward_ranks': forward_ranks
        })
    
    return results

def create_six_month_windows(start_date, end_date):
    """Create non-overlapping 6-month windows for selection and evaluation."""
    # ... (Existing function, no change)
    windows = []
    current = start_date
    window_id = 1
    # Use a simpler date offset for demonstration
    while current < end_date:
        selection_start = current
        evaluation_end = selection_start + pd.DateOffset(months=6) + pd.DateOffset(months=6) # 1 year window
        
        if evaluation_end > end_date:
            evaluation_end = end_date
            
        selection_end = current + pd.DateOffset(months=6)
        evaluation_start = selection_end
        
        windows.append({
            'Window ID': window_id,
            'Selection Start': selection_start.strftime('%Y-%m-%d'),
            'Selection End': selection_end.strftime('%Y-%m-%d'),
            'Evaluation Start': evaluation_start.strftime('%Y-%m-%d'),
            'Evaluation End': evaluation_end.strftime('%Y-%m-%d')
        })
        
        current = evaluation_end # Move to the end of the evaluation period
        window_id += 1
    
    return pd.DataFrame(windows)

def simulate_portfolio(selection_results, strategy='sharpe', custom_weights=None):
    """Simulate equally weighted portfolio, now supporting custom score."""
    portfolio_value = 1.0
    start_date = None
    end_date = None
    
    for result in selection_results:
        metrics_df = result['metrics_df']
        forward_returns = result['forward_returns']
        
        if strategy == 'custom' and custom_weights is not None:
            # Calculate custom score and select funds
            custom_scores = calculate_custom_score(metrics_df, custom_weights)
            # Merge score back into the dataframe for selection
            metrics_df['custom_score'] = custom_scores
            selected_funds = metrics_df.nlargest(DEFAULT_TOP_N, 'custom_score').index.tolist()
        else:
            selected_funds = result[f'{strategy}_selected']
            
        if forward_returns is None or len(selected_funds) == 0:
            continue
        
        # Get returns for selected funds
        selected_returns = [forward_returns.get(fund, np.nan) for fund in selected_funds]
        selected_returns = [r for r in selected_returns if not np.isnan(r)]
        
        if len(selected_returns) == 0:
            continue
        
        # Average return
        avg_return = np.mean(selected_returns)
        
        # Apply to portfolio
        portfolio_value *= (1 + avg_return)
        
        if start_date is None:
            start_date = result['selection_date']
        end_date = result['selection_date'] + pd.DateOffset(months=6)
        
    return portfolio_value, start_date, end_date

def calculate_portfolio_cagr(portfolio_value, start_date, end_date):
    """Calculate CAGR for portfolio."""
    # ... (Existing function, no change)
    if portfolio_value <= 0 or start_date is None or end_date is None: return np.nan
    
    total_days = (end_date - start_date).days
    total_years = total_days / 365.25
    
    if total_years <= 0: return np.nan
    
    cagr = (portfolio_value ** (1 / total_years)) - 1
    return cagr

# ============================================================================
# DASHBOARD UI
# ============================================================================

def main():
    st.title("📊 Advanced Fund Selection Dashboard")
    st.markdown("Detailed Fund Selection and **Custom Strategy Builder**")
    
    # --- Sidebar ---
    st.sidebar.header("⚙️ Configuration")
    cat_name = st.sidebar.selectbox("Category", list(CATEGORY_MAP.keys()), index=0)
    cat_key = CATEGORY_MAP[cat_name]
    top_n = st.sidebar.number_input("Top N Funds", min_value=1, max_value=20, value=DEFAULT_TOP_N, step=1)
    
    # --- Load Data ---
    with st.spinner("Loading and processing data..."):
        nav_wide, scheme_names = load_fund_data(cat_key)
        nifty_data = load_nifty_data()
        
    if nav_wide is None or nifty_data is None:
        st.error("❌ Data files not found or failed to load. Please check the data directory.")
        return
    
    if len(nav_wide) < LOOKBACK + HOLDING_PERIOD:
        st.error("❌ Insufficient fund data. Need at least 1.5 years of data.")
        return
    
    # Calculate and display Nifty 100 CAGR
    fund_start_date = nav_wide.index[0]
    fund_end_date = nav_wide.index[-1]
    nifty_cagr, nifty_start, nifty_end = calculate_simple_nifty_cagr(nifty_data, fund_start_date, fund_end_date)
    
    if not np.isnan(nifty_cagr):
        col1, col2, col3 = st.columns(3)
        with col1: st.metric("Nifty 100 CAGR", f"{nifty_cagr*100:.2f}%")
        with col2: st.caption(f"From: {nifty_start.strftime('%Y-%m-%d')}")
        with col3: st.caption(f"To: {nifty_end.strftime('%Y-%m-%d')}")

    st.markdown("---")
    
    # --- Process Selections ---
    with st.spinner("Calculating all strategy metrics and selection windows..."):
        # Process selections (now calculating ALL metrics)
        selection_results = process_all_selections(nav_wide, nifty_data, top_n)
    
    if not selection_results:
        st.error("❌ No selection results generated. Check data range and lookback period.")
        return
    
    # --- Strategy Performance Tabs ---
    st.markdown("### Strategy Performance Comparison (Top 5 Funds, 6-Month Rebalance)")
    
    # Calculate portfolio CAGRs for base strategies
    sharpe_final_value, _, _ = simulate_portfolio(selection_results, 'sharpe')
    sharpe_cagr = calculate_portfolio_cagr(sharpe_final_value, selection_results[0]['selection_date'], selection_results[-1]['selection_date'] + pd.DateOffset(months=6))
    
    sortino_final_value, _, _ = simulate_portfolio(selection_results, 'sortino')
    sortino_cagr = calculate_portfolio_cagr(sortino_final_value, selection_results[0]['selection_date'], selection_results[-1]['selection_date'] + pd.DateOffset(months=6))
    
    col1, col2, col3 = st.columns(3)
    with col1: st.metric("Sharpe Strategy CAGR", f"{sharpe_cagr*100:.2f}%" if not np.isnan(sharpe_cagr) else "N/A")
    with col2: st.metric("Sortino Strategy CAGR", f"{sortino_cagr*100:.2f}%" if not np.isnan(sortino_cagr) else "N/A")
    with col3: st.metric("Nifty 100 CAGR", f"{nifty_cagr*100:.2f}%" if not np.isnan(nifty_cagr) else "N/A")
    
    st.markdown("---")
    
    # --- Custom Strategy Builder & Results ---
    
    st.subheader("🛠️ Custom Strategy Builder")
    st.markdown("Assign weights (0-100%) to rank funds using a composite score.")

    # Custom Strategy Weight Sliders
    col_w1, col_w2, col_w3, col_w4, col_w5, col_w6 = st.columns(6)
    w_sharpe = col_w1.slider("Sharpe", 0, 100, 20)
    w_sortino = col_w2.slider("Sortino", 0, 100, 20)
    w_alpha = col_w3.slider("Alpha", 0, 100, 20)
    w_calmar = col_w4.slider("Calmar", 0, 100, 20)
    w_mdd = col_w5.slider("Low MaxDD", 0, 100, 10)
    w_beta = col_w6.slider("Low Beta", 0, 100, 10)

    custom_weights = {
        'sharpe': w_sharpe, 'sortino': w_sortino, 'alpha': w_alpha, 
        'calmar': w_calmar, 'max_drawdown': w_mdd, 'beta': w_beta
    }
    
    # Custom Strategy CAGR
    custom_final_value, _, _ = simulate_portfolio(selection_results, 'custom', custom_weights)
    custom_cagr = calculate_portfolio_cagr(custom_final_value, selection_results[0]['selection_date'], selection_results[-1]['selection_date'] + pd.DateOffset(months=6))
    st.metric("Custom Strategy CAGR", f"{custom_cagr*100:.2f}%" if not np.isnan(custom_cagr) else "N/A")
    st.markdown("---")
    
    # --- Detailed View ---
    st.subheader("🔍 Detailed Fund Selection (Point-in-Time)")
    
    dates = [r['selection_date'] for r in selection_results]
    selected_date_idx = st.selectbox("Select Rebalance Date", range(len(dates)), format_func=lambda x: dates[x].strftime('%Y-%m-%d'))
    
    if selected_date_idx < len(selection_results):
        result = selection_results[selected_date_idx]
        metrics_df = result['metrics_df']
        
        # Calculate custom score for this specific date's display
        metrics_df['Custom Score'] = calculate_custom_score(metrics_df, custom_weights)
        
        # Generate the three result tables side-by-side
        tab_sharpe, tab_sortino, tab_custom = st.tabs(["Sharpe Selection", "Sortino Selection", "Custom Selection"])

        def display_selection_table(strategy_key, tab, score_col=None):
            with tab:
                st.markdown(f"#### Top {top_n} Selected Funds")
                
                if score_col:
                    funds = metrics_df.nlargest(top_n, score_col).index.tolist()
                else:
                    funds = result[f'{strategy_key}_selected']

                display_df = []
                for fund in funds:
                    if fund in metrics_df.index:
                        row = metrics_df.loc[fund]
                        display_df.append({
                            'Fund Name': scheme_names.get(fund, fund),
                            'Sharpe Ratio': f"{row['sharpe']:.3f}" if not np.isnan(row['sharpe']) else "N/A",
                            'Sortino Ratio': f"{row['sortino']:.3f}" if not np.isnan(row['sortino']) else "N/A",
                            'Alpha (Annual%)': f"{row['alpha']*100:.2f}" if not np.isnan(row['alpha']) else "N/A",
                            'Calmar Ratio': f"{row['calmar']:.2f}" if not np.isnan(row['calmar']) else "N/A",
                            'Max Drawdown (%)': f"{row['max_drawdown']*100:.2f}" if not np.isnan(row['max_drawdown']) else "N/A",
                            'Beta': f"{row['beta']:.2f}" if not np.isnan(row['beta']) else "N/A",
                            'Forward Return (%)': f"{row['forward_return']*100:.2f}" if not np.isnan(row['forward_return']) else "N/A",
                            'Forward Rank': f"{row['forward_rank']:.0f}" if not np.isnan(row['forward_rank']) else "N/A"
                        })
                
                if display_df:
                    st.dataframe(pd.DataFrame(display_df), use_container_width=True)

        display_selection_table('sharpe', tab_sharpe)
        display_selection_table('sortino', tab_sortino)
        display_selection_table('custom', tab_custom, 'Custom Score')

if __name__ == "__main__":
    main()
