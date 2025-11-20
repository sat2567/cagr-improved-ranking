import streamlit as st
import pandas as pd
import numpy as np
import os
import warnings
import plotly.graph_objects as go
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

# ============================================================================
# 1. CONFIGURATION & CONSTANTS
# ============================================================================

st.set_page_config(
    page_title="Fund Analysis Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Default Parameters
DEFAULT_HOLDING = 126              # Approx 6 months holding (trading days)
DEFAULT_TOP_N = 5
RISK_FREE_RATE = 0.06              # 6% annual (Used for Alpha/Treynor)
EXECUTION_LAG = 1                  # Days between Analysis and Entry (T+1)
TRADING_DAYS_YEAR = 252            # Standardized trading days in a year

# Calculate Daily Risk Free Rate (Compounded)
DAILY_RISK_FREE_RATE = (1 + RISK_FREE_RATE) ** (1/TRADING_DAYS_YEAR) - 1

CATEGORY_MAP = {
    "Large Cap": "largecap",
    "Small Cap": "smallcap",
    "Mid Cap": "midcap",
    "Large & Mid Cap": "large_and_midcap",
    "Multi Cap": "multicap",
    "International": "international",
}

# ============================================================================
# 2. HELPER FUNCTIONS (MATH)
# ============================================================================

def calculate_sharpe_ratio(returns):
    """Returns Sharpe Ratio (Higher is Better)"""
    if len(returns) < 10 or returns.std() == 0: return np.nan
    excess_returns = returns - DAILY_RISK_FREE_RATE
    return (excess_returns.mean() / returns.std()) * np.sqrt(TRADING_DAYS_YEAR)

def calculate_sortino_ratio(returns):
    """Returns Sortino Ratio (Higher is Better)"""
    if len(returns) < 10: return np.nan
    excess_returns = returns - DAILY_RISK_FREE_RATE
    downside_returns = returns[returns < 0]
    if len(downside_returns) < 2 or downside_returns.std() == 0: return np.nan
    return (excess_returns.mean() / downside_returns.std()) * np.sqrt(TRADING_DAYS_YEAR)

def calculate_volatility(returns):
    """Returns Annualized Volatility (Lower is Better)"""
    if len(returns) < 10: return np.nan
    return returns.std() * np.sqrt(TRADING_DAYS_YEAR)

def calculate_information_ratio(fund_returns, bench_returns):
    """Returns Information Ratio (Higher is Better)"""
    common_idx = fund_returns.index.intersection(bench_returns.index)
    if len(common_idx) < 10: return np.nan
    
    f_ret = fund_returns.loc[common_idx]
    b_ret = bench_returns.loc[common_idx]
    
    active_return = f_ret - b_ret
    tracking_error = active_return.std() * np.sqrt(TRADING_DAYS_YEAR)
    
    if tracking_error == 0: return np.nan
    return (active_return.mean() * TRADING_DAYS_YEAR) / tracking_error

def calculate_flexible_momentum(series, w_3m, w_6m, w_12m, use_risk_adjust=False):
    """
    Calculates a composite momentum score based on user-defined weights.
    """
    if len(series) < 70: return np.nan 
    
    price_cur = series.iloc[-1]
    current_date = series.index[-1]
    
    def get_past_price(days_ago):
        target_date = current_date - pd.Timedelta(days=days_ago)
        sub_series = series[series.index <= target_date]
        if sub_series.empty: return np.nan
        return sub_series.iloc[-1]

    ret_3m, ret_6m, ret_12m = 0.0, 0.0, 0.0
    
    if w_3m > 0:
        p_3m = get_past_price(91)
        if pd.isna(p_3m) or p_3m == 0: return np.nan
        ret_3m = (price_cur / p_3m) - 1

    if w_6m > 0:
        p_6m = get_past_price(182)
        if pd.isna(p_6m) or p_6m == 0: return np.nan
        ret_6m = (price_cur / p_6m) - 1

    if w_12m > 0:
        p_12m = get_past_price(365)
        if pd.isna(p_12m) or p_12m == 0: return np.nan
        ret_12m = (price_cur / p_12m) - 1

    raw_score = (ret_3m * w_3m) + (ret_6m * w_6m) + (ret_12m * w_12m)
    
    if use_risk_adjust:
        date_1y_ago = current_date - pd.Timedelta(days=365)
        hist_vol_data = series[series.index >= date_1y_ago]
        if len(hist_vol_data) < 20: return np.nan
        
        vol = hist_vol_data.pct_change().dropna().std() * np.sqrt(TRADING_DAYS_YEAR)
        if vol == 0: return np.nan
        return raw_score / vol
        
    return raw_score

# --- NEW MATH FUNCTIONS ---

def calculate_max_drawdown(series):
    """Returns Max Drawdown (Negative value: -0.30 for 30% loss)"""
    if len(series) < 10: return np.nan
    peak = series.expanding(min_periods=1).max()
    drawdown = (series - peak) / peak
    return drawdown.min()

def calculate_cagr(series):
    """Returns CAGR from a NAV series"""
    if len(series) < 2: return np.nan
    start_val = series.iloc[0]
    end_val = series.iloc[-1]
    days = (series.index[-1] - series.index[0]).days
    if days <= 0 or start_val <= 0: return np.nan
    return (end_val / start_val)**(365.25 / days) - 1

def calculate_calmar_ratio(series):
    """Returns Calmar Ratio (Higher is Better)"""
    cagr = calculate_cagr(series)
    max_dd = calculate_max_drawdown(series)
    
    if np.isnan(cagr) or np.isnan(max_dd) or max_dd >= 0: return np.nan
    # MaxDD is negative, so we use abs()
    return cagr / abs(max_dd) 

def calculate_alpha_beta_treynor(returns, bench_returns):
    """Calculates Beta, Annualized Alpha, and Treynor Ratio"""
    common_idx = returns.index.intersection(bench_returns.index)
    if len(common_idx) < 50: return {'alpha': np.nan, 'beta': np.nan, 'treynor': np.nan}
    
    r_p = returns.loc[common_idx]
    r_m = bench_returns.loc[common_idx]
    
    # Excess Returns
    r_p_excess = r_p - DAILY_RISK_FREE_RATE
    r_m_excess = r_m - DAILY_RISK_FREE_RATE
    
    # 1. Beta
    cov_pm = r_p.cov(r_m)
    var_m = r_m.var()
    if var_m == 0: beta = np.nan
    else: beta = cov_pm / var_m
    
    # 2. Alpha (Annualized)
    # CAPM Expected Excess Return: Beta * (Market Excess Return)
    expected_excess_return = beta * r_m_excess.mean()
    alpha = (r_p_excess.mean() - expected_excess_return) * TRADING_DAYS_YEAR

    # 3. Treynor Ratio
    annual_return = r_p.mean() * TRADING_DAYS_YEAR
    if np.isnan(beta) or beta == 0: treynor = np.nan
    else: treynor = (annual_return - RISK_FREE_RATE) / beta
    
    return {'alpha': alpha, 'beta': beta, 'treynor': treynor}


# ============================================================================
# 3. DATA LOADING (Unchanged)
# ============================================================================

@st.cache_data
def load_fund_data(category_key: str):
    category_to_csv = {
        "largecap": "data/largecap_funds.csv",
        "smallcap": "data/smallcap_funds.csv",
        "midcap": "data/midcap_funds.csv",
        "large_and_midcap": "data/large_and_midcap_funds.csv",
        "multicap": "data/multicap_funds.csv",
        "international": "data/international_funds.csv",
    }
    path = category_to_csv.get(category_key)
    if not os.path.exists(path): return None, None
    
    try:
        df = pd.read_csv(path)
        df.columns = [c.lower().strip() for c in df.columns]
        if 'scheme_code' in df.columns: df['scheme_code'] = df['scheme_code'].astype(str)
        
        # Robust Date Parsing
        df['date'] = pd.to_datetime(df['date'], errors='coerce', infer_datetime_format=True)

        df['nav'] = pd.to_numeric(df['nav'], errors='coerce')
        df = df.dropna(subset=['date', 'nav']) 
        df = df.sort_values('date').drop_duplicates(subset=['scheme_code', 'date'], keep='last')
        scheme_map = df[['scheme_code', 'scheme_name']].drop_duplicates('scheme_code').set_index('scheme_code')['scheme_name'].to_dict()
        nav_wide = df.pivot(index='date', columns='scheme_code', values='nav')
        
        # Conservative fill (2 days max)
        nav_wide = nav_wide.ffill(limit=2) 
        
        return nav_wide, scheme_map
    except Exception as e:
        st.error(f"Error loading fund data: {e}"); return None, None

@st.cache_data
def load_nifty_data():
    # Assuming 'data/nifty100_funds.csv' is a placeholder path.
    # The actual file is `nifty100_funds.csv` in the current working directory.
    # Since this is a file provided by the user, we should assume the path is correct 
    # for the Streamlit cloud environment, or simply use the file name if it's in the root.
    # I'll keep the path as defined by the user's initial code, assuming their file structure.
    path = 'data/nifty100_funds.csv'
    if not os.path.exists(path): 
        # Fallback to current directory if 'data/' prefix is incorrect
        path = 'nifty100_funds.csv'
        if not os.path.exists(path): return None

    try:
        df = pd.read_csv(path)
        df.columns = [c.lower().strip() for c in df.columns]
        if 'close' in df.columns: df = df.rename(columns={'close': 'nav'})
        
        df['date'] = pd.to_datetime(df['date'], errors='coerce', infer_datetime_format=True)
        df['nav'] = pd.to_numeric(df['nav'], errors='coerce')
        df = df.dropna(subset=['date', 'nav'])
        return df.set_index('date')['nav'].sort_index()
    except Exception as e:
        st.error(f"Error loading benchmark data: {e}"); return None

# ============================================================================
# 4. BACKTESTING ENGINE (UPDATED FOR NEW METRICS)
# ============================================================================

def get_lookback_data(nav_wide, analysis_date, strategy_type):
    """Retrieves historical data based on strategy requirements."""
    max_days = 400 
    start_date = analysis_date - pd.Timedelta(days=max_days)
    
    hist_data = nav_wide[nav_wide.index >= start_date]
    hist_data = hist_data[hist_data.index < analysis_date]
    return hist_data

def run_backtest(nav_wide, strategy_type, top_n, holding_days, custom_weights=None, momentum_config=None, benchmark_series=None):
    
    start_date_required = nav_wide.index.min() + pd.Timedelta(days=370)
    try:
        start_idx = nav_wide.index.searchsorted(start_date_required)
    except:
        start_idx = 0
        
    end_idx = len(nav_wide) - holding_days - EXECUTION_LAG
    
    if start_idx >= end_idx: return pd.DataFrame(), pd.DataFrame()

    rebalance_indices = range(start_idx, end_idx, holding_days)
    
    history, equity_curve = [], [{'date': nav_wide.index[start_idx], 'value': 100.0}]
    current_capital = 100.0
    
    for i in rebalance_indices:
        analysis_date = nav_wide.index[i]
        hist_data = get_lookback_data(nav_wide, analysis_date, strategy_type)
        scores = {}
        
        # --- BENCHMARK RETURNS FOR RELATIVE METRICS ---
        bench_rets = None
        if benchmark_series is not None:
            bench_slice = get_lookback_data(benchmark_series.to_frame(), analysis_date, 'sharpe')
            bench_rets = bench_slice['nav'].pct_change().dropna()
        
        # --- A) STANDARD STRATEGIES ---
        if strategy_type != 'custom':
             for col in nav_wide.columns:
                series = hist_data[col].dropna()
                if len(series) < 126: continue
                
                val = np.nan
                if strategy_type == 'momentum':
                    w3 = momentum_config.get('w_3m', 0)
                    w6 = momentum_config.get('w_6m', 0)
                    w12 = momentum_config.get('w_12m', 0)
                    risk_adj = momentum_config.get('risk_adjust', False)
                    val = calculate_flexible_momentum(series, w3, w6, w12, risk_adj)
                else:
                    date_1y_ago = analysis_date - pd.Timedelta(days=365)
                    short_series = series[series.index >= date_1y_ago]
                    rets = short_series.pct_change().dropna()
                    
                    if strategy_type == 'sharpe': val = calculate_sharpe_ratio(rets)
                    elif strategy_type == 'sortino': val = calculate_sortino_ratio(rets)
                
                if not np.isnan(val): scores[col] = val

        # --- B) CUSTOM STRATEGY (UPDATED) ---
        else:
            temp_metrics = []
            
            for col in nav_wide.columns:
                series = hist_data[col].dropna()
                if len(series) < 126: continue
                
                row = {'id': col}
                
                # Returns (Used for Sharpe/Sortino/Info Ratio/Alpha/Treynor)
                date_1y_ago = analysis_date - pd.Timedelta(days=365)
                short_series = series[series.index >= date_1y_ago]
                rets = short_series.pct_change().dropna()
                
                # NAV series (Used for MaxDD/Calmar)
                nav_series_1y = series[series.index >= date_1y_ago]
                
                # 1. Standard Metrics
                if custom_weights.get('sharpe', 0) > 0: row['sharpe'] = calculate_sharpe_ratio(rets)
                if custom_weights.get('sortino', 0) > 0: row['sortino'] = calculate_sortino_ratio(rets)
                if custom_weights.get('volatility', 0) > 0: row['volatility'] = calculate_volatility(rets)
                
                # 2. Momentum
                if custom_weights.get('momentum', 0) > 0: 
                    w3 = momentum_config.get('w_3m', 0)
                    w6 = momentum_config.get('w_6m', 0)
                    w12 = momentum_config.get('w_12m', 0)
                    risk_adj = momentum_config.get('risk_adjust', False)
                    row['momentum'] = calculate_flexible_momentum(series, w3, w6, w12, risk_adj)

                # 3. Alpha/Beta/Treynor/Info (Requires Benchmark)
                required_bench = any(custom_weights.get(m, 0) > 0 for m in ['alpha', 'beta', 'treynor', 'info_ratio'])
                if required_bench and bench_rets is not None and not bench_rets.empty:
                    
                    # Alpha/Beta/Treynor calculation
                    if custom_weights.get('alpha', 0) > 0 or custom_weights.get('beta', 0) > 0 or custom_weights.get('treynor', 0) > 0:
                        capm_metrics = calculate_alpha_beta_treynor(rets, bench_rets)
                        if custom_weights.get('alpha', 0) > 0: row['alpha'] = capm_metrics['alpha']
                        if custom_weights.get('beta', 0) > 0: row['beta'] = capm_metrics['beta']
                        if custom_weights.get('treynor', 0) > 0: row['treynor'] = capm_metrics['treynor']
                        
                    # Information Ratio
                    if custom_weights.get('info_ratio', 0) > 0: 
                         row['info_ratio'] = calculate_information_ratio(rets, bench_rets)
                         
                # 4. Max Drawdown & Calmar (Requires NAV series)
                required_nav_metrics = any(custom_weights.get(m, 0) > 0 for m in ['max_dd', 'calmar'])
                if required_nav_metrics and not nav_series_1y.empty:
                    if custom_weights.get('max_dd', 0) > 0: row['max_dd'] = calculate_max_drawdown(nav_series_1y)
                    if custom_weights.get('calmar', 0) > 0: row['calmar'] = calculate_calmar_ratio(nav_series_1y)

                temp_metrics.append(row)
            
            if temp_metrics:
                metrics_df = pd.DataFrame(temp_metrics).set_index('id')
                final_score_col = pd.Series(0.0, index=metrics_df.index)
                
                # Metrics where HIGHER is BETTER
                for metric in ['sharpe', 'sortino', 'momentum', 'info_ratio', 'alpha', 'treynor', 'calmar']:
                    if metric in metrics_df.columns:
                        final_score_col = final_score_col.add(metrics_df[metric].rank(pct=True) * custom_weights[metric], fill_value=0)
                
                # Metrics where LOWER is BETTER 
                # (Volatility, Beta: Lower means less risk)
                for metric in ['volatility', 'beta']:
                    if metric in metrics_df.columns:
                        final_score_col = final_score_col.add(metrics_df[metric].rank(pct=True, ascending=False) * custom_weights[metric], fill_value=0)
                
                # Max Drawdown (it's negative, so ranking by value already means smallest loss is highest rank)
                if 'max_dd' in metrics_df.columns:
                    final_score_col = final_score_col.add(metrics_df['max_dd'].rank(pct=True) * custom_weights['max_dd'], fill_value=0)

                scores = final_score_col.to_dict()

        if not scores: continue
        selected = [k for k, v in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]]
        
        # --- EXECUTION --- (Remaining logic unchanged)
        entry_idx = i + EXECUTION_LAG
        exit_idx = entry_idx + holding_days
        if exit_idx >= len(nav_wide): break
        
        entry_date, exit_date = nav_wide.index[entry_idx], nav_wide.index[exit_idx]
        period_rets = (nav_wide.iloc[exit_idx] / nav_wide.iloc[entry_idx]) - 1
        
        valid_rets = period_rets[selected].dropna()
        avg_ret = valid_rets.mean() if not valid_rets.empty else 0.0
        
        current_capital *= (1 + avg_ret)
        
        history.append({
            'analysis_date': analysis_date,
            'entry_date': entry_date,
            'exit_date': exit_date,
            'selected_funds': ",".join(map(str, selected)),
            'period_return': avg_ret,
            'cum_value': current_capital
        })
        equity_curve.append({'date': exit_date, 'value': current_capital})
        
    return pd.DataFrame(history), pd.DataFrame(equity_curve)

def generate_snapshot_table(nav_wide, analysis_date, holding_days, strategy_type, names_map, custom_weights=None, momentum_config=None, benchmark_series=None):
    try: idx = nav_wide.index.get_loc(analysis_date)
    except KeyError: return pd.DataFrame()

    entry_idx = idx + EXECUTION_LAG
    exit_idx = entry_idx + holding_days
    has_future = False if exit_idx >= len(nav_wide) else True
    
    hist_data = get_lookback_data(nav_wide, analysis_date, strategy_type) 
    
    bench_rets = None
    if benchmark_series is not None:
        bench_slice = get_lookback_data(benchmark_series.to_frame(), analysis_date, 'sharpe')
        bench_rets = bench_slice['nav'].pct_change().dropna()

    temp_data = []

    for col in nav_wide.columns:
        series = hist_data[col].dropna()
        if len(series) < 126: continue
        
        row = {'id': col, 'name': names_map.get(col, col)}
        
        date_1y_ago = analysis_date - pd.Timedelta(days=365)
        short_series = series[series.index >= date_1y_ago]
        rets = short_series.pct_change().dropna()
        nav_series_1y = series[series.index >= date_1y_ago]

        # --- SCORES (ONLY CALCULATE IF WEIGHT > 0 OR IF STANDARD STRATEGY) ---
        if strategy_type == 'sharpe': row['Score'] = calculate_sharpe_ratio(rets)
        elif strategy_type == 'sortino': row['Score'] = calculate_sortino_ratio(rets)
        elif strategy_type == 'momentum':
            w3 = momentum_config.get('w_3m', 0)
            w6 = momentum_config.get('w_6m', 0)
            w12 = momentum_config.get('w_12m', 0)
            risk_adj = momentum_config.get('risk_adjust', False)
            row['Score'] = calculate_flexible_momentum(series, w3, w6, w12, risk_adj)
        elif strategy_type == 'custom':
            
            # Standard Metrics
            if custom_weights.get('sharpe',0)>0: row['sharpe'] = calculate_sharpe_ratio(rets)
            if custom_weights.get('sortino',0)>0: row['sortino'] = calculate_sortino_ratio(rets)
            if custom_weights.get('volatility',0)>0: row['volatility'] = calculate_volatility(rets)
            if custom_weights.get('momentum',0)>0: 
                 w3 = momentum_config.get('w_3m', 0)
                 w6 = momentum_config.get('w_6m', 0)
                 w12 = momentum_config.get('w_12m', 0)
                 risk_adj = momentum_config.get('risk_adjust', False)
                 row['momentum'] = calculate_flexible_momentum(series, w3, w6, w12, risk_adj)

            # Benchmark Dependent Metrics
            required_bench = any(custom_weights.get(m, 0) > 0 for m in ['alpha', 'beta', 'treynor', 'info_ratio'])
            if required_bench and bench_rets is not None and not bench_rets.empty:
                
                capm_metrics = calculate_alpha_beta_treynor(rets, bench_rets)
                if custom_weights.get('alpha', 0) > 0: row['alpha'] = capm_metrics['alpha']
                if custom_weights.get('beta', 0) > 0: row['beta'] = capm_metrics['beta']
                if custom_weights.get('treynor', 0) > 0: row['treynor'] = capm_metrics['treynor']
                if custom_weights.get('info_ratio',0)>0: row['info_ratio'] = calculate_information_ratio(rets, bench_rets)
            
            # Drawdown/Calmar Metrics
            required_nav_metrics = any(custom_weights.get(m, 0) > 0 for m in ['max_dd', 'calmar'])
            if required_nav_metrics and not nav_series_1y.empty:
                # Max Drawdown is typically shown as a positive percentage (10% loss) but calculated as negative (-0.10)
                if custom_weights.get('max_dd', 0) > 0: row['max_dd'] = abs(calculate_max_drawdown(nav_series_1y)) * 100 
                if custom_weights.get('calmar', 0) > 0: row['calmar'] = calculate_calmar_ratio(nav_series_1y)
        
        # --- FORWARD RETURN ---
        raw_ret = np.nan
        if has_future:
            p_entry = nav_wide[col].iloc[entry_idx]
            p_exit = nav_wide[col].iloc[exit_idx]
            if pd.notnull(p_entry) and pd.notnull(p_exit) and p_entry > 0:
                raw_ret = (p_exit / p_entry) - 1

        row['Forward Return %'] = raw_ret * 100 if not np.isnan(raw_ret) else np.nan
        
        temp_data.append(row)

    df = pd.DataFrame(temp_data)
    if df.empty: return df

    # --- RANKING ---
    if strategy_type == 'custom':
        df['Score'] = 0.0
        
        # List of metrics and their ranking direction (True=Ascending/Higher is Better)
        ranking_directions = {
            'sharpe': True, 'sortino': True, 'momentum': True, 'info_ratio': True, 
            'alpha': True, 'treynor': True, 'calmar': True, 
            'volatility': False, 'beta': False, 'max_dd': False 
        }
        
        for metric, ascending in ranking_directions.items():
            if metric in df.columns and custom_weights.get(metric, 0) > 0:
                # Note on Max Drawdown: It's stored as an absolute positive value here (e.g., 30.0 for 30% loss)
                # so LOWER is BETTER (ascending=False).
                df['Score'] = df['Score'].add(
                    df[metric].rank(pct=True, ascending=ascending) * custom_weights[metric], 
                    fill_value=0
                )
                
    df['Strategy Rank'] = df['Score'].rank(ascending=False, method='min')
    
    if has_future:
        df['Actual Rank'] = df['Forward Return %'].rank(ascending=False, method='min')
    
    # Select columns for final display
    cols_to_display = ['name', 'Strategy Rank', 'Forward Return %']
    
    # Add calculated metrics to display if they were used (have a weight > 0)
    for metric in ranking_directions.keys():
        if metric in df.columns:
            cols_to_display.append(metric)

    return df[cols_to_display].sort_values('Strategy Rank')


# ============================================================================
# 5. DASHBOARD UI (UPDATED WITH NEW SLIDERS)
# ============================================================================

def main():
    st.title("📊 Fund Analysis: Custom & Standard Strategies")
    
    # --- Sidebar ---
    st.sidebar.header("⚙️ General Settings")
    cat_key = CATEGORY_MAP[st.sidebar.selectbox("Category", list(CATEGORY_MAP.keys()))]
    
    col_s1, col_s2 = st.sidebar.columns(2)
    top_n = col_s1.number_input("Top N Funds", 1, 20, DEFAULT_TOP_N)
    holding_days = col_s2.number_input("Rebalance (Days)", 20, TRADING_DAYS_YEAR, DEFAULT_HOLDING)
    
    st.sidebar.divider()
    st.sidebar.header("🚀 Momentum Configuration")
    st.sidebar.caption("Weights for Trend Calculation:")
    
    mom_c1, mom_c2, mom_c3 = st.sidebar.columns(3)
    w_3m = mom_c1.number_input("3M Weight", 0.0, 10.0, 1.0, 0.1)
    w_6m = mom_c2.number_input("6M Weight", 0.0, 10.0, 1.0, 0.1)
    w_12m = mom_c3.number_input("1Y Weight", 0.0, 10.0, 1.0, 0.1)
    
    risk_adjust_mom = st.sidebar.checkbox("Risk Adjust Momentum?", value=True, help="Divide score by Volatility")
    
    total_w = w_3m + w_6m + w_12m
    momentum_config = {
        'w_3m': w_3m/total_w if total_w > 0 else 0, 
        'w_6m': w_6m/total_w if total_w > 0 else 0, 
        'w_12m': w_12m/total_w if total_w > 0 else 0,
        'risk_adjust': risk_adjust_mom
    }
    
    # --- Load Data ---
    with st.spinner("Loading Data..."):
        nav_data, names_map = load_fund_data(cat_key)
        nifty_data = load_nifty_data()
        
    if nav_data is None:
        st.error("❌ Fund data not found or failed to load."); return
    
    # --- Display Function (Fixed to avoid NameError) ---
    def display_strategy_results(strat_type, c_weights, nav_data, names_map, top_n, holding_days, momentum_config, nifty_data):
        
        hist_df, eq_curve = run_backtest(nav_data, strat_type, top_n, holding_days, c_weights, momentum_config, nifty_data)
        
        if hist_df is None or hist_df.empty:
            st.warning("Insufficient data for backtest."); return

        # Summary
        start_date = eq_curve.iloc[0]['date']
        end_date = eq_curve.iloc[-1]['date']
        final_val = eq_curve.iloc[-1]['value']
        time_period_years = (end_date-start_date).days/365.25
        strat_cagr = (final_val/100)**(1/time_period_years) - 1 if time_period_years > 0 else 0.0
        
        bench_curve = None
        bench_cagr = 0.0
        if nifty_data is not None:
            sub_nifty = nifty_data[(nifty_data.index >= start_date) & (nifty_data.index <= end_date)]
            if not sub_nifty.empty and sub_nifty.iloc[0] > 0:
                bench_cagr = (sub_nifty.iloc[-1]/sub_nifty.iloc[0])**(1/((sub_nifty.index[-1]-sub_nifty.index[0]).days/365.25)) - 1
                bench_curve = (sub_nifty / sub_nifty.iloc[0]) * 100

        c1, c2, c3 = st.columns(3)
        c1.metric("Strategy CAGR", f"{strat_cagr:.2%}")
        c2.metric("Benchmark CAGR", f"{bench_cagr:.2%}")
        c3.metric("Total Return", f"{final_val - 100:.1f}%")

        # Chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=eq_curve['date'], y=eq_curve['value'], name='Strategy', line=dict(color='#00CC96')))
        if bench_curve is not None:
            fig.add_trace(go.Scatter(x=bench_curve.index, y=bench_curve.values, name='Benchmark', line=dict(color='#EF553B', dash='dot')))
        fig.update_layout(height=400, title="Equity Curve", hovermode="x unified", xaxis=dict(rangeslider=dict(visible=True)))
        st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.subheader("🔍 Deep Dive Snapshot")
        date_options = hist_df['analysis_date'].dt.strftime('%Y-%m-%d').tolist()
        sel_date_str = st.selectbox("Select Rebalance Date", date_options, key=f"sd_{strat_type}")
        
        if sel_date_str:
            sel_date = pd.to_datetime(sel_date_str)
            # Pass all context variables explicitly
            df = generate_snapshot_table(nav_data, sel_date, holding_days, strat_type, names_map, c_weights, momentum_config, nifty_data)
            
            if not df.empty:
                def highlight_top_n(row):
                    if row['Strategy Rank'] <= top_n:
                        return ['background-color: #e6fffa'] * len(row)
                    return [''] * len(row)
                
                # Format map for cleaner display
                format_map = {
                    col: "%.2f%%" for col in ['Forward Return %', 'volatility', 'alpha', 'max_dd'] if col in df.columns
                }
                format_map.update({
                    col: "%.2f" for col in ['sharpe', 'sortino', 'calmar', 'momentum', 'info_ratio', 'treynor', 'beta'] if col in df.columns
                })
                
                # Rename columns for display
                column_renames = {
                    'volatility': 'Volatility %',
                    'alpha': 'Alpha %',
                    'max_dd': 'Max DD %',
                    'info_ratio': 'Info Ratio',
                    'treynor': 'Treynor Ratio',
                    'calmar': 'Calmar Ratio'
                }
                
                # Apply renames and sort to display columns
                df_display = df.rename(columns=column_renames)
                cols_to_display = ['name', 'Strategy Rank', 'Forward Return %'] + [column_renames.get(c, c) for c in df.columns if c not in ['id', 'Score', 'name', 'Strategy Rank', 'Forward Return %', 'Actual Rank']]
                
                st.dataframe(
                    df_display[cols_to_display].style.format(format_map).apply(highlight_top_n, axis=1),
                    use_container_width=True,
                    column_config={
                        "Forward Return %": st.column_config.NumberColumn("Forward Return", format="%.2f%%"),
                        "Max DD %": st.column_config.NumberColumn("Max DD", format="%.2f%%", help="Maximum historical loss over the lookback period."),
                        "Alpha %": st.column_config.NumberColumn("Alpha", format="%.2f%%", help="Annualized measure of manager skill relative to the Nifty 100."),
                    },
                    hide_index=True
                )

    # --- Tabs ---
    tab_mom, tab_sharpe, tab_custom = st.tabs(["🚀 Momentum Strategy", "⚖️ Sharpe Ratio", "🛠️ Custom Strategy"])

    # Explicitly pass all required context variables to avoid NameError
    context_args = (nav_data, names_map, top_n, holding_days, momentum_config, nifty_data)

    with tab_mom:
        st.info(f"Momentum Weights: 3M={momentum_config['w_3m']:.2f}, 6M={momentum_config['w_6m']:.2f}, 1Y={momentum_config['w_12m']:.2f}. Risk Adjust: {momentum_config['risk_adjust']}")
        display_strategy_results('momentum', None, *context_args)

    with tab_sharpe:
        display_strategy_results('sharpe', None, *context_args)

    with tab_custom:
        with st.form("custom_strat_form"):
            st.markdown("##### Risk/Return Metrics (Higher is generally better)")
            col_c1, col_c2, col_c3 = st.columns(3)
            w_sharpe = col_c1.slider("Sharpe Weight", 0, 100, 20, 10, key="s_sharpe")
            w_sortino = col_c2.slider("Sortino Weight", 0, 100, 0, 10, key="s_sortino")
            w_mom = col_c3.slider("Momentum Weight", 0, 100, 20, 10, key="s_mom")

            col_c4, col_c5 = st.columns(2)
            w_ir = col_c4.slider("Info Ratio Weight", 0, 100, 0, 10, key="s_ir")
            w_calmar = col_c5.slider("Calmar Weight", 0, 100, 20, 10, key="s_calmar")

            st.markdown("##### Manager Skill / Systematic Risk Metrics (Requires Benchmark)")
            col_c6, col_c7, col_c8 = st.columns(3)
            w_alpha = col_c6.slider("Alpha Weight", 0, 100, 20, 10, key="s_alpha")
            w_treynor = col_c7.slider("Treynor Weight", 0, 100, 0, 10, key="s_treynor")
            w_beta = col_c8.slider("Beta Weight (LOWER is BETTER)", 0, 100, 10, 10, key="s_beta") # Lower Beta is often preferred

            st.markdown("##### Pure Risk Metrics (LOWER is BETTER)")
            col_c9, col_c10 = st.columns(2)
            w_vol = col_c9.slider("Low Volatility Weight", 0, 100, 10, 10, key="s_vol")
            w_max_dd = col_c10.slider("Max Drawdown Weight", 0, 100, 0, 10, key="s_maxdd")
            
            st.info("The weights are used to score the funds. Metrics marked (LOWER is BETTER) are inversely ranked.")
            
            submit_btn = st.form_submit_button("🚀 Run Custom Strategy", key="submit_custom")

        if submit_btn:
            weights = {
                'sharpe': w_sharpe/100, 'sortino': w_sortino/100, 'momentum': w_mom/100,
                'volatility': w_vol/100, 'info_ratio': w_ir/100, 'calmar': w_calmar/100,
                'alpha': w_alpha/100, 'beta': w_beta/100, 'treynor': w_treynor/100,
                'max_dd': w_max_dd/100
            }
            if sum(weights.values()) == 0: st.error("Select at least one weight > 0")
            else:
                st.session_state['custom_run'] = True
                st.session_state['custom_weights'] = weights
        
        if st.session_state.get('custom_run'):
            # Pass custom weights and all context arguments
            display_strategy_results('custom', st.session_state['custom_weights'], *context_args)

if __name__ == "__main__":
    if 'custom_run' not in st.session_state:
        st.session_state['custom_run'] = False
        st.session_state['custom_weights'] = {}
    main()
