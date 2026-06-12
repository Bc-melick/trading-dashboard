"""
generate_dashboard.py
=====================
Runs the full trading strategy, fetches news & macro data, and writes
a self-contained index.html that GitHub Pages will serve as your dashboard.
 
Dependencies (all free, no credit card):
    pip install pandas numpy yfinance plotly requests fredapi
 
Free API keys needed:
    - NewsAPI  : https://newsapi.org/register   (free, no credit card)
    - FRED     : https://fred.stlouisfed.org/docs/api/api_key.html (free, no credit card)
 
Set these as GitHub Actions secrets named:
    NEWS_API_KEY
    FRED_API_KEY
"""
 
import os
import json
import time
import requests
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.offline import plot
import yfinance as yf
from datetime import datetime, timedelta
 
# =============================================================================
# CONFIGURATION
# =============================================================================
 
NEWS_API_KEY  = os.environ.get('NEWS_API_KEY', '')
FRED_API_KEY  = os.environ.get('FRED_API_KEY', '')
 
weights           = {'SPY': 0.5, 'QQQ': 0.5}
start_date        = datetime(2015, 1, 1)
end_date          = datetime.now()
BACKTEST_START    = datetime(2015, 2, 1)
STARTING_CAPITAL  = 1_000_000.0
 
# Sector ETFs for top-3 sector performance (1 month)
SECTOR_ETFS = {
    'Technology':       'XLK',
    'Healthcare':       'XLV',
    'Financials':       'XLF',
    'Energy':           'XLE',
    'Consumer Discr.':  'XLY',
    'Consumer Staples': 'XLP',
    'Industrials':      'XLI',
    'Materials':        'XLB',
    'Real Estate':      'XLRE',
    'Utilities':        'XLU',
    'Communication':    'XLC',
}
 
# =============================================================================
# HELPERS
# =============================================================================
 
def fetch_closes(tickers, start, end, max_retries=5, delay=2):
    for attempt in range(max_retries):
        try:
            raw = yf.download(tickers, start=start, end=end,
                              group_by='ticker', auto_adjust=True,
                              progress=False, threads=False)
            if raw.empty:
                raise ValueError('No data returned.')
 
            # Single ticker — raw has flat columns (Open, High, Low, Close…)
            if len(tickers) == 1:
                close = raw['Close'].squeeze()
                return pd.DataFrame({tickers[0]: close}).ffill().dropna()
 
            # Multiple tickers — yfinance returns a MultiIndex (field, ticker)
            # Handle both MultiIndex and cases where yfinance returns flat columns
            if isinstance(raw.columns, pd.MultiIndex):
                # Standard multi-ticker MultiIndex: ('Close', 'AAPL') etc.
                if 'Close' in raw.columns.get_level_values(0):
                    close_df = raw['Close']
                else:
                    # Some yfinance versions use ('AAPL', 'Close') ordering
                    close_df = raw.xs('Close', axis=1, level=1)
                # Only keep tickers we asked for that are present
                cols = [t for t in tickers if t in close_df.columns]
                return close_df[cols].ffill().dropna(how='all')
            else:
                # Flat columns — single ticker returned despite list input
                close = raw['Close'].squeeze()
                return pd.DataFrame({tickers[0]: close}).ffill().dropna()
 
        except Exception as e:
            print(f'Attempt {attempt+1} failed: {e}')
            time.sleep(delay)
    # All retries failed — return empty DataFrame so downstream code
    # can handle it gracefully rather than crashing the whole script
    print(f'WARNING: fetch_closes failed for {tickers[:5]}… returning empty')
    return pd.DataFrame()
 
 
def calculate_rsi(series, window=14):
    delta = series.diff()
    gain  = delta.where(delta > 0, 0.0).rolling(window=window).mean()
    loss  = (-delta.where(delta < 0, 0.0)).rolling(window=window).mean()
    rs    = gain / loss
    return 100 - (100 / (1 + rs))
 
 
def calculate_macd(series, short=12, long=26, signal=9):
    ema_s = series.ewm(span=short,   adjust=False).mean()
    ema_l = series.ewm(span=long,    adjust=False).mean()
    macd  = ema_s - ema_l
    sig   = macd.ewm(span=signal,    adjust=False).mean()
    return macd, sig
 
# =============================================================================
# STRATEGY — signal generation (mirrors trading_strategy.py exactly)
# =============================================================================
 
all_tickers = list(weights.keys()) + ['QQQ']
price_data  = fetch_closes(all_tickers, start_date, end_date)
 
blended_price = sum(price_data[t] * w for t, w in weights.items())
blended_price.name = 'Blended_Price'
qqq_price = price_data['QQQ']
spy_price  = price_data['SPY']
 
# Fetch S&P 500 index (^GSPC) for signal banner prices
# Use yf.download directly with a try/except since ^GSPC can behave
# differently from equity tickers in the multi-ticker fetch helper.
try:
    _spx_raw = yf.download('^GSPC', start=start_date, end=end_date,
                           auto_adjust=True, progress=False)
    if not _spx_raw.empty and 'Close' in _spx_raw.columns:
        spx_price = _spx_raw['Close'].squeeze().ffill().dropna()
    else:
        raise ValueError('Empty SPX data')
except Exception as _e:
    print(f'SPX fetch failed ({_e}), falling back to SPY proxy')
    spx_price = spy_price  # fallback — values will be SPY not SPX
 
ema_20  = blended_price.ewm(span=20,  adjust=False).mean()
ema_50  = blended_price.ewm(span=50,  adjust=False).mean()
ema_100 = blended_price.ewm(span=100, adjust=False).mean()
ema_200 = blended_price.ewm(span=200, adjust=False).mean()
 
rsi               = calculate_rsi(blended_price).fillna(0)
macd, signal_line = calculate_macd(blended_price)
macd              = macd.fillna(0)
signal_line       = signal_line.fillna(0)
 
signals_df = pd.DataFrame({
    'Blended_Price':   blended_price.values,
    'EMA_20':          ema_20.values,
    'EMA_50':          ema_50.values,
    'EMA_100':         ema_100.values,
    'EMA_200':         ema_200.values,
    'RSI':             rsi.values,
    'MACD':            macd.values,
    'Signal':          None,
    'Condition':       None,
    'EMA20_Below_EMA200': (ema_20 < ema_200).values,
}, index=blended_price.index)
 
buy_signals       = []
reduction_signals = []
 
consecutive_above_50              = 0
consecutive_below_20              = 0
consecutive_below_50              = 0
consecutive_above_20              = 0
consecutive_above_200             = 0
consecutive_above_50_after_reduce = 0
consecutive_above_20_after_reduce = 0
last_signal           = None
last_buy_price        = None
previous_high         = None
macd_cross_day        = None
macd_cross_rsi        = None
pending_cross_100     = False
pending_cross_200     = False
pending_macd_cross    = False
consecutive_100_rising = 0
 
 
def reset_all_counters():
    return 0, 0, 0, 0, 0, 0, 0
 
 
def trading_days_since_last_buy(sdf, current_idx):
    buy_rows = sdf[sdf['Signal'] == 'Buy']
    if buy_rows.empty:
        return float('inf')
    last_pos    = blended_price.index.get_loc(buy_rows.index[-1])
    current_pos = blended_price.index.get_loc(blended_price.index[current_idx])
    return current_pos - last_pos
 
 
for i in range(1, len(blended_price)):
    price      = blended_price.iloc[i]
    prev_price = blended_price.iloc[i - 1]
 
    current_rsi = rsi.iloc[i] if not pd.isna(rsi.iloc[i]) else None
 
    above_50 = price > ema_50.iloc[i]
    below_20 = price < ema_20.iloc[i]
    below_50 = price < ema_50.iloc[i]
    above_20 = price > ema_20.iloc[i]
 
    crossed_above_100 = (prev_price <= ema_100.iloc[i-1]) and (price > ema_100.iloc[i])
    crossed_above_200 = (prev_price <= ema_200.iloc[i-1]) and (price > ema_200.iloc[i])
 
    macd_crossed_today = (macd.iloc[i-1] <= signal_line.iloc[i-1]) and (macd.iloc[i] > signal_line.iloc[i])
    if macd_crossed_today:
        macd_cross_day = i
        macd_cross_rsi = rsi.iloc[i]
 
    macd_above_signal = macd.iloc[i] > signal_line.iloc[i]
 
    consecutive_above_50 = consecutive_above_50 + 1 if above_50 else 0
    consecutive_below_20 = consecutive_below_20 + 1 if below_20 else 0
    consecutive_below_50 = consecutive_below_50 + 1 if below_50 else 0
    consecutive_above_20 = consecutive_above_20 + 1 if above_20 else 0
 
    if price > ema_200.iloc[i]:
        consecutive_above_200 += 1
    else:
        consecutive_above_200 = 0
 
    if last_signal == 'reduce' and above_50:
        consecutive_above_50_after_reduce += 1
    else:
        consecutive_above_50_after_reduce = 0
 
    if last_signal == 'reduce' and above_20:
        consecutive_above_20_after_reduce += 1
    else:
        consecutive_above_20_after_reduce = 0
 
    if previous_high is None or price > previous_high:
        previous_high = price
 
    ema_100_rising   = ema_100.iloc[i] > ema_100.iloc[i - 1]
    consecutive_100_rising = consecutive_100_rising + 1 if ema_100_rising else 0
    ema_100_rising_3d = consecutive_100_rising >= 3
 
    if crossed_above_100:
        pending_cross_100 = False if ema_100_rising_3d else True
    if crossed_above_200:
        pending_cross_200 = False if ema_100_rising_3d else True
    if macd_crossed_today:
        pending_macd_cross = False if ema_100_rising_3d else True
 
    if pending_cross_100 and price <= ema_100.iloc[i]:
        pending_cross_100 = False
    if pending_cross_200 and price <= ema_200.iloc[i]:
        pending_cross_200 = False
    if pending_macd_cross and macd.iloc[i] <= signal_line.iloc[i]:
        pending_macd_cross = False
 
    fire_cross_100 = (
        (crossed_above_100 and ema_100_rising_3d)
        or (pending_cross_100 and ema_100_rising_3d and price > ema_100.iloc[i])
    )
    fire_cross_200 = (
        (crossed_above_200 and ema_100_rising_3d)
        or (pending_cross_200 and ema_100_rising_3d and price > ema_200.iloc[i])
    )
    fire_macd = (
        ema_100_rising_3d
        and macd.iloc[i] > signal_line.iloc[i]
        and (
            (macd_cross_day is not None and 0 <= i - macd_cross_day <= 2)
            or pending_macd_cross
        )
    )
 
    if fire_cross_100: pending_cross_100 = False
    if fire_cross_200: pending_cross_200 = False
    if fire_macd:      pending_macd_cross = False
 
    stayed_above_100_2d = fire_cross_100
    stayed_above_200_2d = fire_cross_200
    crossed_macd_signal = fire_macd
 
    buy_cond_1 = (stayed_above_100_2d or stayed_above_200_2d or crossed_macd_signal) \
                 and last_signal != 'buy'
 
    if buy_cond_1:
        if current_rsi is not None and 20 <= current_rsi <= 70 and price > prev_price:
            label = 'cross_200' if stayed_above_200_2d else ('cross_100' if stayed_above_100_2d else 'macd')
            buy_signals.append((blended_price.index[i], price))
            signals_df.loc[blended_price.index[i], 'Signal']    = 'Buy'
            signals_df.loc[blended_price.index[i], 'Condition'] = label
            last_signal = 'buy'; last_buy_price = price; previous_high = price
            pending_cross_100 = False; pending_cross_200 = False
            pending_macd_cross = False; consecutive_100_rising = 0
            (consecutive_above_50, consecutive_below_20, consecutive_below_50,
             consecutive_above_20, consecutive_above_200,
             consecutive_above_50_after_reduce,
             consecutive_above_20_after_reduce) = reset_all_counters()
            continue
 
    if (consecutive_above_50_after_reduce >= 4 and last_signal != 'buy'
            and crossed_macd_signal and price > prev_price):
        buy_signals.append((blended_price.index[i], price))
        signals_df.loc[blended_price.index[i], 'Signal']    = 'Buy'
        signals_df.loc[blended_price.index[i], 'Condition'] = 'cross_50'
        last_signal = 'buy'; last_buy_price = price; previous_high = price
        pending_cross_100 = False; pending_cross_200 = False
        pending_macd_cross = False; consecutive_100_rising = 0
        (consecutive_above_50, consecutive_below_20, consecutive_below_50,
         consecutive_above_20, consecutive_above_200,
         consecutive_above_50_after_reduce,
         consecutive_above_20_after_reduce) = reset_all_counters()
        continue
 
    if (ema_20.iloc[i] < ema_100.iloc[i] and consecutive_above_20 >= 2
            and last_signal != 'buy' and price > prev_price):
        buy_signals.append((blended_price.index[i], price))
        signals_df.loc[blended_price.index[i], 'Signal']    = 'Buy'
        signals_df.loc[blended_price.index[i], 'Condition'] = 'cross_20_inverse'
        last_signal = 'buy'; last_buy_price = price; previous_high = price
        pending_cross_100 = False; pending_cross_200 = False
        pending_macd_cross = False; consecutive_100_rising = 0
        (consecutive_above_50, consecutive_below_20, consecutive_below_50,
         consecutive_above_20, consecutive_above_200,
         consecutive_above_50_after_reduce,
         consecutive_above_20_after_reduce) = reset_all_counters()
        continue
 
    days_since_buy = trading_days_since_last_buy(signals_df, i)
    enough_days    = days_since_buy >= 5
 
    if (consecutive_below_50 >= 2 and not macd_above_signal
            and last_signal != 'reduce' and enough_days
            and ema_100.iloc[i] > ema_50.iloc[i]):
        reduction_signals.append((blended_price.index[i], price))
        signals_df.loc[blended_price.index[i], 'Signal']    = 'Reduce'
        signals_df.loc[blended_price.index[i], 'Condition'] = 'below_50_ema_2d_100>50'
        last_signal = 'reduce'
        (consecutive_above_50, consecutive_below_20, consecutive_below_50,
         consecutive_above_20, consecutive_above_200,
         consecutive_above_50_after_reduce,
         consecutive_above_20_after_reduce) = reset_all_counters()
        continue
 
    if (consecutive_below_50 >= 2 and last_signal != 'reduce' and enough_days
            and previous_high is not None and price < previous_high * 0.975
            and current_rsi is not None and current_rsi > 20):
        reduction_signals.append((blended_price.index[i], price))
        signals_df.loc[blended_price.index[i], 'Signal']    = 'Reduce'
        signals_df.loc[blended_price.index[i], 'Condition'] = 'below_50_ema_2d_price_drop_2.5pct'
        last_signal = 'reduce'
        (consecutive_above_50, consecutive_below_20, consecutive_below_50,
         consecutive_above_20, consecutive_above_200,
         consecutive_above_50_after_reduce,
         consecutive_above_20_after_reduce) = reset_all_counters()
        continue
 
    if (ema_200.iloc[i] > ema_50.iloc[i] and consecutive_below_20 >= 2
            and not macd_above_signal and last_signal != 'reduce' and enough_days):
        reduction_signals.append((blended_price.index[i], price))
        signals_df.loc[blended_price.index[i], 'Signal']    = 'Reduce'
        signals_df.loc[blended_price.index[i], 'Condition'] = 'below_20_ema_2d_200>50'
        last_signal = 'reduce'
        (consecutive_above_50, consecutive_below_20, consecutive_below_50,
         consecutive_above_20, consecutive_above_200,
         consecutive_above_50_after_reduce,
         consecutive_above_20_after_reduce) = reset_all_counters()
        continue
 
    if (consecutive_below_20 >= 3 and last_signal != 'reduce' and enough_days
            and not macd_above_signal
            and previous_high is not None and price < previous_high * 0.975):
        reduction_signals.append((blended_price.index[i], price))
        signals_df.loc[blended_price.index[i], 'Signal']    = 'Reduce'
        signals_df.loc[blended_price.index[i], 'Condition'] = 'below_20_ema_3d_price_drop_2.5pct'
        last_signal = 'reduce'
        (consecutive_above_50, consecutive_below_20, consecutive_below_50,
         consecutive_above_20, consecutive_above_200,
         consecutive_above_50_after_reduce,
         consecutive_above_20_after_reduce) = reset_all_counters()
        continue
 
signals_df['Signal'] = signals_df['Signal'].fillna('None')
 
# =============================================================================
# BACKTEST ENGINE
# =============================================================================
 
bt_mask  = signals_df.index >= pd.Timestamp(BACKTEST_START)
bt_df    = signals_df[bt_mask].copy()
bt_qqq   = qqq_price[bt_mask].copy()
bt_spy   = spy_price[bt_mask].copy()
bt_blend = blended_price[bt_mask].copy()
 
portfolio_value       = STARTING_CAPITAL
exposure              = 1.0
qqq_shares            = (portfolio_value * exposure) / bt_qqq.iloc[0]
cash                  = 0.0
incrementing_active   = False
last_blend_ref        = None
bt_records            = []
 
for date, row in bt_df.iterrows():
    qqq_px   = bt_qqq.loc[date]
    blend_px = bt_blend.loc[date]
    signal   = row['Signal']
    buy_cond = row['Condition']
 
    portfolio_value = qqq_shares * qqq_px + cash
 
    if signal == 'Buy':
        if buy_cond in ('cross_200', 'cross_100', 'cross_20_inverse', 'macd'):
            qqq_shares = portfolio_value / qqq_px
            cash = 0.0; exposure = 1.0
            incrementing_active = False; last_blend_ref = None
        elif buy_cond == 'cross_50':
            target = min(exposure + 0.10, 1.0)
            qqq_shares = portfolio_value * target / qqq_px
            cash = portfolio_value * (1 - target)
            exposure = target
            incrementing_active = True; last_blend_ref = blend_px
 
    elif signal == 'Reduce':
        ema20_val  = bt_df.loc[date, 'EMA_20']
        ema100_val = bt_df.loc[date, 'EMA_100']
        ema200_val = bt_df.loc[date, 'EMA_200']
        full_exit  = (ema20_val < ema100_val) or (ema20_val < ema200_val)
        target     = 0.0 if full_exit else 0.50
        qqq_shares = portfolio_value * target / qqq_px
        cash       = portfolio_value * (1 - target)
        exposure   = target
        incrementing_active = False; last_blend_ref = None
 
    else:
        if incrementing_active and exposure < 1.0 and last_blend_ref is not None:
            if (blend_px - last_blend_ref) / last_blend_ref >= 0.01:
                target = min(exposure + 0.10, 1.0)
                qqq_shares = portfolio_value * target / qqq_px
                cash = portfolio_value * (1 - target)
                exposure = target; last_blend_ref = blend_px
                if exposure >= 1.0: incrementing_active = False
 
    portfolio_value = qqq_shares * qqq_px + cash
    bt_records.append({
        'Date':            date,
        'Portfolio_Value': round(portfolio_value, 2),
        'Exposure_Pct':    round(exposure * 100, 2),
        'Signal':          signal,
    })
 
bt_results = pd.DataFrame(bt_records).set_index('Date')
spy_shares = STARTING_CAPITAL / bt_spy.iloc[0]
bt_results['Benchmark_Value']         = (spy_shares * bt_spy).values
bt_results['Strategy_Cumulative_Pct'] = ((bt_results['Portfolio_Value']  / STARTING_CAPITAL - 1) * 100).round(2)
bt_results['SPY_Cumulative_Pct']      = ((bt_results['Benchmark_Value']  / STARTING_CAPITAL - 1) * 100).round(2)
 
# =============================================================================
# PERFORMANCE METRICS
# =============================================================================
 
def period_return(series, days=None):
    """Return % gain over last N calendar days, or full period if days=None."""
    s = series.iloc[-days:] if days else series
    return round((s.iloc[-1] / s.iloc[0] - 1) * 100, 2)
 
def ytd_return(series):
    """Return % gain from the first trading day of the current year to today.
    Uses date-based slicing so it matches the annual performance table exactly."""
    ytd_start = pd.Timestamp(datetime(datetime.now().year, 1, 1))
    # Normalise index timezone so comparison always works
    idx = series.index
    if idx.tz is not None:
        idx = idx.tz_localize(None)
        s = series.copy()
        s.index = idx
    else:
        s = series
    s_ytd = s[s.index >= ytd_start]
    if len(s_ytd) < 2:
        return round((s.iloc[-1] / s.iloc[0] - 1) * 100, 2)
    return round((s_ytd.iloc[-1] / s_ytd.iloc[0] - 1) * 100, 2)
 
def ann_return(series):
    dr = series.pct_change().dropna()
    return round(((1 + dr.mean()) ** 252 - 1) * 100, 2)
 
def ann_period_return(series, trading_days):
    """Annualize a return calculated over a slice of N trading days.
    Formula: (end/start)^(252/n) - 1"""
    s = series.iloc[-trading_days:]
    if len(s) < 2:
        return 0.0
    total = s.iloc[-1] / s.iloc[0]
    years = len(s) / 252
    return round((total ** (1 / years) - 1) * 100, 2)
 
def max_drawdown(series):
    roll_max = series.cummax()
    return round(((series - roll_max) / roll_max).min() * 100, 2)
 
strat_v = bt_results['Portfolio_Value']
bench_v = bt_results['Benchmark_Value']
 
# Trading days approximations
metrics = {
    'today_signal': signals_df['Signal'].iloc[-1],
    'today_condition': signals_df['Condition'].iloc[-1] or '',
    'today_exposure': bt_results['Exposure_Pct'].iloc[-1],
    'strat': {
        'ytd':    ytd_return(strat_v),
        '1yr':    period_return(strat_v, 252),
        '3yr':    period_return(strat_v, 756),
        '5yr':    period_return(strat_v, 1260),
        'all':    period_return(strat_v),
        'ann':    ann_return(strat_v),
        'mdd':    max_drawdown(strat_v),
        'end_val': round(strat_v.iloc[-1], 2),
    },
    'bench': {
        'ytd':    ytd_return(bench_v),
        '1yr':    period_return(bench_v, 252),
        '3yr':    period_return(bench_v, 756),
        '5yr':    period_return(bench_v, 1260),
        'all':    period_return(bench_v),
        'ann':    ann_return(bench_v),
        'mdd':    max_drawdown(bench_v),
        'end_val': round(bench_v.iloc[-1], 2),
    }
}
 
# Annual performance table
# For the current calendar year, use Jan 1 as the start so it matches
# the YTD figure shown in the trailing returns and performance cards.
annual_rows = []
current_year = datetime.now().year
bt_results['Year'] = bt_results.index.year
for year, grp in bt_results.groupby('Year'):
    if year == current_year:
        # YTD: anchor to Jan 1 so it matches the YTD cards and trailing table
        ytd_start_ts = pd.Timestamp(datetime(current_year, 1, 1)).tz_localize(None)
        idx_tz_naive = bt_results.index.tz_localize(None) if bt_results.index.tz is not None \
                       else bt_results.index
        grp_ytd = bt_results[idx_tz_naive >= ytd_start_ts]
        if len(grp_ytd) < 2:
            grp_ytd = grp
        sr = round((grp_ytd['Portfolio_Value'].iloc[-1] / grp_ytd['Portfolio_Value'].iloc[0] - 1) * 100, 2)
        br = round((grp_ytd['Benchmark_Value'].iloc[-1]  / grp_ytd['Benchmark_Value'].iloc[0]  - 1) * 100, 2)
    else:
        sr = round((grp['Portfolio_Value'].iloc[-1] / grp['Portfolio_Value'].iloc[0] - 1) * 100, 2)
        br = round((grp['Benchmark_Value'].iloc[-1]  / grp['Benchmark_Value'].iloc[0]  - 1) * 100, 2)
    annual_rows.append({'Year': year, 'Strategy': sr, 'SPY': br, 'Alpha': round(sr - br, 2)})
annual_df = pd.DataFrame(annual_rows)
 
# =============================================================================
# MARKET DATA  — sectors & top movers (1M, 6M, 1Y)
# =============================================================================
 
# ~150 liquid tickers: S&P 100 core + key high-momentum names across all sectors
# Sized to fetch reliably in a single yfinance call within GitHub Actions limits
LARGE_CAPS = [
    # Mega cap tech & software
    'AAPL','MSFT','NVDA','AMZN','GOOGL','META','TSLA','AVGO','ORCL','ADBE',
    'CRM','NOW','INTU','CSCO','IBM','QCOM','TXN','AMD','INTC','ACN',
    # Semis & hardware (key momentum names)
    'MU','AMAT','LRCX','KLAC','MRVL','SMCI','PLTR','ARM','DELL','HPQ',
    # Cybersecurity & cloud
    'CRWD','PANW','FTNT','NET','ZS','DDOG','SNOW','COIN',
    # Internet & media
    'NFLX','UBER','ABNB','BKNG','TTD','ROKU','SPOT','RBLX',
    # Payments & fintech
    'V','MA','PYPL','SQ','AXP','COF','HOOD',
    # Financials
    'JPM','BAC','WFC','GS','MS','BLK','SCHW','C',
    'SPGI','MCO','ICE','CME','PGR','CB','MMC',
    # Healthcare & biotech
    'LLY','UNH','JNJ','ABBV','MRK','PFE','TMO','ABT','DHR','SYK',
    'AMGN','GILD','REGN','VRTX','MRNA','ISRG','BSX','ELV','CVS','CI',
    # Consumer discretionary
    'AMZN','WMT','COST','HD','MCD','SBUX','CMG','NKE','LULU','BKNG',
    'TGT','LOW','TJX','ROST','DKNG','RCL','MAR','HLT','LVS','WYNN',
    # Consumer staples
    'PG','KO','PEP','PM','MO','MDLZ','CL','GIS',
    # Industrials
    'HON','RTX','LMT','BA','CAT','DE','GE','ETN','UPS','FDX',
    'NOC','GD','MMM','EMR','ITW','CSX','UNP','DAL','UAL',
    # Energy
    'XOM','CVX','COP','OXY','SLB','MPC','PSX','VLO','EOG','DVN',
    # Communication
    'DIS','CMCSA','T','VZ','TMUS','CHTR','PARA',
    # Real estate
    'AMT','PLD','EQIX','CCI','PSA','DLR','O','SPG','VICI',
    # Utilities
    'NEE','DUK','SO','D','AEP','SRE','EXC',
    # Materials
    'LIN','APD','SHW','ECL','NEM','FCX','ALB',
    # High-momentum & growth
    'SHOP','MELI','NU','KKR','APO','ARES','F','GM',
]
 
# Fetch enough history to cover 1 year for all timeframes in one call
market_start   = datetime.now() - timedelta(days=370)
sector_tickers = list(SECTOR_ETFS.values())
sector_data    = fetch_closes(sector_tickers, market_start, datetime.now())
stock_data     = fetch_closes(LARGE_CAPS,     market_start, datetime.now())
 
TIMEFRAME_DAYS = {'1 Month': 31, '6 Months': 182, '1 Year': 365}
 
def compute_returns(data, columns, days):
    """Return dict of {name: pct_return} for the given lookback period."""
    if data is None or data.empty:
        return {}
    cutoff = datetime.now() - timedelta(days=days)
    cutoff_ts = pd.Timestamp(cutoff).tz_localize(None)
    results = {}
    for col in columns:
        if col not in data.columns:
            continue
        try:
            idx = data.index.tz_localize(None) if data.index.tz is not None else data.index
            sub = data[col][idx >= cutoff_ts].dropna()
            if len(sub) >= 2 and sub.iloc[0] != 0:
                results[col] = round((sub.iloc[-1] / sub.iloc[0] - 1) * 100, 2)
        except Exception:
            continue
    return results
 
# Pre-compute all timeframes
sector_returns_all = {}
stock_returns_all  = {}
ticker_to_name     = {v: k for k, v in SECTOR_ETFS.items()}
 
for label, days in TIMEFRAME_DAYS.items():
    raw_sec  = compute_returns(sector_data, sector_tickers, days)
    # Map ticker -> sector name and get top 3
    named    = {ticker_to_name.get(t, t): v for t, v in raw_sec.items()}
    sector_returns_all[label] = sorted(named.items(), key=lambda x: x[1], reverse=True)[:5]
 
    raw_stk  = compute_returns(stock_data, LARGE_CAPS, days)
    stock_returns_all[label]  = sorted(raw_stk.items(), key=lambda x: x[1], reverse=True)[:20]
 
# =============================================================================
# MACRO DATA via FRED
# =============================================================================
 
def fred_series(series_id, api_key, limit=8):
    """Fetch the latest N observations from FRED API (desc order = newest first)."""
    if not api_key:
        return None
    url = (f'https://api.stlouisfed.org/fred/series/observations'
           f'?series_id={series_id}&api_key={api_key}&file_type=json'
           f'&sort_order=desc&limit={limit}')
    try:
        r = requests.get(url, timeout=15)
        obs = r.json().get('observations', [])
        # Return list of (date_str, float_value) tuples, newest first
        result = []
        for o in obs:
            if o['value'] not in ('.', '', None):
                try:
                    result.append((o['date'], float(o['value'])))
                except:
                    pass
        return result if result else None
    except:
        return None
 
def macro_row(label, obs, suffix='%', note=''):
    """
    Build a macro table row dict from a list of (date, value) tuples.
    obs[0] = latest, obs[3] = ~3 months ago, obs[6] = ~6 months ago
    (FRED monthly series: each obs is one month apart)
    Returns dict with keys: label, latest, val_3m, val_6m,
                            trend_3m, trend_6m, note
    """
    def fmt(v): return f'{v:.2f}{suffix}' if v is not None else 'N/A'
    def arrow(cur, old):
        if cur is None or old is None: return ''
        diff = cur - old
        if   diff >  0.01: return f'▲ +{diff:.2f}{suffix}'
        elif diff < -0.01: return f'▼ {diff:.2f}{suffix}'
        else:              return f'→ {diff:+.2f}{suffix}'
    latest_val = obs[0][1]  if obs and len(obs) > 0 else None
    val_3m     = obs[3][1]  if obs and len(obs) > 3 else None
    val_6m     = obs[6][1]  if obs and len(obs) > 6 else None
    date_3m    = obs[3][0]  if obs and len(obs) > 3 else ''
    date_6m    = obs[6][0]  if obs and len(obs) > 6 else ''
    return {
        'label':    label,
        'latest':   fmt(latest_val),
        'val_3m':   fmt(val_3m),
        'val_6m':   fmt(val_6m),
        'date_3m':  date_3m[:7] if date_3m else '',   # YYYY-MM
        'date_6m':  date_6m[:7] if date_6m else '',
        'trend_3m': arrow(latest_val, val_3m),
        'trend_6m': arrow(latest_val, val_6m),
        'note':     note,
    }
 
def yf_macro_fallback():
    """Fallback when no FRED key — pull rate proxies from yfinance."""
    proxies = {'^TNX': '10-Yr Treasury', '^IRX': 'Short-Term Rate (13-wk T-Bill)'}
    rows = []
    for ticker, label in proxies.items():
        try:
            df = yf.download(ticker, period='250d', auto_adjust=True, progress=False)
            if not df.empty and len(df) >= 2:
                close      = df['Close'].squeeze().dropna()
                latest_val = round(float(close.iloc[-1]), 2)
                val_3m     = round(float(close.iloc[-63]), 2) if len(close) >= 63 else None
                val_6m     = round(float(close.iloc[-126]),2) if len(close) >= 126 else None
                def fmt(v): return f'{v:.2f}%' if v is not None else 'N/A'
                def arrow(cur, old):
                    if cur is None or old is None: return ''
                    diff = cur - old
                    if   diff >  0.01: return f'▲ +{diff:.2f}%'
                    elif diff < -0.01: return f'▼ {diff:.2f}%'
                    else:              return f'→ {diff:+.2f}%'
                rows.append({
                    'label':   label,
                    'latest':  fmt(latest_val),
                    'val_3m':  fmt(val_3m),
                    'val_6m':  fmt(val_6m),
                    'date_3m': '~3 months ago',
                    'date_6m': '~6 months ago',
                    'trend_3m': arrow(latest_val, val_3m),
                    'trend_6m': arrow(latest_val, val_6m),
                    'note': '',
                })
        except:
            rows.append({'label': label, 'latest': 'N/A', 'val_3m': 'N/A',
                         'val_6m': 'N/A', 'date_3m': '', 'date_6m': '',
                         'trend_3m': '', 'trend_6m': '', 'note': ''})
    rows.append({'label': 'CPI / PCE / Unemployment',
                 'latest': 'Add FRED_API_KEY for full macro data',
                 'val_3m': '', 'val_6m': '', 'date_3m': '', 'date_6m': '',
                 'trend_3m': '', 'trend_6m': '', 'note': ''})
    return rows
 
# Fetch all FRED series (8 obs = ~6 months of monthly data + buffer)
fred_fed    = fred_series('FEDFUNDS', FRED_API_KEY)   # Fed Funds Rate (monthly)
fred_10yr   = fred_series('GS10',     FRED_API_KEY)   # 10-Year Treasury (monthly)
fred_unemp  = fred_series('UNRATE',   FRED_API_KEY)   # Unemployment Rate (monthly)
fred_cpi    = fred_series('CPIAUCSL', FRED_API_KEY)   # CPI index level (monthly)
fred_pce    = fred_series('PCEPI',    FRED_API_KEY)   # PCE index level (monthly)
 
def cpi_yoy_row(obs, label='CPI (YoY %)'):
    """CPI/PCE are index levels — convert to YoY % change.
    Need 13+ obs (current month + 12 months prior) for full comparison."""
    # Re-fetch with more history for YoY calc
    return None  # placeholder; handled below with extended fetch
 
def fred_yoy_series(series_id, api_key):
    """Fetch 20 months of data and compute YoY % changes.
    Returns list of (date, yoy_pct) tuples, newest first."""
    if not api_key: return None
    url = (f'https://api.stlouisfed.org/fred/series/observations'
           f'?series_id={series_id}&api_key={api_key}&file_type=json'
           f'&sort_order=desc&limit=20')
    try:
        r   = requests.get(url, timeout=15)
        obs = r.json().get('observations', [])
        vals = [(o['date'], float(o['value'])) for o in obs
                if o['value'] not in ('.', '', None)]
        if len(vals) < 14: return None
        # vals[0]=latest, vals[12]=12 months ago, vals[15]=15 months ago, vals[18]=18 months ago
        yoy_rows = []
        for idx, months_back_yoy in [(0,12),(3,12),(6,12)]:
            if idx + months_back_yoy < len(vals):
                cur  = vals[idx][1]
                base = vals[idx + months_back_yoy][1]
                pct  = round((cur / base - 1) * 100, 2)
                yoy_rows.append((vals[idx][0], pct))
        return yoy_rows if yoy_rows else None
    except:
        return None
 
fred_cpi_yoy = fred_yoy_series('CPIAUCSL', FRED_API_KEY)
fred_pce_yoy = fred_yoy_series('PCEPI',    FRED_API_KEY)
 
if FRED_API_KEY:
    macro_rows_data = [
        macro_row('Fed Funds Rate',   fred_fed,   suffix='%'),
        macro_row('10-Yr Treasury',   fred_10yr,  suffix='%'),
        macro_row('Unemployment',     fred_unemp, suffix='%'),
        macro_row('CPI (YoY %)',      fred_cpi_yoy, suffix='%',
                  note='Year-over-year % change'),
        macro_row('PCE (YoY %)',      fred_pce_yoy, suffix='%',
                  note='Year-over-year % change'),
    ]
else:
    macro_rows_data = yf_macro_fallback()
 
# =============================================================================
# NEWS via NewsAPI
# =============================================================================
 
def fetch_news(query, api_key, page_size=4):
    if not api_key:
        return []
    url = (f"https://newsapi.org/v2/everything?q={query}"
           f"&language=en&sortBy=publishedAt&pageSize={page_size}"
           f"&apiKey={api_key}")
    try:
        r = requests.get(url, timeout=10)
        articles = r.json().get('articles', [])
        return [{'title': a['title'], 'url': a['url'],
                 'source': a['source']['name'],
                 'published': a['publishedAt'][:10]}
                for a in articles if a.get('title') and '[Removed]' not in a['title']]
    except:
        return []
 
geo_news   = fetch_news('geopolitical conflict war sanctions', NEWS_API_KEY)
macro_news = fetch_news('inflation interest rates federal reserve economy', NEWS_API_KEY)
 
# =============================================================================
# BUILD PLOTLY CHART
# =============================================================================
 
now       = datetime.now()
ytd_start = datetime(now.year, 1, 1)
timeframes = [
    ("All",     start_date,                   end_date),
    ("10-Year", now - timedelta(days=10*365), end_date),
    ("5-Year",  now - timedelta(days=5*365),  end_date),
    ("3-Year",  now - timedelta(days=3*365),  end_date),
    ("1-Year",  now - timedelta(days=365),    end_date),
    ("YTD",     ytd_start,                    end_date),
]
 
def yr(start, end, series):
    sub = series.loc[start:end]
    if sub.empty: return [series.min()*0.95, series.max()*1.05]
    return [sub.min()*0.95, sub.max()*1.05]
 
# Convert index to ISO date strings — required for correct browser rendering
bp_dates    = [d.strftime('%Y-%m-%d') for d in blended_price.index]
ema20_dates = [d.strftime('%Y-%m-%d') for d in ema_20.index]
bt_dates    = [d.strftime('%Y-%m-%d') for d in bt_results.index]
 
# Chart 1 — Blended price + signals
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=bp_dates, y=blended_price.tolist(), mode='lines',
    name='Blended Price', line=dict(width=2, color='#60a5fa'),
    hovertemplate='%{x}<br>Price: %{y:.2f}<extra></extra>'))
for ema, lbl, col in [(ema_20,'20 EMA','#f87171'),(ema_50,'50 EMA','#4ade80'),
                       (ema_100,'100 EMA','#c084fc'),(ema_200,'200 EMA','#fbbf24')]:
    fig1.add_trace(go.Scatter(
        x=[d.strftime('%Y-%m-%d') for d in ema.index],
        y=ema.tolist(), mode='lines', name=lbl,
        line=dict(width=1, dash='dot', color=col), opacity=0.7, hoverinfo='skip'))
if buy_signals:
    fig1.add_trace(go.Scatter(
        x=[s[0].strftime('%Y-%m-%d') for s in buy_signals],
        y=[s[1] for s in buy_signals],
        mode='markers', name='Buy', marker=dict(color='#4ade80', size=10, symbol='triangle-up'),
        hovertemplate='BUY<br>%{x}<br>%{y:.2f}<extra></extra>'))
if reduction_signals:
    fig1.add_trace(go.Scatter(
        x=[s[0].strftime('%Y-%m-%d') for s in reduction_signals],
        y=[s[1] for s in reduction_signals],
        mode='markers', name='Reduce', marker=dict(color='#f87171', size=10, symbol='triangle-down'),
        hovertemplate='REDUCE<br>%{x}<br>%{y:.2f}<extra></extra>'))
 
# Pre-compute initial y-range (full dataset)
bp_yrange = [float(blended_price.min()) * 0.95, float(blended_price.max()) * 1.05]
 
def yr_str(ts, te, series):
    """y-range using string date filtering."""
    ts_s = pd.Timestamp(ts).strftime('%Y-%m-%d')
    te_s = pd.Timestamp(te).strftime('%Y-%m-%d')
    sub  = series.loc[ts_s:te_s]
    if sub.empty: return [float(series.min())*0.95, float(series.max())*1.05]
    return [float(sub.min())*0.95, float(sub.max())*1.05]
 
buttons1 = [dict(label=lbl, method='relayout',
    args=[{'xaxis.range': [pd.Timestamp(ts).strftime('%Y-%m-%d'),
                           pd.Timestamp(te).strftime('%Y-%m-%d')],
           'yaxis.range': yr_str(ts, te, blended_price)}])
    for lbl, ts, te in timeframes]
fig1.update_layout(
    paper_bgcolor='#0f172a', plot_bgcolor='#1e293b', font=dict(color='#e2e8f0'),
    margin=dict(l=55, r=10, t=80, b=20), height=460, autosize=True,
    xaxis=dict(title='Date', gridcolor='#334155', type='date',
               range=[bp_dates[0], bp_dates[-1]]),
    yaxis=dict(title='Blended Price', gridcolor='#334155', range=bp_yrange),
    legend=dict(orientation='h', y=1.0, x=0.0, xanchor='left',
                yanchor='bottom', font=dict(size=11), bgcolor='rgba(0,0,0,0)'),
    hovermode='x unified',
    updatemenus=[dict(type='dropdown', direction='down',
        x=1.0, y=1.12, xanchor='right', yanchor='top',
        buttons=buttons1, bgcolor='#334155', bordercolor='#64748b',
        font=dict(color='white'), showactive=True)]
)
chart1_html = plot(fig1, output_type='div', include_plotlyjs=False)
 
# Chart 2 — Portfolio vs benchmark
pv_list   = bt_results['Portfolio_Value'].tolist()
bv_list   = bt_results['Benchmark_Value'].tolist()
exp_list  = bt_results['Exposure_Pct'].tolist()
pv_yrange = [min(min(pv_list), min(bv_list)) * 0.93,
             max(max(pv_list), max(bv_list)) * 1.07]
 
fig2 = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3],
    vertical_spacing=0.06)
fig2.add_trace(go.Scatter(x=bt_dates, y=pv_list,
    mode='lines', name='Strategy', line=dict(width=2, color='#60a5fa'),
    hovertemplate='%{x}<br>$%{y:,.0f}<extra></extra>'), row=1, col=1)
fig2.add_trace(go.Scatter(x=bt_dates, y=bv_list,
    mode='lines', name='SPY B&H', line=dict(width=2, color='#fb923c', dash='dash'),
    hovertemplate='%{x}<br>$%{y:,.0f}<extra></extra>'), row=1, col=1)
fig2.add_trace(go.Scatter(x=bt_dates, y=exp_list,
    mode='lines', name='Exposure %', line=dict(width=1.5, color='#fbbf24'),
    fill='tozeroy', fillcolor='rgba(251,191,36,0.12)',
    hovertemplate='%{x}<br>%{y:.0f}%<extra></extra>'), row=2, col=1)
 
def yr_str_combined(ts, te, s1, s2):
    ts_s=pd.Timestamp(ts).strftime('%Y-%m-%d'); te_s=pd.Timestamp(te).strftime('%Y-%m-%d')
    sub1=s1.loc[ts_s:te_s]; sub2=s2.loc[ts_s:te_s]
    combined=pd.concat([sub1,sub2]).dropna()
    if combined.empty: combined=pd.concat([s1,s2]).dropna()
    return [float(combined.min())*0.93, float(combined.max())*1.07]
 
buttons2 = [dict(label=lbl, method='relayout',
    args=[{'xaxis.range': [pd.Timestamp(ts).strftime('%Y-%m-%d'),
                           pd.Timestamp(te).strftime('%Y-%m-%d')],
           'yaxis.range': yr_str_combined(ts, te,
               bt_results['Portfolio_Value'], bt_results['Benchmark_Value'])}])
    for lbl, ts, te in timeframes]
fig2.update_layout(
    paper_bgcolor='#0f172a', plot_bgcolor='#1e293b', font=dict(color='#e2e8f0'),
    margin=dict(l=55, r=10, t=80, b=20), height=480, autosize=True,
    legend=dict(orientation='h', y=1.0, x=0.0, xanchor='left',
                yanchor='bottom', font=dict(size=11), bgcolor='rgba(0,0,0,0)'),
    hovermode='x unified',
    updatemenus=[dict(type='dropdown', direction='down',
        x=1.0, y=1.08, xanchor='right', yanchor='top',
        buttons=buttons2, bgcolor='#334155', bordercolor='#64748b',
        font=dict(color='white'), showactive=True)]
)
fig2.update_xaxes(type='date', row=1, col=1)
fig2.update_xaxes(type='date', row=2, col=1)
fig2.update_yaxes(title_text='Portfolio Value ($)', gridcolor='#334155',
    range=pv_yrange, row=1, col=1)
fig2.update_yaxes(title_text='Exposure %', range=[0,110], gridcolor='#334155', row=2, col=1)
chart2_html = plot(fig2, output_type='div', include_plotlyjs=False)
 
# =============================================================================
# HELPER — HTML table builder
# =============================================================================
 
def html_table(headers, rows, col_colors=None):
    """col_colors: dict of col_index -> function(val) -> css color string"""
    th = ''.join(f'<th>{h}</th>' for h in headers)
    body = ''
    for row in rows:
        tds = ''
        for ci, cell in enumerate(row):
            style = ''
            if col_colors and ci in col_colors:
                style = f' style="color:{col_colors[ci](cell)}"'
            tds += f'<td{style}>{cell}</td>'
        body += f'<tr>{tds}</tr>'
    return f'<table><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table>'
 
 
def pct_color(val):
    try:
        return '#4ade80' if float(str(val).replace('%','')) >= 0 else '#f87171'
    except:
        return '#e2e8f0'
 
def signal_badge(state):
    # state: 'Buy', 'Reduce', 'Defensive', 'Risk-On'
    configs = {
        'Buy':       ('#4ade80', '#0f172a', 'BUY'),
        'Reduce':    ('#f87171', '#0f172a', 'REDUCE'),
        'Defensive': ('#fbbf24', '#0f172a', 'DEFENSIVE'),
        'Risk-On':   ('#60a5fa', '#0f172a', 'RISK-ON'),
    }
    bg, fg, label = configs.get(state, ('#94a3b8', '#0f172a', state.upper()))
    return f'<span style="background:{bg};color:{fg};padding:6px 18px;border-radius:20px;font-weight:700;font-size:1.1rem">{label}</span>'
 
today_sig  = metrics['today_signal']
today_date = signals_df.index[-1].strftime('%B %d, %Y')
 
# Find the last Buy and last Reduce signal rows
buy_rows    = signals_df[signals_df['Signal'] == 'Buy']
reduce_rows = signals_df[signals_df['Signal'] == 'Reduce']
 
# Determine current state for the banner badge:
#   Buy     = today's signal is Buy
#   Reduce  = today's signal is Reduce
#   Defensive = last signal was Reduce (and no Buy since)
#   Risk-On   = last signal was Buy (and no Reduce since)
if today_sig == 'Buy':
    banner_state = 'Buy'
elif today_sig == 'Reduce':
    banner_state = 'Reduce'
else:
    # Determine which signal came most recently
    last_buy_idx    = buy_rows.index[-1]    if not buy_rows.empty    else None
    last_reduce_idx = reduce_rows.index[-1] if not reduce_rows.empty else None
    if last_buy_idx is None and last_reduce_idx is None:
        banner_state = 'Risk-On'
    elif last_reduce_idx is None:
        banner_state = 'Risk-On'
    elif last_buy_idx is None:
        banner_state = 'Defensive'
    else:
        banner_state = 'Risk-On' if last_buy_idx > last_reduce_idx else 'Defensive'
 
# Use S&P 500 index (^GSPC) to show actual SPX level at each signal date
def spx_at_date(date):
    """Safely retrieve SPX level at a given date using .asof()."""
    try:
        spx_idx = spx_price.copy()
        spx_idx.index = pd.DatetimeIndex(spx_idx.index).tz_localize(None)
        ts  = pd.Timestamp(date).tz_localize(None)
        val = spx_idx.asof(ts)
        return float(val) if val is not None and not pd.isna(val) else None
    except Exception:
        return None
 
if not buy_rows.empty:
    last_buy_date     = buy_rows.index[-1].strftime('%B %d, %Y')
    last_buy_sig_date = buy_rows.index[-1]
    spx_on_buy        = spx_at_date(last_buy_sig_date)
    last_buy_spy      = f'S&P 500: {spx_on_buy:,.2f}' if spx_on_buy else ''
    last_buy_cond     = buy_rows['Condition'].iloc[-1] or ''
else:
    last_buy_date = last_buy_spy = last_buy_cond = 'N/A'
 
if not reduce_rows.empty:
    last_reduce_date     = reduce_rows.index[-1].strftime('%B %d, %Y')
    last_reduce_sig_date = reduce_rows.index[-1]
    spx_on_reduce        = spx_at_date(last_reduce_sig_date)
    last_reduce_spy      = f'S&P 500: {spx_on_reduce:,.2f}' if spx_on_reduce else ''
    last_reduce_cond     = reduce_rows['Condition'].iloc[-1] or ''
else:
    last_reduce_date = last_reduce_spy = last_reduce_cond = 'N/A'
 
# =============================================================================
# ASSEMBLE HTML
# =============================================================================
 
def fmt_pct(v):
    sign = '+' if v >= 0 else ''
    color = '#4ade80' if v >= 0 else '#f87171'
    return f'<span style="color:{color}">{sign}{v:.2f}%</span>'
 
def metric_card(label, value, sub=''):
    return f'''
    <div class="card">
      <div class="card-label">{label}</div>
      <div class="card-value">{value}</div>
      {"<div class='card-sub'>" + sub + "</div>" if sub else ""}
    </div>'''
 
# Returns table rows
# 1, 3, 5-year figures are annualized; YTD and cumulative are not.
s1yr_s = ann_period_return(strat_v, 252)
s1yr_b = ann_period_return(bench_v, 252)
s3yr_s = ann_period_return(strat_v, 756)
s3yr_b = ann_period_return(bench_v, 756)
s5yr_s = ann_period_return(strat_v, 1260)
s5yr_b = ann_period_return(bench_v, 1260)
 
ret_headers = ['Period', 'Strategy', 'SPY B&H', 'Alpha']
ret_rows = [
    ['YTD',
        fmt_pct(metrics['strat']['ytd']),
        fmt_pct(metrics['bench']['ytd']),
        fmt_pct(metrics['strat']['ytd'] - metrics['bench']['ytd'])],
    ['1 Year (Ann.)',
        fmt_pct(s1yr_s),
        fmt_pct(s1yr_b),
        fmt_pct(s1yr_s - s1yr_b)],
    ['3 Year (Ann.)',
        fmt_pct(s3yr_s),
        fmt_pct(s3yr_b),
        fmt_pct(s3yr_s - s3yr_b)],
    ['5 Year (Ann.)',
        fmt_pct(s5yr_s),
        fmt_pct(s5yr_b),
        fmt_pct(s5yr_s - s5yr_b)],
    ['Cumulative Since Feb. 2015',
        fmt_pct(metrics['strat']['all']),
        fmt_pct(metrics['bench']['all']),
        fmt_pct(metrics['strat']['all'] - metrics['bench']['all'])],
    ['Ann. Return Since Feb. 2015',
        fmt_pct(metrics['strat']['ann']),
        fmt_pct(metrics['bench']['ann']),
        fmt_pct(metrics['strat']['ann'] - metrics['bench']['ann'])],
    ['Max Drawdown',
        fmt_pct(metrics['strat']['mdd']),
        fmt_pct(metrics['bench']['mdd']),
        '—'],
]
returns_table = html_table(ret_headers, ret_rows)
 
# Annual table
ann_headers = ['Year', 'Strategy', 'SPY', 'Alpha']
ann_rows = [[int(r['Year']), fmt_pct(r['Strategy']), fmt_pct(r['SPY']), fmt_pct(r['Alpha'])]
            for _, r in annual_df.iterrows()]
annual_table = html_table(ann_headers, ann_rows)
 
# Build sector & stock tables for each timeframe — embedded as JSON for JS dropdown
import json as _json
 
def build_table_data(returns_dict):
    """Convert {label: [(name,pct)]} into JSON-safe dict for JS."""
    out = {}
    for label, rows in returns_dict.items():
        out[label] = [{'rank': i+1, 'name': r[0], 'pct': r[1]} for i, r in enumerate(rows)]
    return _json.dumps(out)
 
sector_json = build_table_data(sector_returns_all)
stock_json  = build_table_data(stock_returns_all)
 
# Macro table
mac_headers = ['Indicator', 'Latest', '3-Month Ago', 'vs 3M', '6-Month Ago', 'vs 6M']
mac_rows = [
    [
        m['label'],
        m['latest'],
        f"{m['val_3m']}<br><span style='font-size:0.72rem;color:#64748b'>{m['date_3m']}</span>",
        m['trend_3m'],
        f"{m['val_6m']}<br><span style='font-size:0.72rem;color:#64748b'>{m['date_6m']}</span>",
        m['trend_6m'],
    ]
    for m in macro_rows_data
]
macro_table = html_table(mac_headers, mac_rows)
 
# News sections
def news_list(articles, fallback):
    if not articles:
        return f'<p class="muted">{fallback}</p>'
    items = ''
    for a in articles:
        items += f'''
        <div class="news-item">
          <a href="{a["url"]}" target="_blank" rel="noopener">{a["title"]}</a>
          <span class="news-meta">{a["source"]} &bull; {a["published"]}</span>
        </div>'''
    return items
 
geo_html   = news_list(geo_news,   'Add NEWS_API_KEY as a GitHub Actions secret to enable live news headlines. Sign up free at newsapi.org/register')
macro_html = news_list(macro_news, 'Add NEWS_API_KEY as a GitHub Actions secret to enable live news headlines. Sign up free at newsapi.org/register')
 
# Background animation script stored as plain string (outside f-string to avoid
# escaping conflicts between Python's {{ }} and JavaScript's { })
bg_script = '''<canvas id="bg-canvas"></canvas>
<script>
(function() {
  const c=document.getElementById('bg-canvas'),ctx=c.getContext('2d');
  let W,H,stars=[];
  function resize(){W=c.width=window.innerWidth;H=c.height=window.innerHeight;}
  resize(); window.addEventListener('resize',resize);
  for(let i=0;i<180;i++) stars.push({
    x:Math.random()*2000,y:Math.random()*2000,
    r:Math.random()*1.4+0.3,a:Math.random()*Math.PI*2,
    s:Math.random()*0.004+0.001,op:Math.random()*0.6+0.2
  });
  let t=0;
  function draw(){
    ctx.clearRect(0,0,W,H);
    const g=ctx.createLinearGradient(0,0,W,H);
    g.addColorStop(0,'#060d1f');g.addColorStop(0.5,'#0a1628');g.addColorStop(1,'#060d1f');
    ctx.fillStyle=g;ctx.fillRect(0,0,W,H);
    ctx.strokeStyle='rgba(30,80,140,0.08)';ctx.lineWidth=1;
    const grid=80,off=(t*8)%grid;
    for(let x=-grid+off;x<W+grid;x+=grid){ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,H);ctx.stroke();}
    for(let y=-grid+off;y<H+grid;y+=grid){ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(W,y);ctx.stroke();}
    stars.forEach(s=>{s.a+=s.s;const px=s.x%W,py=s.y%H;
      const op=s.op*(0.6+0.4*Math.sin(s.a));
      ctx.beginPath();ctx.arc(px,py,s.r,0,Math.PI*2);
      ctx.fillStyle='rgba(150,200,255,'+op+')';ctx.fill();
    });
    [[W*0.2,H*0.3,'rgba(30,100,200,0.04)'],[W*0.8,H*0.7,'rgba(60,30,180,0.04)']].forEach(function(o){
      const rg=ctx.createRadialGradient(o[0],o[1],0,o[0],o[1],300);
      rg.addColorStop(0,o[2]);rg.addColorStop(1,'transparent');
      ctx.fillStyle=rg;ctx.fillRect(0,0,W,H);
    });
    t+=0.003;requestAnimationFrame(draw);
  }
  draw();
})();
</script>'''
 
html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Trading Strategy Dashboard</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: #060d1f; color: #e2e8f0; font-family: 'Segoe UI', system-ui, sans-serif; padding: 0 0 60px; position: relative; min-height: 100vh; }}
  #bg-canvas {{ position: fixed; top: 0; left: 0; width: 100%; height: 100%; z-index: -1; pointer-events: none; }}
  a {{ color: #60a5fa; text-decoration: none; }} a:hover {{ text-decoration: underline; }}
 
  /* Header */
  .header {{ background: linear-gradient(135deg, #0d1f3c 0%, #060d1f 100%);
    border-bottom: 1px solid #334155; padding: 24px 32px; display: flex;
    justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px; }}
  .header h1 {{ font-size: 1.5rem; font-weight: 700; color: #f1f5f9; }}
  .header .updated {{ font-size: 0.8rem; color: #64748b; }}
 
  /* Signal banner */
  .signal-banner {{ background: rgba(10,20,40,0.92); border-bottom: 1px solid #1e3a5f;
    padding: 20px 32px; display: flex; align-items: stretch; gap: 0; flex-wrap: wrap; }}
  .banner-item {{ display: flex; flex-direction: column; justify-content: flex-start;
    padding: 0 28px 0 0; margin-right: 28px; border-right: 1px solid #1e3a5f; }}
  .banner-item:last-child {{ border-right: none; margin-right: 0; padding-right: 0; }}
  .signal-banner .label {{ font-size: 0.9rem; color: #94a3b8; text-transform: uppercase;
    letter-spacing: 0.08em; }}
  .signal-banner .exposure {{ font-size: 1rem; color: #e2e8f0; }}
 
  /* Main layout */
  .main {{ max-width: 1400px; margin: 0 auto; padding: 32px 24px; display: grid;
    gap: 28px; }}
 
  /* Sections */
  .section {{ background: rgba(15,23,42,0.85); backdrop-filter: blur(4px); border: 1px solid #1e3a5f; border-radius: 12px;
    padding: 24px; }}
  .section h2 {{ font-size: 1.05rem; font-weight: 600; color: #94a3b8;
    text-transform: uppercase; letter-spacing: 0.07em; margin-bottom: 18px;
    padding-bottom: 10px; border-bottom: 1px solid #334155; }}
 
  /* Metric cards */
  .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 14px; }}
  .card {{ background: #0f172a; border: 1px solid #334155; border-radius: 8px;
    padding: 16px; }}
  .card-label {{ font-size: 0.75rem; color: #64748b; text-transform: uppercase;
    letter-spacing: 0.06em; margin-bottom: 6px; }}
  .card-value {{ font-size: 1.25rem; font-weight: 700; }}
  .card-sub {{ font-size: 0.78rem; color: #64748b; margin-top: 4px; }}
 
  /* Tables */
  table {{ width: 100%; border-collapse: collapse; font-size: 0.88rem; }}
  #sector-table {{ font-size: 1.05rem; }}
  #sector-table td, #sector-table th {{ padding: 13px 14px; }}
  th {{ background: #0f172a; color: #64748b; text-transform: uppercase;
    font-size: 0.72rem; letter-spacing: 0.06em; padding: 10px 12px; text-align: left;
    border-bottom: 1px solid #334155; }}
  td {{ padding: 9px 12px; border-bottom: 1px solid #1e293b; color: #e2e8f0; }}
  tr:last-child td {{ border-bottom: none; }}
  tr:hover td {{ background: #0f172a; }}
 
  /* Two-column grid for news/market */
  .two-col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 28px; }}
  @media (max-width: 860px) {{ .two-col {{ grid-template-columns: 1fr; }} }}
 
  /* News */
  .news-item {{ padding: 10px 0; border-bottom: 1px solid #334155; }}
  .news-item:last-child {{ border-bottom: none; }}
  .news-item a {{ font-size: 0.88rem; color: #e2e8f0; line-height: 1.4; }}
  .news-item a:hover {{ color: #60a5fa; }}
  .news-meta {{ display: block; font-size: 0.73rem; color: #64748b; margin-top: 3px; }}
  .muted {{ color: #64748b; font-size: 0.85rem; padding: 12px 0; }}
 
  /* Chart container */
  .chart-wrap {{ overflow-x: auto; width: 100%; }}
  .chart-wrap > div {{ width: 100% !important; }}
</style>
</head>
<body>
{bg_script}
 
<div class="header">
  <h1>📈 Trading Strategy Dashboard</h1>
  <span class="updated">Last updated: {datetime.now().strftime('%B %d, %Y at %H:%M UTC')}</span>
</div>
 
<div class="signal-banner">
 
  <!-- Today's signal -->
  <div class="banner-item">
    <div class="label">Current Signal &nbsp;·&nbsp; {today_date}</div>
    <div style="margin-top:8px">{signal_badge(banner_state)}</div>
  </div>
 
  <!-- Last Buy signal -->
  <div class="banner-item">
    <div class="label">Last Buy Signal</div>
    <div style="margin-top:6px;font-size:1rem;font-weight:700;color:#4ade80">{last_buy_date}</div>
    <div style="font-size:0.78rem;color:#64748b;margin-top:4px">{last_buy_spy}</div>
  </div>
 
  <!-- Last Reduce signal -->
  <div class="banner-item">
    <div class="label">Last Reduce Signal</div>
    <div style="margin-top:6px;font-size:1rem;font-weight:700;color:#f87171">{last_reduce_date}</div>
    <div style="font-size:0.78rem;color:#64748b;margin-top:4px">{last_reduce_spy}</div>
  </div>
 
  <!-- Exposure -->
  <div class="banner-item">
    <div class="label">Current Exposure</div>
    <div style="font-size:1.4rem;font-weight:700;margin-top:6px">{metrics['today_exposure']:.0f}%</div>
  </div>
 
  <!-- Starting investment -->
  <div class="banner-item">
    <div class="label">Starting Investment</div>
    <div style="font-size:1.4rem;font-weight:700;margin-top:6px">$1,000,000</div>
    <div style="font-size:0.78rem;color:#64748b;margin-top:4px">February 1, 2015</div>
  </div>
 
  <!-- Current portfolio value -->
  <div class="banner-item">
    <div class="label">Strategy Portfolio Value</div>
    <div style="font-size:1.4rem;font-weight:700;margin-top:6px">${metrics['strat']['end_val']:,.0f}</div>
    <div style="font-size:0.78rem;color:#64748b;margin-top:4px">{today_date}</div>
  </div>
 
</div>
 
<div class="main">
 
  <!-- PERFORMANCE CARDS -->
  <div class="section">
    <h2>Performance Overview</h2>
    <div class="cards">
      {metric_card('Strategy YTD', fmt_pct(metrics['strat']['ytd']))}
      {metric_card('Strategy 1-Year', fmt_pct(metrics['strat']['1yr']))}
      {metric_card('Strategy 5-Year', fmt_pct(metrics['strat']['5yr']))}
      {metric_card('Strategy All-Time', fmt_pct(metrics['strat']['all']), 'Since Feb 2015')}
      {metric_card('Ann. Return', fmt_pct(metrics['strat']['ann']))}
      {metric_card('Max Drawdown', fmt_pct(metrics['strat']['mdd']))}
      {metric_card('SPY YTD', fmt_pct(metrics['bench']['ytd']))}
      {metric_card('SPY All-Time', fmt_pct(metrics['bench']['all']), 'Since Feb 2015')}
    </div>
  </div>
 
  <!-- RETURNS TABLE -->
  <div class="two-col">
    <div class="section">
      <h2>Trailing Returns vs SPY</h2>
      {returns_table}
    </div>
    <div class="section">
      <h2>Annual Performance</h2>
      {annual_table}
    </div>
  </div>
 
  <!-- CHART 1 -->
  <div class="section">
    <h2>Trading Signal History</h2>
    <div class="chart-wrap">{chart1_html}</div>
  </div>
 
  <!-- CHART 2 -->
  <div class="section">
    <h2>Portfolio Value vs SPY Benchmark</h2>
    <div class="chart-wrap">{chart2_html}</div>
  </div>
 
  <!-- MARKET DATA -->
  <div class="two-col">
    <div class="section">
      <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:18px;padding-bottom:10px;border-bottom:1px solid #334155">
        <h2 style="margin:0;border:none;padding:0">Top 3 Sector Performance</h2>
        <select id="sector-tf" onchange="updateSectorTable()" style="background:#0f172a;color:#e2e8f0;border:1px solid #334155;border-radius:6px;padding:4px 10px;font-size:0.85rem;cursor:pointer">
          <option>1 Month</option><option>6 Months</option><option>1 Year</option>
        </select>
      </div>
      <table id="sector-table">
        <thead><tr><th>Rank</th><th>Sector</th><th>Return</th></tr></thead>
        <tbody id="sector-tbody"></tbody>
      </table>
    </div>
    <div class="section">
      <h2>Macro Indicators</h2>
      {macro_table}
      <p class="muted" style="margin-top:12px">Source: FRED (St. Louis Fed) &bull; 3M/6M trend shows change vs. that period. Add FRED_API_KEY secret to enable live data.</p>
    </div>
  </div>
 
  <!-- TOP 20 STOCKS -->
  <div class="section">
    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:18px;padding-bottom:10px;border-bottom:1px solid #334155">
      <h2 style="margin:0;border:none;padding:0">Top 20 Stock Performers</h2>
      <select id="stock-tf" onchange="updateStockTable()" style="background:#0f172a;color:#e2e8f0;border:1px solid #334155;border-radius:6px;padding:4px 10px;font-size:0.85rem;cursor:pointer">
        <option>1 Month</option><option>6 Months</option><option>1 Year</option>
      </select>
    </div>
    <table id="stock-table">
      <thead><tr><th>Rank</th><th>Ticker</th><th>Return</th></tr></thead>
      <tbody id="stock-tbody"></tbody>
    </table>
  </div>
 
  <!-- NEWS -->
  <div class="two-col">
    <div class="section">
      <h2>🌍 Geopolitical News</h2>
      {geo_html}
    </div>
    <div class="section">
      <h2>📊 Macro &amp; Economic News</h2>
      {macro_html}
    </div>
  </div>
 
</div>
 
<script>
const SECTOR_DATA = {sector_json};
const STOCK_DATA  = {stock_json};
 
function colorPct(pct) {{
  const sign  = pct >= 0 ? '+' : '';
  const color = pct >= 0 ? '#4ade80' : '#f87171';
  return `<span style="color:${{color}}">${{sign}}${{pct.toFixed(2)}}%</span>`;
}}
 
function updateSectorTable() {{
  const tf   = document.getElementById('sector-tf').value;
  const rows = SECTOR_DATA[tf] || [];
  document.getElementById('sector-tbody').innerHTML = rows.map(r =>
    `<tr><td>${{r.rank}}</td><td>${{r.name}}</td><td>${{colorPct(r.pct)}}</td></tr>`
  ).join('');
}}
 
function updateStockTable() {{
  const tf   = document.getElementById('stock-tf').value;
  const rows = STOCK_DATA[tf] || [];
  document.getElementById('stock-tbody').innerHTML = rows.map(r =>
    `<tr><td>${{r.rank}}</td><td>${{r.name}}</td><td>${{colorPct(r.pct)}}</td></tr>`
  ).join('');
}}
 
// Populate tables on page load
updateSectorTable();
updateStockTable();
</script>
 
</body>
</html>"""
 
with open('index.html', 'w', encoding='utf-8') as f:
    f.write(html)
 
print(f"Dashboard generated: index.html ({len(html):,} chars)")
print(f"Today's signal: {today_sig} | Exposure: {metrics['today_exposure']}%")
