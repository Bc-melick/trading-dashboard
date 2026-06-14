**Trading Strategy — High Level Overview**


Blended Price Index

Rather than using a single market index, the strategy uses a custom blended price as its signal source:

ETFWeightSPY (S&P 500)50%QQQ (Nasdaq 100)50%

All technical indicators (EMAs, RSI, MACD) are calculated on this blended price, not on any single index.


Technical Indicators Used

IndicatorParametersEMA (Exponential Moving Average)20-day, 50-day, 100-day, 200-dayRSI (Relative Strength Index)14-day windowMACD12-day short EMA, 26-day long EMA, 9-day signal line


Portfolio Holding

When invested, the portfolio holds 100% QQQ. The blended price generates the signals, but all actual buying and selling is executed in QQQ.


Buy Signals

Condition 1 — 100-Day EMA Crossover


Blended price crosses above the 100-day EMA
The 100-day EMA must have been rising for 3 consecutive days on the signal date
If the 100 EMA is still falling on the crossover day, the signal is held as pending and fires on the first subsequent day the 100 EMA turns rising for 3 consecutive days
The pending signal is cancelled if price falls back below the 100-day EMA before the condition is met
RSI must be between 20 and 70
Blended price must be rising (today > yesterday)
Position sizing: 100% entry into QQQ


Condition 2 — 200-Day EMA Crossover


Blended price crosses above the 200-day EMA
The 100-day EMA (not the 200) must have been rising for 3 consecutive days
Same pending/cancellation logic as Condition 1
RSI must be between 20 and 70
Blended price must be rising (today > yesterday)
Position sizing: 100% entry into QQQ


Condition 3 — MACD Bullish Crossover


MACD line crosses above the signal line
The 100-day EMA must have been rising for 3 consecutive days on or after the crossover
If the 100 EMA is still falling, the signal is held as pending and fires on the first day the 100 EMA has been rising 3 consecutive days, provided the MACD line is still above the signal line
The pending signal is cancelled if MACD drops back below the signal line before the condition is met
RSI must be between 20 and 70
Blended price must be rising (today > yesterday)
Position sizing: 100% entry into QQQ


Condition 4 — 50-Day EMA Recovery After Reduce


Requires a prior reduce signal to have occurred
Blended price has been above the 50-day EMA for 4+ consecutive days since the reduce
A MACD bullish crossover also occurs on the same day
Blended price must be rising (today > yesterday)
Position sizing: +10% incremental entry (e.g. 50% → 60%)
Each subsequent 1% rise in the blended price adds another +10% until 100% is reached


Condition 5 — 20-Day EMA Cross (Inverse EMA Alignment)


Only active when the 20-day EMA is below the 100-day EMA (bearish alignment)
Blended price has been above the 20-day EMA for 2+ consecutive days
Blended price must be rising (today > yesterday)
Position sizing: 100% entry into QQQ



Reduce Signals

All reduce conditions require at least 5 trading days to have passed since the last buy signal. No double-signaling (consecutive reduces are blocked).

Condition 1 — Below 50-Day EMA with Bearish EMA Alignment


Blended price has been below the 50-day EMA for 2+ consecutive days
MACD line is below the signal line (reduce is blocked if MACD is still above)
100-day EMA is greater than the 50-day EMA (bearish EMA structure)


Condition 2 — Below 50-Day EMA with Price Drop


Blended price has been below the 50-day EMA for 2+ consecutive days
Price has dropped more than 2.5% from the highest price reached since the last buy
RSI is above 20 (not oversold)
No MACD filter — fires regardless of MACD position


Condition 3 — Below 20-Day EMA with Long-Term Bearish Structure


Blended price has been below the 20-day EMA for 2+ consecutive days
MACD line is below the signal line
200-day EMA is greater than the 50-day EMA (long-term bearish)


Condition 4 — Below 20-Day EMA with Significant Price Drop


Blended price has been below the 20-day EMA for 3+ consecutive days
Price has dropped more than 2.5% from the post-buy high
MACD line is below the signal line



Reduce Position Sizing

Condition on Reduce DateResulting Exposure20-day EMA < 100-day EMA or 20-day EMA < 200-day EMA0% (full exit to cash)Neither condition above is true50% exposure

Exposure stays at 0% until the next buy signal is generated.


Incremental Buying (After Cross-50 Buy Signal)

When Buy Condition 4 (50-day EMA recovery) triggers:


Initial entry: +10% (e.g. from 50% to 60%)
Each subsequent +1% rise in the blended price adds another +10%
Continues until 100% exposure is reached
Any reduce signal during this period cancels the incremental buying



Backtest Parameters

ParameterValueStarting Capital$1,000,000Backtest Start DateFebruary 1, 2015Data Start DateJanuary 1, 2015 (EMA warm-up)Holding VehicleQQQBenchmarkSPY (Buy & Hold)Cash Return0% (uninvested cash earns nothing)


Dashboard Ticker Universe (Top 20 Stock Performers)

~153 liquid tickers across all sectors used for the top performer rankings:

Tech & Software: AAPL, MSFT, NVDA, AMZN, GOOGL, META, TSLA, AVGO, ORCL, ADBE, CRM, NOW, INTU, CSCO, IBM, QCOM, TXN, AMD, INTC, ACN

Semis & Hardware: MU, AMAT, LRCX, KLAC, MRVL, SMCI, PLTR, ARM, DELL, HPQ

Cybersecurity & Cloud: CRWD, PANW, FTNT, NET, ZS, DDOG, SNOW, COIN

Internet & Media: NFLX, UBER, ABNB, BKNG, TTD, ROKU, SPOT, RBLX

Payments & Fintech: V, MA, PYPL, SQ, AXP, COF, HOOD

Financials: JPM, BAC, WFC, GS, MS, BLK, SCHW, C, SPGI, MCO, ICE, CME, PGR, CB, MMC

Healthcare & Biotech: LLY, UNH, JNJ, ABBV, MRK, PFE, TMO, ABT, DHR, SYK, AMGN, GILD, REGN, VRTX, MRNA, ISRG, BSX, ELV, CVS, CI

Consumer Discretionary: WMT, COST, HD, MCD, SBUX, CMG, NKE, LULU, TGT, LOW, TJX, ROST, DKNG, RCL, MAR, HLT, LVS, WYNN

Consumer Staples: PG, KO, PEP, PM, MO, MDLZ, CL, GIS

Industrials: HON, RTX, LMT, BA, CAT, DE, GE, ETN, UPS, FDX, NOC, GD, MMM, EMR, ITW, CSX, UNP, DAL, UAL

Energy: XOM, CVX, COP, OXY, SLB, MPC, PSX, VLO, EOG, DVN

Communication: DIS, CMCSA, T, VZ, TMUS, CHTR, PARA

Real Estate: AMT, PLD, EQIX, CCI, PSA, DLR, O, SPG, VICI

Utilities: NEE, DUK, SO, D, AEP, SRE, EXC

Materials: LIN, APD, SHW, ECL, NEM, FCX, ALB

High-Momentum & Growth: SHOP, MELI, NU, KKR, APO, ARES, F, GM
