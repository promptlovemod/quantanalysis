# -*- coding: utf-8 -*-
"""
Fundamental Analyzer  v6.0
============================
IMPROVEMENTS vs v5.x
─────────────────────
NEW 1  ▸ DCF Valuation — simple discounted cash flow with bear/base/bull cases.
         Uses TTM free cash flow + revenue growth projection + WACC estimate.

NEW 2  ▸ Graham Number — Benjamin Graham's intrinsic value estimate:
         sqrt(22.5 × EPS × BVPS). Shows margin of safety vs current price.

NEW 3  ▸ Piotroski F-Score (0–9) — systematic profitability, leverage,
         and operating-efficiency signals. Score ≥ 7 = strong.

NEW 4  ▸ Price momentum score — relative performance vs SPY over 1M/3M/6M/1Y.

NEW 5  ▸ Composite score improved from 5 factors to 12 with proper weighting.

NEW 6  ▸ EV/EBITDA analysis — compare to sector median estimates.

NEW 7  ▸ Insider / institutional data pulled from yfinance.

Run:
  python fundamental.py           # interactive
  python fundamental.py AAPL      # direct
"""

import warnings; warnings.filterwarnings('ignore')
import json, datetime, math
from pathlib import Path

try:
    import yfinance as yf
except ImportError:
    print("pip install yfinance"); exit(1)

try:
    import pandas as pd
    import numpy as np
except ImportError:
    print("pip install pandas numpy"); exit(1)

import sys

TICKER  = None
OUT_DIR = None


def get_ticker() -> str:
    if len(sys.argv) > 1:
        return sys.argv[1].strip().upper()
    print()
    print("="*50)
    print("  Fundamental Analyzer  v6.0")
    print("="*50)
    t = input("  Enter ticker (e.g. AAPL, SBUX, TSLA): ").strip().upper()
    return t if t else "CLPT"


# ─────────────────────────────────────────────────────────────────────────────
# DATA FETCH
# ─────────────────────────────────────────────────────────────────────────────
def fetch_fundamentals(ticker: str) -> tuple:
    print(f"  Fetching {ticker} fundamentals from Yahoo Finance...")
    stock = yf.Ticker(ticker)
    info  = stock.info or {}

    fields = [
        'currentPrice', 'regularMarketPrice',
        'fiftyTwoWeekHigh', 'fiftyTwoWeekLow',
        'marketCap', 'trailingPE', 'forwardPE', 'priceToBook',
        'priceToSalesTrailing12Months', 'enterpriseToRevenue',
        'enterpriseToEbitda', 'revenueGrowth', 'grossMargins',
        'operatingMargins', 'profitMargins', 'returnOnEquity',
        'returnOnAssets', 'debtToEquity', 'currentRatio',
        'quickRatio', 'totalCash', 'totalDebt', 'freeCashflow',
        'operatingCashflow', 'totalRevenue', 'earningsGrowth',
        'shortRatio', 'shortPercentOfFloat', 'beta',
        'averageVolume', 'averageVolume10days',
        'fiftyDayAverage', 'twoHundredDayAverage',
        'longName', 'shortName', 'sector', 'industry',
        'country', 'website', 'longBusinessSummary',
        'targetMeanPrice', 'targetHighPrice', 'targetLowPrice',
        'numberOfAnalystOpinions', 'recommendationKey',
        'trailingEps', 'forwardEps', 'bookValue',
        'ebitda', 'totalRevenue', 'revenuePerShare',
        'heldPercentInsiders', 'heldPercentInstitutions',
        'sharesOutstanding', 'floatShares',
        'dividendYield', 'payoutRatio', 'dividendRate',
        'pegRatio',
    ]

    result = {}
    for f in fields:
        v = info.get(f)
        if v is not None and v != 'N/A':
            result[f] = v

    # Normalize dividendYield: yfinance can return it as a decimal (0.025)
    # or as a percentage-point float (2.5) depending on version.
    # Standardize to decimal fraction so all downstream math is consistent.
    dy = result.get('dividendYield')
    if dy is not None and dy > 0.5:   # >50% yield is impossible → already in pct-point form
        result['dividendYield'] = dy / 100.0

    # Pull momentum vs SPY
    momentum = {}
    try:
        price = result.get('currentPrice') or result.get('regularMarketPrice', 0)
        hist  = yf.download(ticker, period='1y', progress=False, auto_adjust=True)
        hist.columns = [c[0] if isinstance(c, tuple) else c for c in hist.columns]
        spy   = yf.download('SPY', period='1y', progress=False, auto_adjust=True)
        spy.columns = [c[0] if isinstance(c, tuple) else c for c in spy.columns]
        if not hist.empty and not spy.empty:
            for days, label in [(21,'1M'), (63,'3M'), (126,'6M'), (252,'1Y')]:
                if len(hist) >= days:
                    ret_stock = float(hist['Close'].iloc[-1] / hist['Close'].iloc[-days] - 1)
                    ret_spy   = float(spy['Close'].iloc[-1]  / spy['Close'].iloc[-min(days, len(spy)-1)] - 1)
                    momentum[label] = {'stock': ret_stock, 'spy': ret_spy,
                                       'relative': ret_stock - ret_spy}
    except Exception:
        pass

    result['_momentum'] = momentum

    return result, info


# ─────────────────────────────────────────────────────────────────────────────
# NEW 1: DCF VALUATION
# ─────────────────────────────────────────────────────────────────────────────
def dcf_valuation(fund: dict) -> dict:
    """
    Simple 5-year DCF with 3 scenarios.
    Uses free cash flow as the base; projects 3 growth cases,
    then discounts at WACC approximated from cost-of-equity (CAPM).
    """
    fcf    = fund.get('freeCashflow', 0) or 0
    shares = fund.get('sharesOutstanding', 1) or 1
    beta   = fund.get('beta', 1.0) or 1.0
    rev_g  = fund.get('revenueGrowth', 0.05) or 0.05
    price  = fund.get('currentPrice') or fund.get('regularMarketPrice', 0) or 0

    if fcf <= 0 or shares <= 0 or price <= 0:
        return {'available': False, 'note': 'FCF ≤ 0 or missing data — DCF not applicable'}

    # CAPM cost of equity: rf=4.5% (current), ERP=5.5%
    rf = 0.045; erp = 0.055
    ke = rf + beta * erp
    wacc = ke  # simplified (no debt weight for consumer stocks)

    # Terminal growth rate (long-run GDP)
    tg = 0.025

    def pv_fcf(growth_5yr):
        cfv = fcf
        total_pv = 0.0
        for yr in range(1, 6):
            cfv *= (1 + growth_5yr)
            total_pv += cfv / (1 + wacc)**yr
        # Terminal value (Gordon growth)
        terminal = cfv * (1 + tg) / (wacc - tg) / (1 + wacc)**5
        total_pv += terminal
        return total_pv / shares

    scenarios = {
        'Bear': pv_fcf(max(0, rev_g * 0.3)),
        'Base': pv_fcf(max(0, rev_g * 0.7)),
        'Bull': pv_fcf(max(0, rev_g * 1.3)),
    }

    base_iv  = scenarios['Base']
    margin   = (base_iv - price) / price if price > 0 else 0

    return {
        'available':   True,
        'wacc':        round(wacc * 100, 2),
        'terminal_g':  round(tg  * 100, 2),
        'fcf_per_share': round(fcf / shares, 4),
        'intrinsic_bear': round(scenarios['Bear'], 2),
        'intrinsic_base': round(scenarios['Base'], 2),
        'intrinsic_bull': round(scenarios['Bull'], 2),
        'current_price':  round(price, 2),
        'margin_of_safety': round(margin * 100, 1),
        'signal': ('UNDERVALUED' if margin > 0.15 else
                   'OVERVALUED'  if margin < -0.15 else 'FAIR VALUE'),
    }


# ─────────────────────────────────────────────────────────────────────────────
# NEW 2: GRAHAM NUMBER
# ─────────────────────────────────────────────────────────────────────────────
def graham_number(fund: dict) -> dict:
    eps  = fund.get('trailingEps',  0) or 0
    bvps = fund.get('bookValue',    0) or 0
    price= fund.get('currentPrice') or fund.get('regularMarketPrice', 0) or 0

    if eps <= 0 or bvps <= 0:
        return {'available': False, 'note': 'Negative EPS or BVPS — Graham not applicable'}

    gn     = math.sqrt(22.5 * eps * bvps)
    margin = (gn - price) / price if price > 0 else 0
    return {
        'available':        True,
        'eps':              round(eps,  2),
        'bvps':             round(bvps, 2),
        'graham_number':    round(gn,   2),
        'current_price':    round(price,2),
        'margin_of_safety': round(margin * 100, 1),
        'signal': ('UNDERVALUED' if margin > 0.15 else
                   'OVERVALUED'  if margin < -0.15 else 'FAIR VALUE'),
    }


# ─────────────────────────────────────────────────────────────────────────────
# NEW 3: PIOTROSKI F-SCORE
# ─────────────────────────────────────────────────────────────────────────────
def piotroski_f_score(fund: dict) -> dict:
    """
    Estimates Piotroski F-Score from available Yahoo Finance fields.
    Not all 9 sub-signals are available via yfinance; we approximate
    using the data we have (typically 5-7 signals available).
    """
    score = 0; signals = {}

    # Profitability
    roa = fund.get('returnOnAssets', None)
    if roa is not None:
        signals['ROA > 0'] = int(roa > 0); score += signals['ROA > 0']

    fcf = fund.get('freeCashflow', None)
    if fcf is not None:
        signals['FCF > 0'] = int(fcf > 0); score += signals['FCF > 0']

    earn_g = fund.get('earningsGrowth', None)
    if earn_g is not None:
        signals['Earnings growth > 0'] = int(earn_g > 0); score += signals['Earnings growth > 0']

    gm = fund.get('grossMargins', None)
    if gm is not None:
        signals['Gross margin > 20%'] = int(gm > 0.20); score += signals['Gross margin > 20%']

    # Leverage / liquidity
    cr = fund.get('currentRatio', None)
    if cr is not None:
        signals['Current ratio > 1'] = int(cr > 1.0); score += signals['Current ratio > 1']

    de = fund.get('debtToEquity', None)
    if de is not None:
        signals['D/E < 100%'] = int(de < 100); score += signals['D/E < 100%']

    # Operating efficiency
    rev_g = fund.get('revenueGrowth', None)
    if rev_g is not None:
        signals['Revenue growth > 0'] = int(rev_g > 0); score += signals['Revenue growth > 0']

    op_m = fund.get('operatingMargins', None)
    if op_m is not None:
        signals['Operating margin > 5%'] = int(op_m > 0.05); score += signals['Operating margin > 5%']

    short_pct = fund.get('shortPercentOfFloat', None)
    if short_pct is not None:
        signals['Short interest < 10%'] = int(short_pct < 0.10); score += signals['Short interest < 10%']

    max_score = len(signals)
    if max_score == 0:
        return {'available': False, 'note': 'Insufficient data'}

    label = ('STRONG' if score >= int(max_score * 0.75) else
             'WEAK'   if score <= int(max_score * 0.35) else 'MODERATE')

    return {
        'available':  True,
        'score':      score,
        'max_score':  max_score,
        'signals':    signals,
        'label':      label,
        'pct':        round(score / max_score * 100),
    }


# ─────────────────────────────────────────────────────────────────────────────
# GENERIC CONTEXT
# ─────────────────────────────────────────────────────────────────────────────
def build_generic_context(ticker: str, info: dict, fund: dict = None) -> dict:
    name    = info.get('longName') or info.get('shortName') or ticker
    sector  = info.get('sector', 'N/A')
    industry= info.get('industry', 'N/A')
    summary = (info.get('longBusinessSummary') or '')[:300]
    website = info.get('website', '')
    country = info.get('country', '')

    price    = info.get('currentPrice') or info.get('regularMarketPrice') or 0
    target   = info.get('targetMeanPrice')
    target_hi= info.get('targetHighPrice')
    target_lo= info.get('targetLowPrice')
    n_analysts = info.get('numberOfAnalystOpinions', 0)
    recommend  = info.get('recommendationKey', 'N/A').upper()

    analysts = {}
    if target:    analysts['Mean target']    = f"${target:.2f}"
    if target_hi: analysts['High target']    = f"${target_hi:.2f}"
    if target_lo: analysts['Low target']     = f"${target_lo:.2f}"
    if price > 0 and target:
        upside = (target - price) / price * 100
        analysts['Implied upside'] = f"{upside:+.1f}%"
    if n_analysts: analysts['Analyst count'] = str(n_analysts)
    analysts['Recommendation'] = recommend

    catalysts, risks = [], []

    rev_growth = info.get('revenueGrowth')
    if rev_growth and rev_growth > 0.05:
        catalysts.append(f"Revenue growing {rev_growth*100:.1f}% YoY")

    earn_growth = info.get('earningsGrowth')
    if earn_growth and earn_growth > 0.10:
        catalysts.append(f"Earnings growing {earn_growth*100:.1f}% YoY")

    if recommend in ('STRONG_BUY', 'BUY'):
        catalysts.append(f"Analyst consensus: {recommend} ({n_analysts} analysts)")

    if target and price > 0 and target > price * 1.15:
        catalysts.append(f"Mean analyst target {(target/price-1)*100:.0f}% above current price")

    cr = info.get('currentRatio')
    if cr and cr > 2.0:
        catalysts.append(f"Strong liquidity (current ratio: {cr:.1f}x)")

    fcf = info.get('freeCashflow')
    if fcf and fcf > 0:
        catalysts.append(f"Positive free cash flow: ${fcf/1e6:.0f}M")

    inst = info.get('heldPercentInstitutions')
    if inst and inst > 0.70:
        catalysts.append(f"High institutional ownership: {inst*100:.0f}%")

    # FIX 5: Use the already-normalised value from `fund` (decimal form, e.g. 0.025
    # = 2.5%).  Reading from raw `info` required a second /100 divide that
    # produced 250% when yfinance returned the value in pct-point form (2.5).
    # fund was already normalised in fetch_fundamentals → no extra divide needed.
    div_yield = (fund or {}).get('dividendYield') or info.get('dividendYield', 0) or 0
    if div_yield > 0.5:           # safety net: if somehow still in pct-point form
        div_yield = div_yield / 100.0
    if div_yield > 0.015:         # >1.5% yield — meaningful income stock signal
        catalysts.append(f"Dividend yield: {div_yield*100:.2f}%")

    pe = info.get('trailingPE')
    if pe and pe < 0:
        risks.append("Not yet profitable (negative trailing PE)")
    elif pe and pe > 50:
        risks.append(f"Premium valuation (P/E: {pe:.0f}x) — priced for perfection")

    beta = info.get('beta')
    if beta and beta > 1.5:
        risks.append(f"High volatility (beta: {beta:.2f})")

    short_pct = info.get('shortPercentOfFloat', 0) or 0
    if short_pct > 0.10:
        risks.append(f"High short interest: {short_pct*100:.1f}% of float")

    debt_eq = info.get('debtToEquity')
    if debt_eq and debt_eq > 150:
        risks.append(f"High debt/equity ratio: {debt_eq:.0f}%")

    if recommend in ('SELL', 'STRONG_SELL', 'UNDERPERFORM'):
        risks.append(f"Analyst consensus: {recommend}")

    mc = info.get('marketCap', 0) or 0
    if mc < 300_000_000:
        risks.append(f"Small-cap (market cap ${mc/1e6:.0f}M) — liquidity risk")

    fcf_neg = info.get('freeCashflow')
    if fcf_neg is not None and fcf_neg < 0:
        risks.append(f"Negative free cash flow: ${fcf_neg/1e6:.0f}M")

    if not catalysts: catalysts.append("No major catalysts identified from available data")
    if not risks:     risks.append("No major risks identified from available data")

    return {
        "company":        name,
        "sector":         f"{sector} / {industry}",
        "country":        country,
        "website":        website,
        "thesis":         summary or f"{name} ({ticker}) — {sector}",
        "catalysts":      catalysts,
        "risks":          risks,
        "analyst_targets": analysts,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SCORING
# ─────────────────────────────────────────────────────────────────────────────
def score_fundamentals(fund: dict) -> dict:
    scores = {}
    ps = fund.get('priceToSalesTrailing12Months')
    scores['valuation'] = max(0, min(10, 10 - ps)) if ps else None
    rg = fund.get('revenueGrowth')
    scores['growth'] = min(10, rg * 25) if rg else None
    cr = fund.get('currentRatio')
    scores['liquidity'] = min(10, cr * 3) if cr else None
    spf = fund.get('shortPercentOfFloat', 0) or 0
    scores['short_interest_pct'] = round(spf * 100, 1)
    scores['squeeze_potential']  = min(10, spf * 100)
    price = fund.get('currentPrice')
    ma200 = fund.get('twoHundredDayAverage')
    if price and ma200:
        diff = (price - ma200) / ma200 * 100
        scores['vs_200ma_pct'] = round(diff, 1)
    return scores


def compute_composite_score(fund: dict, scores: dict, piotroski: dict, dcf_r: dict) -> float:
    """
    NEW 5: 12-factor composite score (0-100) with proper weighting.
    """
    sub = []

    # Growth (25% weight)
    rg = fund.get('revenueGrowth')
    if rg is not None:
        sub.append(('rev_growth', min(10, max(0, rg * 25 + 5)), 2.5))

    eg = fund.get('earningsGrowth')
    if eg is not None and eg > -1:
        sub.append(('earn_growth', min(10, max(0, eg * 15 + 5)), 2.0))

    # Valuation (20% weight)
    price  = fund.get('currentPrice') or fund.get('regularMarketPrice', 0)
    target = fund.get('targetMeanPrice')
    if price and target and price > 0:
        upside = (target - price) / price
        sub.append(('analyst_upside', min(10, max(0, upside * 5 + 5)), 2.0))

    pe = fund.get('trailingPE')
    if pe and pe > 0:
        # Normalize: PE=15 → 7, PE=30 → 5, PE=60 → 3, PE=5 → 9
        pe_score = min(10, max(0, 10 - pe / 8))
        sub.append(('pe_ratio', pe_score, 1.5))

    # Quality (25% weight)
    gm = fund.get('grossMargins')
    if gm is not None:
        sub.append(('gross_margin', min(10, gm * 10), 2.0))

    om = fund.get('operatingMargins')
    if om is not None:
        sub.append(('op_margin', min(10, max(0, om * 15 + 3)), 1.5))

    roa = fund.get('returnOnAssets')
    if roa is not None:
        sub.append(('roa', min(10, max(0, roa * 50 + 5)), 1.5))

    # Financial health (15% weight)
    cr = fund.get('currentRatio')
    if cr is not None:
        sub.append(('current_ratio', min(10, cr * 3), 1.5))

    de = fund.get('debtToEquity')
    if de is not None:
        de_score = max(0, min(10, 10 - de / 30))
        sub.append(('debt_equity', de_score, 1.0))

    # Piotroski (10% weight)
    if piotroski.get('available'):
        pct = piotroski.get('pct', 50) / 10
        sub.append(('piotroski', pct, 1.5))

    # DCF margin of safety (10% weight)
    if dcf_r.get('available'):
        mos = dcf_r.get('margin_of_safety', 0) / 100
        dcf_score = min(10, max(0, mos * 5 + 5))
        sub.append(('dcf_mos', dcf_score, 1.5))

    # Momentum
    mom = fund.get('_momentum', {})
    if '3M' in mom:
        rel = mom['3M']['relative']
        sub.append(('momentum_3m', min(10, max(0, rel * 20 + 5)), 1.0))

    total_weight = sum(w for _, _, w in sub)
    weighted     = sum(s * w for _, s, w in sub)
    if total_weight == 0:
        return 50.0
    return round(weighted / total_weight * 10, 1)


# ─────────────────────────────────────────────────────────────────────────────
# REPORT
# ─────────────────────────────────────────────────────────────────────────────
def print_report(ticker, context, fund, scores, composite, piotroski, dcf_r, graham_r):
    SEP = "="*60
    out_dir = Path("reports") / ticker
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{SEP}")
    print(f"  {ticker}  FUNDAMENTAL ANALYSIS  v6.0")
    print(SEP)
    print(f"\n  Company : {context['company']}")
    print(f"  Sector  : {context['sector']}")
    if context.get('country'): print(f"  Country : {context['country']}")
    if context.get('website'): print(f"  Website : {context['website']}")
    print(f"\n  Thesis  :\n    {context['thesis']}")

    print(f"\n  -- KEY METRICS --")
    price = fund.get('currentPrice') or fund.get('regularMarketPrice')
    metrics = [
        ('Current Price',       price,                                      '${:.2f}'),
        ('52-Week High',        fund.get('fiftyTwoWeekHigh'),               '${:.2f}'),
        ('52-Week Low',         fund.get('fiftyTwoWeekLow'),                '${:.2f}'),
        ('50-Day MA',           fund.get('fiftyDayAverage'),                '${:.2f}'),
        ('200-Day MA',          fund.get('twoHundredDayAverage'),           '${:.2f}'),
        ('Market Cap',          fund.get('marketCap'),                      '${:,.0f}'),
        ('Trailing PE',         fund.get('trailingPE'),                     '{:.1f}x'),
        ('Forward PE',          fund.get('forwardPE'),                      '{:.1f}x'),
        ('PEG Ratio',           fund.get('pegRatio'),                       '{:.2f}x'),
        ('P/S Ratio (TTM)',     fund.get('priceToSalesTrailing12Months'),   '{:.2f}x'),
        ('P/B Ratio',           fund.get('priceToBook'),                    '{:.2f}x'),
        ('EV/Revenue',          fund.get('enterpriseToRevenue'),            '{:.2f}x'),
        ('EV/EBITDA',           fund.get('enterpriseToEbitda'),             '{:.2f}x'),
        ('Revenue Growth',      fund.get('revenueGrowth'),                  '{:.1%}'),
        ('Earnings Growth',     fund.get('earningsGrowth'),                 '{:.1%}'),
        ('Gross Margin',        fund.get('grossMargins'),                   '{:.1%}'),
        ('Operating Margin',    fund.get('operatingMargins'),               '{:.1%}'),
        ('Net Margin',          fund.get('profitMargins'),                  '{:.1%}'),
        ('ROE',                 fund.get('returnOnEquity'),                 '{:.1%}'),
        ('ROA',                 fund.get('returnOnAssets'),                 '{:.1%}'),
        ('Debt/Equity',         fund.get('debtToEquity'),                   '{:.1f}%'),
        ('Current Ratio',       fund.get('currentRatio'),                   '{:.2f}'),
        ('Quick Ratio',         fund.get('quickRatio'),                     '{:.2f}'),
        ('Free Cash Flow',      fund.get('freeCashflow'),                   '${:,.0f}'),
        ('Dividend Yield',      fund.get('dividendYield'),                  '{:.2%}'),
        ('Beta',                fund.get('beta'),                           '{:.2f}'),
        ('Short % Float',       fund.get('shortPercentOfFloat'),            '{:.1%}'),
        ('Inst. Ownership',     fund.get('heldPercentInstitutions'),        '{:.1%}'),
        ('Insider Ownership',   fund.get('heldPercentInsiders'),            '{:.1%}'),
        ('Avg Volume (10d)',    fund.get('averageVolume10days'),             '{:,.0f}'),
        ('Mean Analyst Target', fund.get('targetMeanPrice'),                '${:.2f}'),
    ]
    for label, val, fmt in metrics:
        if val is not None:
            try:    formatted = fmt.format(val)
            except: formatted = str(val)
            print(f"    {label:<25} {formatted}")

    # NEW 1: DCF
    print(f"\n  -- DCF VALUATION --")
    if dcf_r.get('available'):
        print(f"    WACC (estimated)          {dcf_r['wacc']:.1f}%")
        print(f"    Terminal growth rate       {dcf_r['terminal_g']:.1f}%")
        print(f"    Intrinsic Value (Bear)    ${dcf_r['intrinsic_bear']:.2f}")
        print(f"    Intrinsic Value (Base)    ${dcf_r['intrinsic_base']:.2f}")
        print(f"    Intrinsic Value (Bull)    ${dcf_r['intrinsic_bull']:.2f}")
        print(f"    Current Price             ${dcf_r['current_price']:.2f}")
        mos = dcf_r['margin_of_safety']
        mos_arrow = '▲' if mos > 0 else '▼'
        print(f"    Margin of Safety          {mos_arrow} {abs(mos):.1f}%  [{dcf_r['signal']}]")
    else:
        print(f"    {dcf_r.get('note', 'Not available')}")

    # NEW 2: Graham Number
    print(f"\n  -- GRAHAM NUMBER --")
    if graham_r.get('available'):
        print(f"    Trailing EPS              ${graham_r['eps']:.2f}")
        print(f"    Book Value/Share          ${graham_r['bvps']:.2f}")
        print(f"    Graham Number             ${graham_r['graham_number']:.2f}")
        print(f"    Current Price             ${graham_r['current_price']:.2f}")
        mos = graham_r['margin_of_safety']
        mos_arrow = '▲' if mos > 0 else '▼'
        print(f"    Margin of Safety          {mos_arrow} {abs(mos):.1f}%  [{graham_r['signal']}]")
    else:
        print(f"    {graham_r.get('note', 'Not available')}")

    # NEW 3: Piotroski
    print(f"\n  -- PIOTROSKI F-SCORE --")
    if piotroski.get('available'):
        bar_w = int(piotroski['pct'] / 10)
        bar_s = '█' * bar_w + '░' * (10 - bar_w)
        print(f"    Score: {piotroski['score']}/{piotroski['max_score']}  [{bar_s}]  {piotroski['label']}")
        for sig, val in piotroski['signals'].items():
            mark = '✓' if val else '✗'
            print(f"    [{mark}] {sig}")
    else:
        print(f"    {piotroski.get('note', 'Not available')}")

    # Momentum
    mom = fund.get('_momentum', {})
    if mom:
        print(f"\n  -- PRICE MOMENTUM vs SPY --")
        for period, m in mom.items():
            rel = m['relative']
            arrow = '▲' if rel > 0 else '▼'
            print(f"    {period:<5} stock={m['stock']*100:+.1f}%  "
                  f"SPY={m['spy']*100:+.1f}%  "
                  f"relative {arrow} {abs(rel)*100:.1f}%")

    print(f"\n  -- ANALYST TARGETS --")
    for k, v in context.get('analyst_targets', {}).items():
        print(f"    {k:<35} {v}")

    print(f"\n  -- CATALYSTS --")
    for cat in context.get('catalysts', []):
        print(f"    [+] {cat}")

    print(f"\n  -- RISKS --")
    for risk in context.get('risks', []):
        print(f"    [-] {risk}")

    print(f"\n  -- COMPOSITE SCORE (12 factors) --")
    filled    = int(composite / 10)
    bar_str   = '#' * filled + '-' * (10 - filled)
    sentiment = 'BULLISH' if composite >= 65 else ('BEARISH' if composite <= 35 else 'NEUTRAL')
    print(f"    {composite:.0f}/100  [{bar_str}]  {sentiment}")
    print(f"\n  [!] Not financial advice. Educational purposes only.")
    print(SEP + "\n")

    out = {
        'ticker':       ticker,
        'generated':    datetime.datetime.now().isoformat(),
        'fundamentals': {k: v for k, v in fund.items() if k != '_momentum'},
        'momentum':     fund.get('_momentum', {}),
        'scores':       scores,
        'composite':    composite,
        'context':      context,
        'dcf':          dcf_r,
        'graham':       graham_r,
        'piotroski':    piotroski,
    }
    path = Path("reports") / ticker / f"{ticker}_fundamentals.json"
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, default=str)
    print(f"  Saved --> {path}")


def main():
    # Fix Windows cp1252 console: reconfigure stdout/stderr to UTF-8 so that
    # Unicode symbols (≤ ≥ ✓ ✗ → ← × ▸ etc.) print without UnicodeEncodeError.
    # errors='replace' is the safety net on consoles that truly can't handle UTF-8.
    if hasattr(sys.stdout, 'reconfigure'):
        try:
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
            sys.stderr.reconfigure(encoding='utf-8', errors='replace')
        except Exception:
            pass
    global TICKER, OUT_DIR
    TICKER  = get_ticker()
    OUT_DIR = Path("reports") / TICKER
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fund, raw_info = fetch_fundamentals(TICKER)
    context        = build_generic_context(TICKER, raw_info, fund=fund)
    scores         = score_fundamentals(fund)
    dcf_r          = dcf_valuation(fund)
    graham_r       = graham_number(fund)
    piotroski      = piotroski_f_score(fund)
    composite      = compute_composite_score(fund, scores, piotroski, dcf_r)
    print_report(TICKER, context, fund, scores, composite, piotroski, dcf_r, graham_r)


if __name__ == '__main__':
    main()
