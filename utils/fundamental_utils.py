import math

import numpy as np
import pandas as pd


def safe_float(value, default=np.nan) -> float:
    try:
        if value is None:
            return float(default)
        if isinstance(value, str) and not value.strip():
            return float(default)
        out = float(value)
        if math.isnan(out) or math.isinf(out):
            return float(default)
        return out
    except Exception:
        return float(default)


def _normalize_label(label: str) -> str:
    return "".join(ch for ch in str(label).lower() if ch.isalnum())


def load_statement_frame(source, attr_names) -> "pd.DataFrame | None":
    for attr in attr_names:
        try:
            frame = getattr(source, attr, None)
            if isinstance(frame, pd.DataFrame) and not frame.empty:
                return frame
        except Exception:
            continue
    return None


def latest_statement_value(frame: "pd.DataFrame | None", labels) -> float:
    if frame is None or frame.empty:
        return np.nan
    normalized = {_normalize_label(lbl): lbl for lbl in frame.index}
    for label in labels:
        key = _normalize_label(label)
        if key not in normalized:
            continue
        row = pd.to_numeric(frame.loc[normalized[key]], errors="coerce").dropna()
        if not row.empty:
            return safe_float(row.iloc[0])
    return np.nan


def statement_series(frame: "pd.DataFrame | None", labels) -> "pd.Series | None":
    if frame is None or frame.empty:
        return None
    normalized = {_normalize_label(lbl): lbl for lbl in frame.index}
    for label in labels:
        key = _normalize_label(label)
        if key not in normalized:
            continue
        row = pd.to_numeric(frame.loc[normalized[key]], errors="coerce").dropna()
        if row.empty:
            return None
        try:
            row.index = pd.to_datetime(row.index)
        except Exception:
            pass
        return row.sort_index()
    return None


def estimate_wacc(info: dict,
                  tax_rate: float = 0.21,
                  risk_free_rate: float = 0.045,
                  equity_risk_premium: float = 0.055,
                  debt_spread: float = 0.020,
                  country_risk_premium: float = 0.0) -> dict:
    beta = safe_float(info.get("beta"), 1.0)
    market_cap = max(safe_float(info.get("marketCap"), np.nan), 0.0)
    total_debt = max(safe_float(info.get("totalDebt"), 0.0), 0.0)
    interest_expense = safe_float(info.get("interestExpense"), np.nan)

    cost_of_equity = risk_free_rate + beta * (equity_risk_premium + country_risk_premium)
    if total_debt > 0 and not math.isnan(interest_expense) and interest_expense > 0:
        pre_tax_cost_of_debt = interest_expense / total_debt
    else:
        pre_tax_cost_of_debt = risk_free_rate + debt_spread + country_risk_premium
    after_tax_cost_of_debt = max(0.0, pre_tax_cost_of_debt * (1.0 - tax_rate))

    if market_cap <= 0:
        equity_weight = 1.0
        debt_weight = 0.0
    else:
        capital = max(market_cap + total_debt, 1.0)
        equity_weight = market_cap / capital
        debt_weight = total_debt / capital
    wacc = equity_weight * cost_of_equity + debt_weight * after_tax_cost_of_debt
    wacc = float(max(wacc, risk_free_rate + 0.01))
    return {
        "wacc": wacc,
        "cost_of_equity": float(cost_of_equity),
        "pre_tax_cost_of_debt": float(pre_tax_cost_of_debt),
        "after_tax_cost_of_debt": float(after_tax_cost_of_debt),
        "equity_weight": float(equity_weight),
        "debt_weight": float(debt_weight),
        "tax_rate": float(tax_rate),
    }


def build_dcf_state(info: dict,
                    years: int = 5,
                    terminal_growth: float = 0.025,
                    tax_rate: float = 0.21,
                    risk_free_rate: float = 0.045,
                    equity_risk_premium: float = 0.055,
                    debt_spread: float = 0.020,
                    country_risk_premium: float = 0.0) -> dict:
    price = safe_float(info.get("currentPrice"), safe_float(info.get("regularMarketPrice"), np.nan))
    shares = max(safe_float(info.get("sharesOutstanding"), np.nan), 0.0)
    fcf = safe_float(info.get("freeCashflow"), np.nan)
    revenue = safe_float(info.get("totalRevenue"), np.nan)
    cash = max(safe_float(info.get("totalCash"), 0.0), 0.0)
    debt = max(safe_float(info.get("totalDebt"), 0.0), 0.0)
    base_growth = safe_float(info.get("revenueGrowth"), 0.05)
    wacc_info = estimate_wacc(
        info,
        tax_rate=tax_rate,
        risk_free_rate=risk_free_rate,
        equity_risk_premium=equity_risk_premium,
        debt_spread=debt_spread,
        country_risk_premium=country_risk_premium,
    )
    wacc = wacc_info["wacc"]

    if price <= 0 or shares <= 0:
        return {"available": False, "note": "missing price or share count"}
    if math.isnan(fcf) or fcf <= 0:
        return {"available": False, "note": "FCF <= 0 or unavailable"}

    tg = min(float(terminal_growth), max(0.0, wacc - 0.01))
    return {
        "available": True,
        "price": float(price),
        "shares": float(shares),
        "fcf": float(fcf),
        "revenue": float(revenue) if not math.isnan(revenue) else np.nan,
        "cash": float(cash),
        "debt": float(debt),
        "net_debt": float(debt - cash),
        "years": int(years),
        "wacc": float(wacc),
        "base_growth": float(base_growth),
        "terminal_growth": float(tg),
        "wacc_breakdown": wacc_info,
    }


def dcf_value_per_share(state: dict, growth: float, discount_rate: float | None = None) -> float:
    if not state.get("available"):
        return np.nan
    wacc = float(discount_rate if discount_rate is not None else state["wacc"])
    terminal_growth = min(float(state["terminal_growth"]), wacc - 0.005)
    if wacc <= terminal_growth:
        return np.nan
    cash_flow = float(state["fcf"])
    pv = 0.0
    for year in range(1, int(state["years"]) + 1):
        cash_flow *= (1.0 + growth)
        pv += cash_flow / ((1.0 + wacc) ** year)
    terminal = cash_flow * (1.0 + terminal_growth) / (wacc - terminal_growth)
    terminal /= (1.0 + wacc) ** int(state["years"])
    equity_value = pv + terminal - float(state["net_debt"])
    return float(equity_value / state["shares"])


def dcf_valuation(state: dict,
                  scenario_multipliers: dict | None = None) -> dict:
    if not state.get("available"):
        return {"available": False, "note": state.get("note", "DCF unavailable")}
    base_growth = max(-0.50, float(state.get("base_growth", 0.05)))
    mult = scenario_multipliers or {"Bear": 0.4, "Base": 0.8, "Bull": 1.2}
    scenarios = {}
    for name, factor in mult.items():
        growth = max(-0.20, min(0.60, base_growth * factor))
        scenarios[name] = dcf_value_per_share(state, growth)
    base_value = scenarios["Base"]
    price = float(state["price"])
    margin = (base_value - price) / price if price > 0 else 0.0
    return {
        "available": True,
        "wacc": round(float(state["wacc"]) * 100, 2),
        "terminal_g": round(float(state["terminal_growth"]) * 100, 2),
        "fcf_per_share": round(float(state["fcf"]) / float(state["shares"]), 4),
        "intrinsic_bear": round(float(scenarios["Bear"]), 2),
        "intrinsic_base": round(float(base_value), 2),
        "intrinsic_bull": round(float(scenarios["Bull"]), 2),
        "current_price": round(price, 2),
        "margin_of_safety": round(margin * 100, 1),
        "signal": (
            "UNDERVALUED" if margin > 0.15 else
            "OVERVALUED" if margin < -0.15 else
            "FAIR VALUE"
        ),
    }


def reverse_dcf_analysis(state: dict,
                         min_growth: float = -0.20,
                         max_growth: float = 1.00,
                         tol: float = 1e-5,
                         max_iter: int = 80) -> dict:
    if not state.get("available"):
        return {"available": False, "note": state.get("note", "Reverse DCF unavailable")}

    price = float(state["price"])
    lo_val = dcf_value_per_share(state, min_growth)
    hi_val = dcf_value_per_share(state, max_growth)
    if math.isnan(lo_val) or math.isnan(hi_val):
        return {"available": False, "note": "Unable to evaluate DCF bounds"}

    bounded = None
    if price <= lo_val:
        implied_growth = min_growth
        bounded = "below_min_growth"
    elif price >= hi_val:
        implied_growth = max_growth
        bounded = "above_max_growth"
    else:
        lo, hi = min_growth, max_growth
        for _ in range(max_iter):
            mid = (lo + hi) / 2.0
            mid_val = dcf_value_per_share(state, mid)
            if math.isnan(mid_val):
                break
            if abs(mid_val - price) <= tol:
                lo = hi = mid
                break
            if mid_val > price:
                hi = mid
            else:
                lo = mid
        implied_growth = (lo + hi) / 2.0

    reported_growth = safe_float(state.get("base_growth"), np.nan)
    gap = implied_growth - reported_growth if not math.isnan(reported_growth) else np.nan
    return {
        "available": True,
        "current_price": round(price, 2),
        "assumed_wacc": round(float(state["wacc"]) * 100, 2),
        "terminal_growth": round(float(state["terminal_growth"]) * 100, 2),
        "implied_growth_5y": round(float(implied_growth) * 100, 2),
        "reported_growth": None if math.isnan(reported_growth) else round(float(reported_growth) * 100, 2),
        "growth_gap": None if math.isnan(gap) else round(float(gap) * 100, 2),
        "bounded": bounded,
        "interpretation": (
            "Aggressive expectations" if implied_growth > 0.20 else
            "Reasonable expectations" if implied_growth > 0.05 else
            "Muted expectations"
        ),
    }


def dcf_surface_analysis(state: dict,
                         growth_grid=None,
                         discount_grid=None) -> dict:
    if not state.get("available"):
        return {"available": False, "note": state.get("note", "DCF surface unavailable")}
    growths = list(growth_grid) if growth_grid is not None else np.linspace(-0.10, 0.30, 9).tolist()
    discounts = list(discount_grid) if discount_grid is not None else np.linspace(0.08, 0.16, 5).tolist()
    matrix = []
    flat_vals = []
    for disc in discounts:
        row = []
        for growth in growths:
            value = dcf_value_per_share(state, growth, discount_rate=disc)
            if math.isnan(value) or math.isinf(value):
                row.append(None)
            else:
                row.append(round(float(value), 2))
                flat_vals.append(float(value))
        matrix.append(row)
    if not flat_vals:
        return {"available": False, "note": "DCF surface grid invalid"}
    price = float(state["price"])
    above = [val for val in flat_vals if val >= price]
    return {
        "available": True,
        "growth_grid": [round(float(g) * 100, 2) for g in growths],
        "discount_grid": [round(float(d) * 100, 2) for d in discounts],
        "fair_value_grid": matrix,
        "pct_above_price": round(len(above) / len(flat_vals) * 100, 1),
        "median_fair_value": round(float(np.median(flat_vals)), 2),
        "min_fair_value": round(float(np.min(flat_vals)), 2),
        "max_fair_value": round(float(np.max(flat_vals)), 2),
        "valuation_sensitivity": round(float(np.max(flat_vals) - np.min(flat_vals)), 2),
        "current_price": round(price, 2),
    }


def compute_dilution_metrics(current_shares: float,
                             share_history: "pd.Series | None" = None,
                             stock_based_comp: float | None = None,
                             revenue: float | None = None,
                             free_cashflow: float | None = None) -> dict:
    current_shares = max(safe_float(current_shares, 0.0), 0.0)
    sbc = safe_float(stock_based_comp, np.nan)
    revenue = safe_float(revenue, np.nan)
    fcf = safe_float(free_cashflow, np.nan)
    share_growth_1y = np.nan
    share_cagr_3y = np.nan
    annual_points = 0

    if share_history is not None:
        series = pd.Series(share_history).dropna()
        if not series.empty:
            try:
                series.index = pd.to_datetime(series.index)
                series = series.sort_index().resample("YE").last().dropna()
            except Exception:
                series = series.sort_index()
            annual_points = int(len(series))
            if len(series) >= 2 and series.iloc[-2] > 0:
                share_growth_1y = safe_float(series.iloc[-1] / series.iloc[-2] - 1.0)
            if len(series) >= 4 and series.iloc[-4] > 0:
                share_cagr_3y = safe_float((series.iloc[-1] / series.iloc[-4]) ** (1 / 3) - 1.0)

    sbc_ratio = np.nan
    if not math.isnan(sbc) and not math.isnan(revenue) and revenue > 0:
        sbc_ratio = safe_float(sbc / revenue)
    dilution_adj_fcf_ps = np.nan
    if current_shares > 0 and not math.isnan(fcf):
        dilution_adj_fcf_ps = safe_float(fcf / current_shares)

    risk_flag = (
        (not math.isnan(share_growth_1y) and share_growth_1y > 0.03) or
        (not math.isnan(sbc_ratio) and sbc_ratio > 0.05)
    )
    risk_label = "HIGH" if risk_flag else "NORMAL"
    if (not math.isnan(share_growth_1y) and share_growth_1y < -0.02) and not risk_flag:
        risk_label = "SHARE REDUCTION"

    return {
        "available": annual_points > 0 or not math.isnan(sbc_ratio),
        "share_growth_1y": None if math.isnan(share_growth_1y) else round(float(share_growth_1y) * 100, 2),
        "share_cagr_3y": None if math.isnan(share_cagr_3y) else round(float(share_cagr_3y) * 100, 2),
        "sbc_ratio": None if math.isnan(sbc_ratio) else round(float(sbc_ratio) * 100, 2),
        "dilution_adjusted_fcf_per_share": None if math.isnan(dilution_adj_fcf_ps) else round(float(dilution_adj_fcf_ps), 4),
        "risk_flag": bool(risk_flag),
        "risk_label": risk_label,
        "annual_points": annual_points,
    }


def compute_speculative_growth_profile(info: dict,
                                       dilution: dict | None = None,
                                       dcf_result: dict | None = None,
                                       reverse_dcf: dict | None = None) -> dict:
    dilution = dilution or {}
    dcf_result = dcf_result or {}
    reverse_dcf = reverse_dcf or {}

    revenue = safe_float(info.get("totalRevenue"), np.nan)
    fcf = safe_float(info.get("freeCashflow"), np.nan)
    op_margin = safe_float(info.get("operatingMargins"), np.nan)
    profit_margin = safe_float(info.get("profitMargins"), np.nan)
    ps_ratio = safe_float(info.get("priceToSalesTrailing12Months"), np.nan)
    ev_rev = safe_float(info.get("enterpriseToRevenue"), np.nan)
    dilution_1y = safe_float(dilution.get("share_growth_1y"), np.nan)
    sbc_ratio = safe_float(dilution.get("sbc_ratio"), np.nan)
    implied_growth = safe_float(reverse_dcf.get("implied_growth_5y"), np.nan)

    triggers = []
    score = 0.0
    if math.isnan(fcf) or fcf <= 0:
        triggers.append("negative_fcf_or_unavailable")
        score += 1.5
    if not bool(dcf_result.get("available")):
        triggers.append("dcf_unavailable")
        score += 1.0
    if not math.isnan(dilution_1y) and dilution_1y > 3.0:
        triggers.append("high_dilution")
        score += min(2.0, dilution_1y / 6.0)
    if not math.isnan(sbc_ratio) and sbc_ratio > 5.0:
        triggers.append("high_sbc_burden")
        score += min(2.0, sbc_ratio / 8.0)
    if not math.isnan(ev_rev) and ev_rev > 20.0:
        triggers.append("extreme_ev_revenue")
        score += min(2.0, ev_rev / 30.0)
    elif not math.isnan(ps_ratio) and ps_ratio > 15.0:
        triggers.append("extreme_price_sales")
        score += min(1.5, ps_ratio / 20.0)
    if (not math.isnan(op_margin) and op_margin < 0.0) or (not math.isnan(profit_margin) and profit_margin < 0.0):
        triggers.append("negative_margins")
        score += 1.0
    if not math.isnan(implied_growth) and implied_growth > 20.0:
        triggers.append("reverse_dcf_aggressive_growth")
        score += min(1.5, implied_growth / 30.0)

    risk = score >= 3.0 or len(triggers) >= 3
    haircut = float(np.clip(score / 10.0, 0.0, 0.35))
    confidence_multiplier = float(np.clip(1.0 - haircut, 0.55, 1.0))
    return {
        "available": True,
        "speculative_growth_risk": bool(risk),
        "speculative_growth_score": round(float(score), 3),
        "speculative_growth_haircut": round(float(haircut), 4),
        "fundamental_confidence_multiplier": round(float(confidence_multiplier), 4),
        "triggers": triggers,
        "context": {
            "revenue": None if math.isnan(revenue) else float(revenue),
            "fcf": None if math.isnan(fcf) else float(fcf),
            "ev_revenue": None if math.isnan(ev_rev) else float(ev_rev),
            "price_sales": None if math.isnan(ps_ratio) else float(ps_ratio),
            "dilution_1y_pct": None if math.isnan(dilution_1y) else float(dilution_1y),
            "sbc_ratio_pct": None if math.isnan(sbc_ratio) else float(sbc_ratio),
            "implied_growth_5y_pct": None if math.isnan(implied_growth) else float(implied_growth),
        },
    }
