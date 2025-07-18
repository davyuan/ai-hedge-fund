from __future__ import annotations

from datetime import datetime, timedelta
import json
from typing_extensions import Literal

from src.graph.state import AgentState, show_agent_reasoning
from src.tools.api import (
    get_company_news,
    get_financial_metrics,
    get_insider_trades,
    get_market_cap,
    search_line_items,
)
from src.utils.progress import progress
from src.graph.state import AgentState, show_agent_reasoning
from src.tools.api import get_financial_metrics, get_market_cap, search_line_items
from pydantic import BaseModel
import json
from typing_extensions import Literal
from typing import Annotated
import datetime as dt
from src.utils.progress import progress
from semantic_kernel.functions import kernel_function

class AnalysisDataPlugin4MichaelBurry:

    @kernel_function(description="Provides essential data for a specified stock ticker on a specified end date. "
                        "This function specifically requires an 'end date' to get data as of that point in time. "
                        "The 'end_date' MUST be provided in 'YYYY-MM-DD' format, for example, '2025-07-06'."
                        "The data returned includes disruptive_analysis, innovation_analysis and valuation_analysis.")
    def get_analysis_data(self, ticker:Annotated[str, "The stock ticker symbol (e.g., 'TSLA', 'GOOG', 'AAPL') for which to retrieve analysis data."],
            end_date: Annotated[str, "REQUIRED: The end date for data retrieval. This MUST be in 'YYYY-MM-DD' format (e.g., '2025-07-06'). Data will be retrieved as of this exact date."]        ) -> Annotated[str, "Returns analysis data Cathie Wood is interested in, based on the ticker and end date."]:
        #print(f"AnalysisDataPlugin4MichaelBurry called with ticker:{ticker}, end_date:{end_date}")
        analysis_data = {}

        # We look one year back for insider trades / news flow
        start_date = (datetime.fromisoformat(end_date) - timedelta(days=365)).date().isoformat()

        analysis_data: dict[str, dict] = {}
        burry_analysis: dict[str, dict] = {}

        # ------------------------------------------------------------------
        # Fetch raw data
        # ------------------------------------------------------------------
        progress.update_status("michael_burry_agent", ticker, "Fetching financial metrics")
        metrics = get_financial_metrics(ticker, end_date, period="ttm", limit=5)

        progress.update_status("michael_burry_agent", ticker, "Fetching line items")
        line_items = search_line_items(
            ticker,
            [
                "free_cash_flow",
                "net_income",
                "total_debt",
                "cash_and_equivalents",
                "total_assets",
                "total_liabilities",
                "outstanding_shares",
                "issuance_or_purchase_of_equity_shares",
            ],
            end_date,
        )

        progress.update_status("michael_burry_agent", ticker, "Fetching insider trades")
        insider_trades = get_insider_trades(ticker, end_date=end_date, start_date=start_date)

        progress.update_status("michael_burry_agent", ticker, "Fetching company news")
        news = get_company_news(ticker, end_date=end_date, start_date=start_date, limit=250)

        progress.update_status("michael_burry_agent", ticker, "Fetching market cap")
        market_cap = get_market_cap(ticker, end_date)

        # ------------------------------------------------------------------
        # Run sub‑analyses
        # ------------------------------------------------------------------
        progress.update_status("michael_burry_agent", ticker, "Analyzing value")
        value_analysis = _analyze_value(metrics, line_items, market_cap)

        progress.update_status("michael_burry_agent", ticker, "Analyzing balance sheet")
        balance_sheet_analysis = _analyze_balance_sheet(metrics, line_items)

        progress.update_status("michael_burry_agent", ticker, "Analyzing insider activity")
        insider_analysis = _analyze_insider_activity(insider_trades)

        progress.update_status("michael_burry_agent", ticker, "Analyzing contrarian sentiment")
        contrarian_analysis = _analyze_contrarian_sentiment(news)

        # ------------------------------------------------------------------
        # Aggregate score & derive preliminary signal
        # ------------------------------------------------------------------
        total_score = (
            value_analysis["score"]
            + balance_sheet_analysis["score"]
            + insider_analysis["score"]
            + contrarian_analysis["score"]
        )
        max_score = (
            value_analysis["max_score"]
            + balance_sheet_analysis["max_score"]
            + insider_analysis["max_score"]
            + contrarian_analysis["max_score"]
        )

        if total_score >= 0.7 * max_score:
            signal = "bullish"
        elif total_score <= 0.3 * max_score:
            signal = "bearish"
        else:
            signal = "neutral"

        # ------------------------------------------------------------------
        # Collect data for LLM reasoning & output
        # ------------------------------------------------------------------
        analysis_data[ticker] = {
            "signal": signal,
            "score": total_score,
            "max_score": max_score,
            "value_analysis": value_analysis,
            "balance_sheet_analysis": balance_sheet_analysis,
            "insider_analysis": insider_analysis,
            "contrarian_analysis": contrarian_analysis,
            "market_cap": market_cap,
        }

        progress.update_status("michael_burry_agent", ticker, "Done")

        return analysis_data

###############################################################################
# Sub‑analysis helpers
###############################################################################


def _latest_line_item(line_items: list):
    """Return the most recent line‑item object or *None*."""
    return line_items[0] if line_items else None


# ----- Value ----------------------------------------------------------------

def _analyze_value(metrics, line_items, market_cap):
    """Free cash‑flow yield, EV/EBIT, other classic deep‑value metrics."""

    max_score = 6  # 4 pts for FCF‑yield, 2 pts for EV/EBIT
    score = 0
    details: list[str] = []

    # Free‑cash‑flow yield
    latest_item = _latest_line_item(line_items)
    fcf = getattr(latest_item, "free_cash_flow", None) if latest_item else None
    if fcf is not None and market_cap:
        fcf_yield = fcf / market_cap
        if fcf_yield >= 0.15:
            score += 4
            details.append(f"Extraordinary FCF yield {fcf_yield:.1%}")
        elif fcf_yield >= 0.12:
            score += 3
            details.append(f"Very high FCF yield {fcf_yield:.1%}")
        elif fcf_yield >= 0.08:
            score += 2
            details.append(f"Respectable FCF yield {fcf_yield:.1%}")
        else:
            details.append(f"Low FCF yield {fcf_yield:.1%}")
    else:
        details.append("FCF data unavailable")

    # EV/EBIT (from financial metrics)
    if metrics:
        ev_ebit = getattr(metrics[0], "ev_to_ebit", None)
        if ev_ebit is not None:
            if ev_ebit < 6:
                score += 2
                details.append(f"EV/EBIT {ev_ebit:.1f} (<6)")
            elif ev_ebit < 10:
                score += 1
                details.append(f"EV/EBIT {ev_ebit:.1f} (<10)")
            else:
                details.append(f"High EV/EBIT {ev_ebit:.1f}")
        else:
            details.append("EV/EBIT data unavailable")
    else:
        details.append("Financial metrics unavailable")

    return {"score": score, "max_score": max_score, "details": "; ".join(details)}


# ----- Balance sheet --------------------------------------------------------

def _analyze_balance_sheet(metrics, line_items):
    """Leverage and liquidity checks."""

    max_score = 3
    score = 0
    details: list[str] = []

    latest_metrics = metrics[0] if metrics else None
    latest_item = _latest_line_item(line_items)

    debt_to_equity = getattr(latest_metrics, "debt_to_equity", None) if latest_metrics else None
    if debt_to_equity is not None:
        if debt_to_equity < 0.5:
            score += 2
            details.append(f"Low D/E {debt_to_equity:.2f}")
        elif debt_to_equity < 1:
            score += 1
            details.append(f"Moderate D/E {debt_to_equity:.2f}")
        else:
            details.append(f"High leverage D/E {debt_to_equity:.2f}")
    else:
        details.append("Debt‑to‑equity data unavailable")

    # Quick liquidity sanity check (cash vs total debt)
    if latest_item is not None:
        cash = getattr(latest_item, "cash_and_equivalents", None)
        total_debt = getattr(latest_item, "total_debt", None)
        if cash is not None and total_debt is not None:
            if cash > total_debt:
                score += 1
                details.append("Net cash position")
            else:
                details.append("Net debt position")
        else:
            details.append("Cash/debt data unavailable")

    return {"score": score, "max_score": max_score, "details": "; ".join(details)}


# ----- Insider activity -----------------------------------------------------

def _analyze_insider_activity(insider_trades):
    """Net insider buying over the last 12 months acts as a hard catalyst."""

    max_score = 2
    score = 0
    details: list[str] = []

    if not insider_trades:
        details.append("No insider trade data")
        return {"score": score, "max_score": max_score, "details": "; ".join(details)}

    shares_bought = sum(t.transaction_shares or 0 for t in insider_trades if (t.transaction_shares or 0) > 0)
    shares_sold = abs(sum(t.transaction_shares or 0 for t in insider_trades if (t.transaction_shares or 0) < 0))
    net = shares_bought - shares_sold
    if net > 0:
        score += 2 if net / max(shares_sold, 1) > 1 else 1
        details.append(f"Net insider buying of {net:,} shares")
    else:
        details.append("Net insider selling")

    return {"score": score, "max_score": max_score, "details": "; ".join(details)}


# ----- Contrarian sentiment -------------------------------------------------

def _analyze_contrarian_sentiment(news):
    """Very rough gauge: a wall of recent negative headlines can be a *positive* for a contrarian."""

    max_score = 1
    score = 0
    details: list[str] = []

    if not news:
        details.append("No recent news")
        return {"score": score, "max_score": max_score, "details": "; ".join(details)}

    # Count negative sentiment articles
    sentiment_negative_count = sum(
        1 for n in news if n.sentiment and n.sentiment.lower() in ["negative", "bearish"]
    )
    
    if sentiment_negative_count >= 5:
        score += 1  # The more hated, the better (assuming fundamentals hold up)
        details.append(f"{sentiment_negative_count} negative headlines (contrarian opportunity)")
    else:
        details.append("Limited negative press")

    return {"score": score, "max_score": max_score, "details": "; ".join(details)}
