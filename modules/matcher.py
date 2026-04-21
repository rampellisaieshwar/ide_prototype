"""
Record Matcher
--------------
Matches retailer payment records to company invoices using a
weighted confidence score across three signals.

Signals and weights:
    ID similarity    50%  — fuzzy match on reference numbers
    Amount proximity 35%  — how close the paid amount is to invoiced
    Date proximity   15%  — days between invoice date and payment date

Why fuzzy matching on IDs:
    "INV-1001" on the company side becomes "REF-1001" on the retailer side.
    Exact matching fails. RapidFuzz token_sort_ratio handles abbreviations,
    prefixes, and minor formatting differences.
"""

from rapidfuzz import fuzz
from datetime import datetime


WEIGHT_ID     = 0.50
WEIGHT_AMOUNT = 0.35
WEIGHT_DATE   = 0.15
MATCH_THRESHOLD = 0.55


def _id_score(inv_id: str, ref_id: str) -> float:
    """Fuzzy similarity between two reference strings, 0–1."""
    # Strip common prefixes before comparing
    clean_inv = inv_id.replace("INV-", "").replace("INV", "").strip()
    clean_ref = ref_id.replace("REF-", "").replace("REF", "").strip()
    return fuzz.token_sort_ratio(clean_inv, clean_ref) / 100.0


def _amount_score(invoiced: float, paid: float) -> float:
    """
    Score based on how close paid amount is to invoiced amount.
    Full score if exact. Degrades linearly up to 30% difference.
    """
    if invoiced == 0:
        return 0.0
    diff_pct = abs(invoiced - paid) / invoiced
    return max(0.0, 1.0 - (diff_pct / 0.30))


def _date_score(inv_date: str, pay_date: str) -> float:
    """
    Score based on days between invoice and payment.
    Full score within 7 days. Zero score beyond 60 days.
    """
    try:
        d1 = datetime.strptime(inv_date, "%Y-%m-%d")
        d2 = datetime.strptime(pay_date, "%Y-%m-%d")
        days = abs((d2 - d1).days)
        if days <= 7:
            return 1.0
        elif days >= 60:
            return 0.0
        else:
            return 1.0 - ((days - 7) / 53)
    except Exception:
        return 0.5  # neutral score if dates cannot be parsed


def match(invoice: dict, payment: dict) -> dict:
    """
    Match one invoice against one payment record.

    Returns a match result with confidence score and signal breakdown.
    """
    s_id     = _id_score(str(invoice["invoice_id"]), str(payment["ref_no"]))
    s_amount = _amount_score(float(invoice["amount"]), float(payment["paid_amount"]))
    s_date   = _date_score(str(invoice["date"]), str(payment["payment_date"]))

    confidence = (
        WEIGHT_ID     * s_id     +
        WEIGHT_AMOUNT * s_amount +
        WEIGHT_DATE   * s_date
    )

    return {
        "invoice_id":   invoice["invoice_id"],
        "ref_no":       payment["ref_no"],
        "confidence":   round(confidence, 3),
        "matched":      confidence >= MATCH_THRESHOLD,
        "signal_id":    round(s_id, 3),
        "signal_amount": round(s_amount, 3),
        "signal_date":  round(s_date, 3),
    }


def find_best_match(invoice: dict, payments: list) -> dict | None:
    """Find the highest-confidence match for an invoice across all payments."""
    results = [match(invoice, p) for p in payments]
    results.sort(key=lambda x: x["confidence"], reverse=True)
    best = results[0] if results else None
    return best if (best and best["matched"]) else None
