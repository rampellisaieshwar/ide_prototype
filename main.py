"""
Intelligent Discrepancy Engine — CLI Utility
-------------------------------------------
Run this as a command-line tool:
    python main.py
"""

import pandas as pd
from modules.matcher import find_best_match
from modules.classifier import ClaimClassifier


def load_data():
    try:
        invoices = pd.read_csv("data/company_invoices.csv")
        payments = pd.read_csv("data/retailer_payments.csv")
        return invoices.to_dict("records"), payments.to_dict("records")
    except FileNotFoundError:
        print("Error: data/ directory not found or files missing.")
        return [], []


def reconcile(invoices, payments, classifier):
    results = []
    for invoice in invoices:
        best = find_best_match(invoice, payments)
        if best is None:
            results.append({
                "invoice_id":   invoice["invoice_id"],
                "status":       "UNMATCHED",
                "invoiced":     invoice["amount"],
                "paid":         0,
                "discrepancy":  invoice["amount"],
                "claim":        None,
                "ai_category":  "NO_MATCH",
                "confidence":   0,
                "needs_review": True,
            })
            continue

        payment = next(p for p in payments if p["ref_no"] == best["ref_no"])
        invoiced = float(invoice["amount"])
        paid     = float(payment["paid_amount"])
        delta    = round(invoiced - paid, 2)
        claim_text = str(payment.get("claim_description", "") or "").strip()
        ai_result = classifier.classify(claim_text)

        results.append({
            "invoice_id":     invoice["invoice_id"],
            "ref_no":         payment["ref_no"],
            "match_conf":     best["confidence"],
            "status":         "DISCREPANCY" if delta > 0 else "CLEARED",
            "invoiced":       invoiced,
            "paid":           paid,
            "discrepancy":    delta,
            "claim_text":     claim_text if claim_text else "—",
            "ai_category":    ai_result["category"],
            "ai_confidence":  ai_result["confidence"],
            "needs_review":   ai_result["review"],
        })
    return results


def print_report(results):
    print("\n" + "="*80)
    print("  INTELLIGENT DISCREPANCY ENGINE — CLI REPORT")
    print("="*80)
    
    for r in results:
        status_icon = "❌" if r["status"] == "DISCREPANCY" else "✅" if r["status"] == "CLEARED" else "❓"
        print(f"{status_icon} {r['invoice_id']} | Status: {r['status']} | Short: ₹{r.get('discrepancy', 0):,.2f}")
    
    print("="*80 + "\n")


if __name__ == "__main__":
    # Running as CLI
    print("Initializing Engine...")
    invoices, payments = load_data()
    if not invoices:
        exit(1)
        
    classifier = ClaimClassifier()
    results = reconcile(invoices, payments, classifier)
    print_report(results)
