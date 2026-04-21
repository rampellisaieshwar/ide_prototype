"""
Intelligent Discrepancy Engine — Prototype
-------------------------------------------
Orchestrates the reconciliation pipeline:

    1. Load company invoices and retailer payments
    2. Match records using weighted fuzzy scoring
    3. Detect discrepancies (delta between invoiced and paid)
    4. Classify claims using AI/NLP (TF-IDF + Logistic Regression)
    5. Output reconciliation report

Prototype scope:
    This prototype demonstrates the core matching and AI classification logic.
    Document OCR, warehouse DCC, and the insurance trigger are designed in the
    full system architecture document and are not implemented here.
"""

import pandas as pd
import streamlit as st
from modules.matcher import find_best_match
from modules.classifier import ClaimClassifier


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def load_data():
    invoices = pd.read_csv("data/company_invoices.csv")
    payments = pd.read_csv("data/retailer_payments.csv")
    return invoices.to_dict("records"), payments.to_dict("records")


# ---------------------------------------------------------------------------
# Reconciliation engine
# ---------------------------------------------------------------------------

def reconcile(invoices, payments, classifier):
    results = []

    for invoice in invoices:
        best = find_best_match(invoice, payments)

        if best is None:
            results.append({
                "invoice_id":   invoice["invoice_id"],
                "status":       "UNMATCHED",
                "invoiced":     invoice["amount"],
                "paid":         None,
                "discrepancy":  None,
                "claim":        None,
                "ai_category":  None,
                "confidence":   None,
                "needs_review": None,
            })
            continue

        # Find the matched payment record
        payment = next(p for p in payments if p["ref_no"] == best["ref_no"])

        invoiced = float(invoice["amount"])
        paid     = float(payment["paid_amount"])
        delta    = round(invoiced - paid, 2)

        # Classify the claim using the AI module
        claim_text   = str(payment.get("claim_description", "") or "").strip()
        ai_result    = classifier.classify(claim_text)

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


# ---------------------------------------------------------------------------
# Report printer
# ---------------------------------------------------------------------------

def print_report(results):
    sep  = "─" * 90
    sep2 = "═" * 90

    print()
    print(sep2)
    print("  INTELLIGENT DISCREPANCY ENGINE — RECONCILIATION REPORT")
    print(sep2)

    cleared      = [r for r in results if r["status"] == "CLEARED"]
    discrepancies = [r for r in results if r["status"] == "DISCREPANCY"]
    unmatched    = [r for r in results if r["status"] == "UNMATCHED"]
    review_queue = [r for r in results if r.get("needs_review")]

    print(f"\n  Total invoices:     {len(results)}")
    print(f"  Cleared:            {len(cleared)}")
    print(f"  Discrepancies:      {len(discrepancies)}")
    print(f"  Unmatched:          {len(unmatched)}")
    print(f"  Flagged for review: {len(review_queue)}")

    # --- Discrepancy details
    if discrepancies:
        print(f"\n{sep}")
        print("  DISCREPANCIES DETECTED")
        print(sep)
        for r in discrepancies:
            print(f"\n  Invoice:     {r['invoice_id']}  →  Matched to: {r['ref_no']}  (match confidence: {r['match_conf']})")
            print(f"  Invoiced:    ₹{r['invoiced']:,.0f}    Paid: ₹{r['paid']:,.0f}    Short by: ₹{r['discrepancy']:,.0f}")
            print(f"  Claim text:  \"{r['claim_text']}\"")
            print(f"  AI Category: {r['ai_category']}  (confidence: {r['ai_confidence']})", end="")
            if r["needs_review"]:
                print("  ⚠  LOW CONFIDENCE → HUMAN REVIEW REQUIRED", end="")
            print()

    # --- Cleared
    if cleared:
        print(f"\n{sep}")
        print("  CLEARED (no discrepancy)")
        print(sep)
        for r in cleared:
            print(f"  {r['invoice_id']}  →  {r['ref_no']}  |  ₹{r['invoiced']:,.0f}  |  match conf: {r['match_conf']}")

    # --- Unmatched
    if unmatched:
        print(f"\n{sep}")
        print("  UNMATCHED INVOICES")
        print(sep)
        for r in unmatched:
            print(f"  {r['invoice_id']}  |  ₹{r['invoiced']:,.0f}  — no payment record found")

    print(f"\n{sep2}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    running_under_streamlit = st.runtime.exists()

    if running_under_streamlit:
        import app  # noqa: F401
    else:
        print("\nLoading data...")
        invoices, payments = load_data()

        print("Training claim classifier (TF-IDF + Logistic Regression)...")
        classifier = ClaimClassifier()
        print("Classifier ready.\n")

        print("Running reconciliation pipeline...")
        results = reconcile(invoices, payments, classifier)

        print_report(results)
