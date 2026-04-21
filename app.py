"""
Intelligent Discrepancy Engine — Streamlit Web App
----------------------------------------------------
Run locally:  streamlit run app.py
Deployed:     https://your-app.streamlit.app
"""

import streamlit as st
import pandas as pd
import io
from modules.matcher import find_best_match
from modules.classifier import ClaimClassifier

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Intelligent Discrepancy Engine",
    page_icon="🔍",
    layout="wide",
)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🔍 Intelligent Discrepancy Engine")
st.markdown(
    "AI-driven reconciliation system for FMCG invoice disputes. "
    "Upload your company invoices and retailer payment records — "
    "the engine matches them, detects discrepancies, and classifies claims using NLP."
)
st.divider()

# ── Load classifier once (cached so it doesn't retrain on every interaction) ─
@st.cache_resource
def load_classifier():
    return ClaimClassifier()

classifier = load_classifier()

# ── Sample data ───────────────────────────────────────────────────────────────
SAMPLE_INVOICES = """invoice_id,date,retailer,amount,sku,quantity
INV-1001,2024-01-10,DMart,15000,SKU-RICE-5KG,100
INV-1002,2024-01-11,BigBazaar,8400,SKU-OIL-1L,60
INV-1003,2024-01-11,Reliance,22000,SKU-FLOUR-10KG,110
INV-1004,2024-01-12,DMart,4500,SKU-SUGAR-1KG,90
INV-1005,2024-01-13,Spencer,11000,SKU-SALT-1KG,200
INV-1006,2024-01-14,BigBazaar,19500,SKU-OIL-5L,65"""

SAMPLE_PAYMENTS = """ref_no,payment_date,store_name,paid_amount,item_code,units_received,claim_description
REF-1001,2024-01-15,D Mart,15000,RICE-5KG,100,
REF-1002,2024-01-15,Big Bazaar,7560,OIL-1LTR,60,3 units received with leakage in packing
REF-1003,2024-01-16,Reliance Fresh,20900,WHEAT-10KG,105,short received - 5 bags missing from delivery
REF-1004,2024-01-17,D-Mart,4050,SUGAR-1KG,90,rate charged higher than agreed price in PO
REF-1005,2024-01-18,Spencers,11000,TABLE-SALT,200,
REF-1006,2024-01-19,Big Bazaar,17550,SUNFLOWER-OIL-5L,65,goods arrived in damaged condition - bottles broken"""

# ── Sidebar — data input ───────────────────────────────────────────────────────
with st.sidebar:
    st.header("Data Input")
    use_sample = st.toggle("Use sample data", value=True)

    if use_sample:
        inv_df  = pd.read_csv(io.StringIO(SAMPLE_INVOICES))
        pay_df  = pd.read_csv(io.StringIO(SAMPLE_PAYMENTS))
        st.success("Sample data loaded")
    else:
        inv_file = st.file_uploader("Company Invoices (CSV)", type="csv")
        pay_file = st.file_uploader("Retailer Payments (CSV)", type="csv")

        if inv_file and pay_file:
            inv_df = pd.read_csv(inv_file)
            pay_df = pd.read_csv(pay_file)
            st.success("Files uploaded")
        else:
            st.info("Upload both files to continue")
            inv_df = pay_df = None

    st.divider()
    st.markdown("**Expected columns**")
    st.markdown("*Invoices:* invoice_id, date, retailer, amount, sku, quantity")
    st.markdown("*Payments:* ref_no, payment_date, store_name, paid_amount, item_code, units_received, claim_description")

# ── Preview data ──────────────────────────────────────────────────────────────
if inv_df is not None and pay_df is not None:

    with st.expander("Preview loaded data", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Company Invoices**")
            st.dataframe(inv_df, use_container_width=True)
        with col2:
            st.markdown("**Retailer Payments**")
            st.dataframe(pay_df, use_container_width=True)

    st.divider()

    # ── Run reconciliation ────────────────────────────────────────────────────
    if st.button("▶  Run Reconciliation", type="primary", use_container_width=True):

        invoices = inv_df.to_dict("records")
        payments = pay_df.to_dict("records")
        results  = []

        progress = st.progress(0, text="Matching records...")

        for i, invoice in enumerate(invoices):
            best = find_best_match(invoice, payments)

            if best is None:
                results.append({
                    "Invoice ID":    invoice["invoice_id"],
                    "Matched Ref":   "—",
                    "Match Conf":    None,
                    "Status":        "UNMATCHED",
                    "Invoiced (₹)":  float(invoice["amount"]),
                    "Paid (₹)":      None,
                    "Short by (₹)":  None,
                    "Claim Text":    "—",
                    "AI Category":   "—",
                    "AI Confidence": None,
                    "Needs Review":  False,
                })
            else:
                payment  = next(p for p in payments if p["ref_no"] == best["ref_no"])
                invoiced = float(invoice["amount"])
                paid     = float(payment["paid_amount"])
                delta    = round(invoiced - paid, 2)
                claim    = str(payment.get("claim_description") or "").strip()
                ai       = classifier.classify(claim)

                results.append({
                    "Invoice ID":    invoice["invoice_id"],
                    "Matched Ref":   payment["ref_no"],
                    "Match Conf":    best["confidence"],
                    "Status":        "DISCREPANCY" if delta > 0 else "CLEARED",
                    "Invoiced (₹)":  invoiced,
                    "Paid (₹)":      paid,
                    "Short by (₹)":  delta if delta > 0 else 0,
                    "Claim Text":    claim if claim else "—",
                    "AI Category":   ai["category"],
                    "AI Confidence": ai["confidence"],
                    "Needs Review":  ai["review"],
                })

            progress.progress((i + 1) / len(invoices), text=f"Processing {invoice['invoice_id']}...")

        progress.empty()

        res_df = pd.DataFrame(results)

        # ── Summary metrics ───────────────────────────────────────────────────
        st.subheader("Summary")
        m1, m2, m3, m4, m5 = st.columns(5)
        cleared      = res_df[res_df["Status"] == "CLEARED"]
        discrepancies = res_df[res_df["Status"] == "DISCREPANCY"]
        unmatched    = res_df[res_df["Status"] == "UNMATCHED"]
        review       = res_df[res_df["Needs Review"] == True]
        total_short  = discrepancies["Short by (₹)"].sum()

        m1.metric("Total Invoices", len(res_df))
        m2.metric("Cleared", len(cleared))
        m3.metric("Discrepancies", len(discrepancies))
        m4.metric("Unmatched", len(unmatched))
        m5.metric("Flagged for Review", len(review))

        if total_short > 0:
            st.warning(f"Total amount short: ₹{total_short:,.0f}")

        st.divider()

        # ── Discrepancies ─────────────────────────────────────────────────────
        if len(discrepancies) > 0:
            st.subheader("Discrepancies Detected")

            for _, row in discrepancies.iterrows():
                needs_review = row["Needs Review"]
                color = "🟡" if needs_review else "🔴"
                with st.container(border=True):
                    c1, c2, c3 = st.columns([2, 2, 3])
                    with c1:
                        st.markdown(f"**{row['Invoice ID']}** → `{row['Matched Ref']}`")
                        st.caption(f"Match confidence: {row['Match Conf']}")
                    with c2:
                        st.markdown(f"Invoiced: **₹{row['Invoiced (₹)']:,.0f}**")
                        st.markdown(f"Paid: ₹{row['Paid (₹)']:,.0f}")
                        st.markdown(f"Short by: **₹{row['Short by (₹)']:,.0f}**")
                    with c3:
                        st.markdown(f"*\"{row['Claim Text']}\"*")
                        cat_color = {
                            "DAMAGE":       "🔴",
                            "SHORT_SUPPLY": "🟠",
                            "PRICE_ISSUE":  "🟡",
                            "DISCOUNT":     "🔵",
                            "NO_CLAIM":     "⚪",
                        }.get(row["AI Category"], "⚪")
                        st.markdown(
                            f"AI: {cat_color} **{row['AI Category']}** "
                            f"(confidence: {row['AI Confidence']})"
                        )
                        if needs_review:
                            st.warning("Low confidence — flagged for human review", icon="⚠️")

        # ── Cleared ───────────────────────────────────────────────────────────
        if len(cleared) > 0:
            st.subheader("Cleared")
            st.dataframe(
                cleared[["Invoice ID", "Matched Ref", "Match Conf", "Invoiced (₹)", "Paid (₹)"]],
                use_container_width=True,
                hide_index=True,
            )

        # ── Unmatched ─────────────────────────────────────────────────────────
        if len(unmatched) > 0:
            st.subheader("Unmatched Invoices")
            st.dataframe(
                unmatched[["Invoice ID", "Invoiced (₹)"]],
                use_container_width=True,
                hide_index=True,
            )

        # ── Full results download ─────────────────────────────────────────────
        st.divider()
        st.download_button(
            label="⬇  Download Full Report (CSV)",
            data=res_df.to_csv(index=False).encode("utf-8"),
            file_name="reconciliation_report.csv",
            mime="text/csv",
            use_container_width=True,
        )

else:
    st.info("Load data from the sidebar to get started.")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Intelligent Discrepancy Engine · Prototype · "
    "AI claim classification: TF-IDF + Logistic Regression · "
    "Record matching: RapidFuzz weighted scoring"
)
