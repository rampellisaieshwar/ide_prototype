"""
Intelligent Discrepancy Engine — Streamlit Web App
----------------------------------------------------
Run locally:  streamlit run app.py
Deployed:     https://ideprototype.streamlit.app
"""

import streamlit as st
import pandas as pd
import io

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Intelligent Discrepancy Engine",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.write("<!-- DEBUG: Page Config Done -->", unsafe_allow_html=True)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main {
        background-color: #0e1117;
    }

    /* Glassmorphism Card */
    .st-emotion-cache-16p6y9v, .st-emotion-cache-10vuk6c {
        background: rgba(255, 255, 255, 0.03) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
        padding: 20px !important;
        backdrop-filter: blur(10px);
    }

    /* Title Styling */
    h1 {
        background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700 !important;
    }

    /* Metrics Background */
    [data-testid="stMetricValue"] {
        font-size: 2rem !important;
        font-weight: 700 !important;
        color: #4facfe !important;
    }

    /* Primary Button */
    .stButton > button {
        background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        transition: transform 0.2s ease;
    }

    .stButton > button:hover {
        transform: scale(1.02);
        box-shadow: 0 4px 15px rgba(79, 172, 254, 0.4);
    }

    /* Status Badges */
    .status-badge {
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
    }
    .status-cleared { background: rgba(0, 255, 127, 0.1); color: #00ff7f; }
    .status-discrepancy { background: rgba(255, 69, 58, 0.1); color: #ff453a; }
    .status-unmatched { background: rgba(142, 142, 147, 0.1); color: #8e8e93; }

</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("Intelligent Discrepancy Engine")
st.markdown(
    "**AI-driven reconciliation system for FMCG invoice disputes.** "
    "Matches records, detects discrepancies, and classifies claims using NLP."
)
st.divider()

# ── Loading Modules & Classifier ─────────────────────────────────────────────
@st.cache_resource
def load_engine():
    from modules.matcher import find_best_match
    from modules.classifier import ClaimClassifier
    return find_best_match, ClaimClassifier()

try:
    find_best_match, classifier = load_engine()
except Exception as e:
    st.error(f"Error loading AI model: {e}")
    st.stop()

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
    st.header("🎛️ Dashboard Controls")
    use_sample = st.toggle("Use sample data", value=True)

    if use_sample:
        inv_df  = pd.read_csv(io.StringIO(SAMPLE_INVOICES))
        pay_df  = pd.read_csv(io.StringIO(SAMPLE_PAYMENTS))
        st.success("Mode: Sample Data Ready")
    else:
        inv_file = st.file_uploader("Company Invoices (CSV)", type="csv")
        pay_file = st.file_uploader("Retailer Payments (CSV)", type="csv")

        if inv_file and pay_file:
            inv_df = pd.read_csv(inv_file)
            pay_df = pd.read_csv(pay_file)
            st.success("Files uploaded successfully")
        else:
            st.info("Please upload both CSV files to proceed.")
            inv_df = pay_df = None

    st.divider()
    with st.expander("Help & Schema"):
        st.markdown("**Expected columns**")
        st.markdown("*Invoices:* invoice_id, date, retailer, amount, sku, quantity")
        st.markdown("*Payments:* ref_no, payment_date, store_name, paid_amount, item_code, units_received, claim_description")

# ── Main Content ──────────────────────────────────────────────────────────────
if inv_df is not None and pay_df is not None:

    # Preview
    with st.expander("📊 Data Overview", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Company Invoices**")
            st.dataframe(inv_df, use_container_width=True, hide_index=True)
        with col2:
            st.markdown("**Retailer Payments**")
            st.dataframe(pay_df, use_container_width=True, hide_index=True)

    st.divider()

    # Reconciliation Logic
    if st.button("▶  Execute Reconciliation Engine", use_container_width=True):
        
        invoices = inv_df.to_dict("records")
        payments = pay_df.to_dict("records")
        results  = []

        progress_bar = st.progress(0)
        status_text = st.empty()

        for i, invoice in enumerate(invoices):
            status_text.text(f"Analyzing {invoice['invoice_id']}...")
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

            progress_bar.progress((i + 1) / len(invoices))

        status_text.empty()
        progress_bar.empty()

        res_df = pd.DataFrame(results)

        # ── Summary Metrics ───────────────────────────────────────────────────
        st.subheader("Engine Summary")
        m1, m2, m3, m4, m5 = st.columns(5)
        
        cleared      = res_df[res_df["Status"] == "CLEARED"]
        discrepancies = res_df[res_df["Status"] == "DISCREPANCY"]
        unmatched    = res_df[res_df["Status"] == "UNMATCHED"]
        review       = res_df[res_df["Needs Review"] == True]
        total_short  = discrepancies["Short by (₹)"].sum()

        m1.metric("Total", len(res_df))
        m2.metric("Cleared", len(cleared))
        m3.metric("Discrepancies", len(discrepancies), delta=len(discrepancies), delta_color="inverse")
        m4.metric("Unmatched", len(unmatched))
        m5.metric("For Review", len(review))

        if total_short > 0:
            st.error(f"Total Revenue Leakage Found: ₹{total_short:,.2f}")

        # ── Discrepancy Detail Cards ──────────────────────────────────────────
        if len(discrepancies) > 0:
            st.subheader("🚨 Detected Discrepancies")
            
            for _, row in discrepancies.iterrows():
                needs_review = row["Needs Review"]
                with st.container(border=True):
                    c1, c2, c3 = st.columns([1, 1, 1.5])
                    with c1:
                        st.markdown(f"**{row['Invoice ID']}**")
                        st.code(f"Ref: {row['Matched Ref']}")
                        st.caption(f"Match Confidence: {row['Match Conf']}")
                    with c2:
                        st.markdown(f"<span class='status-badge status-discrepancy'>DISCREPANCY</span>", unsafe_allow_html=True)
                        st.markdown(f"Leakage: **₹{row['Short by (₹)']:,.2f}**")
                        st.caption(f"Invoiced: ₹{row['Invoiced (₹)']:,.0f}")
                    with c3:
                        cat_color = {
                            "DAMAGE":       "🔴",
                            "SHORT_SUPPLY": "🟠",
                            "PRICE_ISSUE":  "🟡",
                            "DISCOUNT":     "🔵",
                        }.get(row["AI Category"], "⚪")
                        
                        st.markdown(f"**AI Reason: {cat_color} {row['AI Category']}**")
                        st.caption(f"Text: _{row['Claim Text']}_")
                        if needs_review:
                            st.warning("Low AI confidence — manual review needed.", icon="⚠️")

        # ── Bottom sections ───────────────────────────────────────────────────
        col1, col2 = st.columns(2)
        with col1:
            if len(cleared) > 0:
                st.subheader("✅ Cleared Records")
                st.dataframe(cleared[["Invoice ID", "Matched Ref", "Invoiced (₹)"]], use_container_width=True, hide_index=True)
        with col2:
            if len(unmatched) > 0:
                st.subheader("❓ Unmatched Invoices")
                st.dataframe(unmatched[["Invoice ID", "Invoiced (₹)"]], use_container_width=True, hide_index=True)

        # ── Download ──────────────────────────────────────────────────────────
        st.divider()
        st.download_button(
            label="⬇️ Export Detailed Reconciliation Report (CSV)",
            data=res_df.to_csv(index=False).encode("utf-8"),
            file_name="ide_report.csv",
            mime="text/csv",
            use_container_width=True,
        )

else:
    st.info("👋 Welcome! Use the sidebar to load your data and begin reconciliation.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "**Intelligent Discrepancy Engine v1.0** · "
    "Designed for high-throughput FMCG reconciliation."
)
