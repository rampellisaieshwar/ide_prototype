"""
Intelligent Discrepancy Engine — Full 4-Stage Reconciliation Prototype
----------------------------------------------------------------------
Consolidated Super App featuring the complete Chain of Custody audit:
SAP -> Warehouse -> Driver -> Retailer
"""

import streamlit as st
import pandas as pd
import io
import numpy as np
from datetime import datetime
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Intelligent Discrepancy Engine | Full Pipeline",
    page_icon="🔍",
    layout="wide",
)

# ── CORE LOGIC: 4-WAY RECONCILIATION ─────────────────────────────────────────

def reconcile_chain(sap_record, warehouse_records, pod_records, grn_records):
    """
    Traces a single invoice through all 4 stages of evidence.
    """
    invoice_id = sap_record["invoice_id"]
    sap_qty = float(sap_record["quantity"])
    sap_amt = float(sap_record["amount"])
    sku = sap_record["sku"]

    # 1. Matching Warehouse Stage
    wh = next((r for r in warehouse_records if r["invoice_id"] == invoice_id), None)
    wh_qty = float(wh["qty_dispatched"]) if wh else 0.0
    
    # 2. Matching Driver POD Stage
    pod = next((r for r in pod_records if r["invoice_id"] == invoice_id), None)
    pod_qty = float(pod["qty_delivered"]) if pod else 0.0
    
    # 3. Matching Retailer Stage
    grn = next((r for r in grn_records if r["ref_no"].endswith(invoice_id.split("-")[-1])), None)
    grn_qty = float(grn["units_received"]) if grn else 0.0
    grn_paid = float(grn["paid_amount"]) if grn else 0.0
    claim_text = str(grn.get("claim_description") or "") if grn else ""

    # Determine Leakage Point
    results = {
        "id": invoice_id,
        "sku": sku,
        "sap": sap_qty,
        "wh": wh_qty,
        "pod": pod_qty,
        "grn": grn_qty,
        "paid": grn_paid,
        "expected_paid": sap_amt,
        "claim": claim_text,
        "stages": []
    }

    # Audit the chain
    if wh_qty < sap_qty:
        results["stages"].append({"status": "MISSING", "stage": "WAREHOUSE", "delta": sap_qty - wh_qty})
    if pod_qty < wh_qty:
        results["stages"].append({"status": "LOSS", "stage": "TRANSIT", "delta": wh_qty - pod_qty})
    if grn_qty < pod_qty:
        results["stages"].append({"status": "CLAIM", "stage": "RETAILER", "delta": pod_qty - grn_qty})
    
    return results

# ── CLAIM CLASSIFIER (Hybrid: Fuzzy + Naive Bayes) ──────────────────────────
TRAINING_DATA = [
    # DAMAGE
    ("damaged items received", "DAMAGE"), ("goods arrived broken", "DAMAGE"), ("leaking on arrival", "DAMAGE"),
    ("broken bottles found in crate", "DAMAGE"), ("packing torn and product exposed", "DAMAGE"),
    ("items crushed during transit", "DAMAGE"), ("carton wet and damaged", "DAMAGE"),
    ("product leaked in shipment", "DAMAGE"), ("physical damage to goods", "DAMAGE"),
    
    # SHORT_SUPPLY
    ("short received", "SHORT_SUPPLY"), ("missing from delivery", "SHORT_SUPPLY"), ("receieved less", "SHORT_SUPPLY"),
    ("short from delivery", "SHORT_SUPPLY"), ("received 95 units out of 100", "SHORT_SUPPLY"),
    ("partial shipment received", "SHORT_SUPPLY"), ("box missing during offloading", "SHORT_SUPPLY"),
    ("units not found in crate", "SHORT_SUPPLY"), ("shipment incomplete", "SHORT_SUPPLY"),
    
    # PRICE_ISSUE
    ("rate charged higher", "PRICE_ISSUE"), ("price mismatch", "PRICE_ISSUE"), ("wrong rate", "PRICE_ISSUE"),
    ("incorrect billing rate", "PRICE_ISSUE"), ("MRP changed but old rate billed", "PRICE_ISSUE"),
    ("rate difference found", "PRICE_ISSUE"), ("PO price vs Invoice price mismatch", "PRICE_ISSUE"),
    
    # DISCOUNT
    ("scheme discount not applied", "DISCOUNT"), ("trade discount missing", "DISCOUNT"),
    ("promotional discount not shown", "DISCOUNT"), ("festival offer rebate missing", "DISCOUNT"),
    
    # SCHEME_AGREEMENT (New Category)
    ("5% annual growth scheme deduction as per Q4 agreement", "SCHEME_AGREEMENT"),
    ("deduction as per annual contract", "SCHEME_AGREEMENT"),
    ("quarterly scheme adjustment", "SCHEME_AGREEMENT"),
    ("loyalty rebate as per agreement between parties", "SCHEME_AGREEMENT"),
    ("volume target achievement scheme", "SCHEME_AGREEMENT"),
    ("contractual discount for Q3", "SCHEME_AGREEMENT"),
    ("agreement mismatch in scheme application", "SCHEME_AGREEMENT")
]

class ClaimClassifier:
    def __init__(self):
        self.pipeline = Pipeline([
            ("vect", TfidfVectorizer(ngram_range=(1, 2), lowercase=True)),
            ("clf", MultinomialNB(alpha=0.01))
        ])
        self._train()
        
    def _train(self):
        texts, labels = [item[0] for item in TRAINING_DATA], [item[1] for item in TRAINING_DATA]
        self.pipeline.fit(texts, labels)
        
    def classify(self, text: str) -> dict:
        if not text or not text.strip(): return {"category": "NO_CLAIM", "confidence": 1.0, "review": False}
        
        # 1. Fuzzy Logic Bridge (High Confidence for known patterns)
        best_match = None
        best_score = 0
        for train_text, label in TRAINING_DATA:
            score = fuzz.ratio(text.lower(), train_text.lower())
            if score > best_score:
                best_score = score
                best_match = label
        
        if best_score > 90:
            return {"category": best_match, "confidence": round(best_score/100.0, 3), "review": False}
            
        # 2. AI Probabilistic Model (For unseen variations)
        proba = self.pipeline.predict_proba([text])[0]
        idx = int(np.argmax(proba))
        conf = round(float(proba[idx]), 3)
        return {"category": self.pipeline.classes_[idx], "confidence": conf, "review": conf < 0.60}

@st.cache_resource
def get_classifier(data_len): 
    return ClaimClassifier()

# ── UI STYLING ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .main { background-color: #0e1117; }
    
    .stTabs [data-baseweb="tab-list"] { gap: 24px; background-color: transparent; }
    .stTabs [data-baseweb="tab"] {
        height: 60px; background-color: rgba(255, 255, 255, 0.05); border-radius: 4px; font-weight: 600;
    }
    .stTabs [aria-selected="true"] { background-color: rgba(79, 172, 254, 0.1) !important; border-bottom: 2px solid #4facfe !important; }

    .vision-card { background: rgba(255, 255, 255, 0.03); border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 12px; padding: 1.5rem; margin-bottom: 1rem; line-height: 1.6; }
    
    /* Audit Step Visualization */
    .audit-step { padding: 10px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.1); margin: 0 5px; flex: 1; text-align: center; font-size: 0.9rem; }
    .step-label { font-size: 0.7rem; color: #8e8e93; text-transform: uppercase; margin-bottom: 4px; }
    .step-val { font-weight: 700; font-size: 1.1rem; }
    .step-match { border-color: rgba(0, 255, 127, 0.3); background: rgba(0, 255, 127, 0.05); }
    .step-fail { border-color: rgba(255, 69, 58, 0.3); background: rgba(255, 69, 58, 0.05); }

</style>
""", unsafe_allow_html=True)

# ── MAIN NAVIGATION ───────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["Vision & Context", "System Architecture", "Live 4-Stage Prototype"])

# ── TAB 1: THE VISION ────────────────────────────────────────────────────────
with tab1:
    st.markdown("<h1 style='background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>Intelligent Discrepancy Engine</h1>", unsafe_allow_html=True)
    st.markdown("""
    <div class="vision-card">
        <i>"When I first read this problem statement, my honest reaction was that this is not a small assignment... Reconciliation in FMCG sounds like a data-matching task on the surface, but the more I thought through how it actually works in the real world, the more layers I found."</i>
    </div>
    """, unsafe_allow_html=True)
    
    st.subheader("The Problem, As I See It")
    st.markdown("""
    <div class="vision-card">
        <b>This is not a simple database join.</b> Invoice numbers do not match across systems. SKU codes differ. Dates shift. 
        Amounts are partially paid with reasons written on hand-scanned paper. This is a <b>probabilistic record linkage and reasoning problem under uncertainty.</b>
    </div>
    """, unsafe_allow_html=True)

# ── TAB 2: ARCHITECTURE ──────────────────────────────────────────────────────
with tab2:
    st.subheader("The Multi-Document Evidence Chain")
    st.markdown("Reconciliation is high-confidence because the system uses all four evidence stages:")
    
    c1, c2, c3, c4 = st.columns(4)
    c1.info("**1. SAP Invoice**\n\nThe original intent.")
    c2.info("**2. Warehouse DCC**\n\nProof of dispatch condition.")
    c3.info("**3. Driver POD**\n\nProof of physical handover.")
    c4.info("**4. Retailer Document**\n\nFinal receipt and payment.")

    st.markdown("---")
    st.subheader("System Lifecycle")
    st.markdown("""
    **Evidence Flow:**
    `SAP Invoice` ➔ `Warehouse DCC` ➔ `Driver POD` ➔ `Retailer GRN`
    
    **Engine Flow:**
    `Audit Chain` ➔ `Detect Delta` ➔ `AI Classification` ➔ `Resolution`
    """)

# ── TAB 3: LIVE 4-STAGE PROTOTYPE ────────────────────────────────────────────
with tab3:
    classifier = get_classifier(len(TRAINING_DATA))

    # ── SAMPLE DATA GENERATION ────────────────────────────────────────────────
    SAMPLE_INVOICES = """invoice_id,sku,amount,quantity,date
    INV-1001,SKU-RICE-5KG,15000,100,2024-01-10
    INV-1002,SKU-OIL-1L,8400,60,2024-01-11
    INV-1003,SKU-FLOUR-10KG,22000,110,2024-01-11
    INV-1004,SKU-WHEAT-50KG,50000,500,2024-01-12"""

    SAMPLE_WAREHOUSE = """invoice_id,qty_dispatched,condition
    INV-1001,100,GOOD
    INV-1002,60,GOOD
    INV-1003,105,DAMAGED_CARTON
    INV-1004,500,GOOD"""

    SAMPLE_POD = """invoice_id,qty_delivered,remarks
    INV-1001,100,Handover Success
    INV-1002,57,3 Units Broken in Transit
    INV-1003,105,Received
    INV-1004,500,Signed and Verified"""

    SAMPLE_RETAILER = """ref_no,paid_amount,units_received,claim_description
    REF-1001,15000,100,
    REF-1002,7500,57,broken bottles found in crate
    REF-1003,20000,100,short from delivery
    REF-1004,47500,500,5% annual growth scheme deduction as per Q4 agreement"""

    # ── SIDEBAR CONTROLS ──────────────────────────────────────────────────────
    with st.sidebar:
        st.header("🎛️ Pipeline Config")
        use_sample = st.toggle("Use Full Pipeline Sample", value=True)
        if use_sample:
            sap_df = pd.read_csv(io.StringIO(SAMPLE_INVOICES))
            wh_df = pd.read_csv(io.StringIO(SAMPLE_WAREHOUSE))
            pod_df = pd.read_csv(io.StringIO(SAMPLE_POD))
            grn_df = pd.read_csv(io.StringIO(SAMPLE_RETAILER))
            st.success("4-Stage Data Injected")
        else:
            sap_file = st.file_uploader("1. SAP Invoices")
            wh_file = st.file_uploader("2. Warehouse DCC")
            pod_file = st.file_uploader("3. Driver PODs")
            grn_file = st.file_uploader("4. Retailer Payments")
            sap_df = pd.read_csv(sap_file) if sap_file else None
            wh_df = pd.read_csv(wh_file) if wh_file else None
            pod_df = pd.read_csv(pod_file) if pod_file else None
            grn_df = pd.read_csv(grn_file) if grn_file else None

    # ── EXECUTION ─────────────────────────────────────────────────────────────
    if all(v is not None for v in [sap_df, wh_df, pod_df, grn_df]):
        if st.button("▶  Run Full Chain Reconciliation", use_container_width=True):
            saps = sap_df.to_dict("records")
            results = []
            
            for s_record in saps:
                report = reconcile_chain(
                    s_record, wh_df.to_dict("records"), 
                    pod_df.to_dict("records"), grn_df.to_dict("records")
                )
                results.append(report)
            
            # Rendering Audit Results
            for r in results:
                with st.container(border=True):
                    # Chain of Custody Visualization
                    cols = st.columns([1, 1, 1, 1, 1, 2])
                    
                    def get_box(label, val, prev=None):
                        is_diff = prev is not None and val != prev
                        cls = "step-fail" if is_diff else "step-match"
                        return f"<div class='audit-step {cls}'><div class='step-label'>{label}</div><div class='step-val'>{val}</div></div>"

                    cols[0].markdown(get_box("SAP", r["sap"]), unsafe_allow_html=True)
                    cols[1].markdown(get_box("WH", r["wh"], r["sap"]), unsafe_allow_html=True)
                    cols[2].markdown(get_box("POD", r["pod"], r["wh"]), unsafe_allow_html=True)
                    cols[3].markdown(get_box("GRN", r["grn"], r["pod"]), unsafe_allow_html=True)
                    
                    revenue_leak = round(r["expected_paid"] - r["paid"], 2)
                    leak_status = "🔴 LEAKAGE" if revenue_leak > 0 else "🟢 CLEARED"
                    cols[4].markdown(f"**{r['id']}**\n\n{leak_status}")
                    
                    with cols[5]:
                        if revenue_leak > 0:
                            st.markdown(f"Leak: **₹{revenue_leak:,.0f}**")
                            if r["claim"]:
                                ai = classifier.classify(r["claim"])
                                conf_color = "#00ff7f" if ai["confidence"] > 0.8 else "#ff453a"
                                st.markdown(f"AI: <span style='color:{conf_color}; font-weight:700'>{ai['category']}</span> (Confident: {ai['confidence']})", unsafe_allow_html=True)
                                st.caption(f"Text: _{r['claim']}_")
                        else:
                            st.success("Reconciliation Perfect")

    else:
        st.info("👋 Use the sidebar to load the 4-stage evidence chain and begin the audit.")

st.divider()
st.caption("Engine v2.4 · Hybrid Intelligent Auditor · Consolidated")
