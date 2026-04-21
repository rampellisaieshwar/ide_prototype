"""
Intelligent Discrepancy Engine — Consolidated Super App
-------------------------------------------------------
A single-file deployment to ensure maximum stability on Streamlit Cloud.
Merged modules: matcher, classifier
"""

import streamlit as st
import pandas as pd
import io
import numpy as np
from datetime import datetime
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Intelligent Discrepancy Engine",
    page_icon="🔍",
    layout="wide",
)

# ── RECORD MATCHER LOGIC ─────────────────────────────────────────────────────
WEIGHT_ID     = 0.50
WEIGHT_AMOUNT = 0.35
WEIGHT_DATE   = 0.15
MATCH_THRESHOLD = 0.55

def _id_score(inv_id: str, ref_id: str) -> float:
    clean_inv = str(inv_id).replace("INV-", "").replace("INV", "").strip()
    clean_ref = str(ref_id).replace("REF-", "").replace("REF", "").strip()
    return fuzz.token_sort_ratio(clean_inv, clean_ref) / 100.0

def _amount_score(invoiced: float, paid: float) -> float:
    if invoiced == 0: return 0.0
    diff_pct = abs(invoiced - paid) / invoiced
    return max(0.0, 1.0 - (diff_pct / 0.30))

def _date_score(inv_date: str, pay_date: str) -> float:
    try:
        d1 = datetime.strptime(str(inv_date), "%Y-%m-%d")
        d2 = datetime.strptime(str(pay_date), "%Y-%m-%d")
        days = abs((d2 - d1).days)
        if days <= 7: return 1.0
        elif days >= 60: return 0.0
        else: return 1.0 - ((days - 7) / 53)
    except: return 0.5

def find_best_match(invoice: dict, payments: list) -> dict | None:
    results = []
    for p in payments:
        s_id = _id_score(invoice["invoice_id"], p["ref_no"])
        s_amt = _amount_score(float(invoice["amount"]), float(p["paid_amount"]))
        s_date = _date_score(invoice["date"], p["payment_date"])
        conf = (WEIGHT_ID * s_id + WEIGHT_AMOUNT * s_amt + WEIGHT_DATE * s_date)
        results.append({"ref_no": p["ref_no"], "confidence": round(conf, 3), "matched": conf >= MATCH_THRESHOLD})
    
    results.sort(key=lambda x: x["confidence"], reverse=True)
    best = results[0] if results else None
    return best if (best and best["matched"]) else None

# ── CLAIM CLASSIFIER LOGIC ───────────────────────────────────────────────────
TRAINING_DATA = [
    ("damaged items received", "DAMAGE"), ("goods arrived broken", "DAMAGE"), ("leaking on arrival", "DAMAGE"),
    ("short received", "SHORT_SUPPLY"), ("missing from delivery", "SHORT_SUPPLY"), ("receieved less", "SHORT_SUPPLY"),
    ("rate charged higher", "PRICE_ISSUE"), ("price mismatch", "PRICE_ISSUE"), ("wrong rate", "PRICE_ISSUE"),
    ("scheme discount not applied", "DISCOUNT"), ("trade discount missing", "DISCOUNT")
]

class ClaimClassifier:
    def __init__(self):
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), lowercase=True)),
            ("clf", LogisticRegression(max_iter=500))
        ])
        self._train()
    def _train(self):
        texts, labels = [item[0] for item in TRAINING_DATA], [item[1] for item in TRAINING_DATA]
        self.pipeline.fit(texts, labels)
    def classify(self, text: str) -> dict:
        if not text or not text.strip(): return {"category": "NO_CLAIM", "confidence": 1.0, "review": False}
        proba = self.pipeline.predict_proba([text])[0]
        idx = int(np.argmax(proba))
        conf = round(float(proba[idx]), 3)
        return {"category": self.pipeline.classes_[idx], "confidence": conf, "review": conf < 0.60}

# ── UI STYLING ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .main { background-color: #0e1117; }
    .st-emotion-cache-16p6y9v {
        background: rgba(255, 255, 255, 0.03) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
        backdrop-filter: blur(10px);
    }
    h1 {
        background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700 !important;
    }
    .stButton > button {
        background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%) !important;
        color: white !important;
        border: none !important;
        font-weight: 600 !important;
    }
    .status-badge { padding: 4px 12px; border-radius: 20px; font-size: 0.8rem; font-weight: 600; display: inline-block; }
    .status-cleared { background: rgba(0, 255, 127, 0.1); color: #00ff7f; }
    .status-discrepancy { background: rgba(255, 69, 58, 0.1); color: #ff453a; }
</style>
""", unsafe_allow_html=True)

# ── EXTREME ROBUSTNESS: Greet the cloud ──────────────────────────────────────
st.title("Intelligent Discrepancy Engine")
st.markdown("AI-driven reconciliation system for FMCG invoice disputes.")

@st.cache_resource
def get_classifier(): return ClaimClassifier()
classifier = get_classifier()

# ── SAMPLE DATA ───────────────────────────────────────────────────────────────
SAMPLE_INVOICES = """invoice_id,date,retailer,amount,sku,quantity
INV-1001,2024-01-10,DMart,15000,SKU-RICE-5KG,100
INV-1002,2024-01-11,BigBazaar,8400,SKU-OIL-1L,60
INV-1003,2024-01-11,Reliance,22000,SKU-FLOUR-10KG,110"""

SAMPLE_PAYMENTS = """ref_no,payment_date,store_name,paid_amount,item_code,units_received,claim_description
REF-1001,2024-01-15,D Mart,15000,RICE-5KG,100,
REF-1002,2024-01-15,Big Bazaar,7560,OIL-1LTR,60,3 units received with leakage in packing
REF-1003,2024-01-16,Reliance Fresh,20900,WHEAT-10KG,105,short received - 5 bags missing from delivery"""

# ── APP LOGIC ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("🎛️ Dashboard")
    use_sample = st.toggle("Use sample data", value=True)
    if use_sample:
        inv_df, pay_df = pd.read_csv(io.StringIO(SAMPLE_INVOICES)), pd.read_csv(io.StringIO(SAMPLE_PAYMENTS))
        st.success("Sample Data Loaded")
    else:
        inv_file, pay_file = st.file_uploader("Invoices"), st.file_uploader("Payments")
        inv_df = pd.read_csv(inv_file) if inv_file else None
        pay_df = pd.read_csv(pay_file) if pay_file else None

if inv_df is not None and pay_df is not None:
    if st.button("▶  Run Reconciliation", use_container_width=True):
        invoices, payments = inv_df.to_dict("records"), pay_df.to_dict("records")
        results = []
        for inv in invoices:
            best = find_best_match(inv, payments)
            if not best:
                results.append({"Invoice ID": inv["invoice_id"], "Status": "UNMATCHED", "Short": 0, "AI": "—"})
            else:
                pay = next(p for p in payments if p["ref_no"] == best["ref_no"])
                delta = round(float(inv["amount"]) - float(pay["paid_amount"]), 2)
                ai = classifier.classify(str(pay.get("claim_description") or ""))
                results.append({"Invoice ID": inv["invoice_id"], "Status": "DISCREPANCY" if delta > 0 else "CLEARED", "Short": delta, "AI": ai["category"]})
        
        res_df = pd.DataFrame(results)
        st.subheader("Summary")
        c1, c2, c3 = st.columns(3)
        c1.metric("Total", len(res_df))
        c2.metric("Discrepancies", len(res_df[res_df["Status"]=="DISCREPANCY"]))
        c3.metric("Shortage", f"₹{res_df['Short'].sum():,.2f}")
        st.dataframe(res_df, use_container_width=True)

st.caption("Engine v1.1 · Consolidated Deployment")
