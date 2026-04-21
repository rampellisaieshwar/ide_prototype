# Intelligent Discrepancy Engine (IDE)

AI-driven reconciliation system for FMCG invoice disputes.

## What it does

- Matches company invoices against retailer payment records using fuzzy weighted scoring
- Detects discrepancies (amount, quantity, reference mismatches)
- Classifies claim descriptions using an NLP model (TF-IDF + Logistic Regression)
- Routes low-confidence classifications to human review

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Project structure

```
app.py                  # Streamlit web app
main.py                 # Command-line version
modules/
    classifier.py       # AI claim classification (TF-IDF + Logistic Regression)
    matcher.py          # Weighted fuzzy record matching (RapidFuzz)
data/
    company_invoices.csv
    retailer_payments.csv
requirements.txt
```

## AI component

Claim classification replaces brittle keyword rules with a trained NLP pipeline.
A retailer writing "leakage in packing" or "bottles broken in transit" does not
use the word "damage" — but the model classifies both correctly as DAMAGE_CLAIM.

The model outputs a confidence score alongside every prediction. Cases below the
threshold are automatically flagged for human review.

## Tech stack

| Component | Technology |
|-----------|-----------|
| Web app | Streamlit |
| Record matching | RapidFuzz (fuzzy scoring) |
| Claim classification | scikit-learn (TF-IDF + Logistic Regression) |
| Data handling | pandas |
