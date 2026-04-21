"""
Claim Classifier — AI/NLP Module
---------------------------------
Replaces brittle keyword matching with a trained NLP pipeline.

Why this matters:
    Retailers describe the same problem in completely different ways.
    "leakage", "oil spilled", "packing issue", "items broken" —
    all of these are damage claims, but none contain the word "damage".
    A rule-based system misses all of them. This classifier does not.

Approach:
    TF-IDF vectorizer converts claim text into numeric features.
    Logistic Regression classifier maps those features to a claim category.
    Confidence score is output alongside every prediction.
    Low confidence routes the case to human review.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
import numpy as np


# ---------------------------------------------------------------------------
# Training data
# Real-world claim descriptions are messy. The training set deliberately
# includes diverse language — abbreviations, partial sentences, spelling
# variations — to reflect what retailers actually write.
# ---------------------------------------------------------------------------

TRAINING_DATA = [
    # DAMAGE
    ("damaged items received",                          "DAMAGE"),
    ("goods arrived broken",                            "DAMAGE"),
    ("3 units leaking on arrival",                      "DAMAGE"),
    ("oil spilled inside carton",                       "DAMAGE"),
    ("packaging torn and product exposed",              "DAMAGE"),
    ("items crushed during transit",                    "DAMAGE"),
    ("bottles broken in the crate",                     "DAMAGE"),
    ("leakage found in packing",                        "DAMAGE"),
    ("products in bad condition",                       "DAMAGE"),
    ("seals broken on arrival",                         "DAMAGE"),
    ("carton wet and damaged",                          "DAMAGE"),
    ("packing issue found",                             "DAMAGE"),
    ("goods not in sellable condition",                 "DAMAGE"),
    ("container punctured",                             "DAMAGE"),
    ("item damaged due to moisture",                    "DAMAGE"),

    # SHORT_SUPPLY
    ("short received",                                  "SHORT_SUPPLY"),
    ("5 units missing from delivery",                   "SHORT_SUPPLY"),
    ("quantity short against invoice",                  "SHORT_SUPPLY"),
    ("received less than invoiced",                     "SHORT_SUPPLY"),
    ("partial delivery only",                           "SHORT_SUPPLY"),
    ("3 cases not delivered",                           "SHORT_SUPPLY"),
    ("short supply due to loading error",               "SHORT_SUPPLY"),
    ("only 45 units received out of 50",                "SHORT_SUPPLY"),
    ("missing items in shipment",                       "SHORT_SUPPLY"),
    ("less quantity sent",                              "SHORT_SUPPLY"),
    ("10 units short",                                  "SHORT_SUPPLY"),
    ("delivery incomplete",                             "SHORT_SUPPLY"),
    ("shortage of 2 cartons",                           "SHORT_SUPPLY"),
    ("not all items delivered",                         "SHORT_SUPPLY"),

    # PRICE_ISSUE
    ("rate charged higher than agreed",                 "PRICE_ISSUE"),
    ("price mismatch with purchase order",              "PRICE_ISSUE"),
    ("incorrect price on invoice",                      "PRICE_ISSUE"),
    ("rate difference found",                           "PRICE_ISSUE"),
    ("invoice price does not match PO",                 "PRICE_ISSUE"),
    ("charged at wrong rate",                           "PRICE_ISSUE"),
    ("price discrepancy",                               "PRICE_ISSUE"),
    ("MRP changed but old rate billed",                 "PRICE_ISSUE"),
    ("billing rate mismatch",                           "PRICE_ISSUE"),
    ("wrong price applied",                             "PRICE_ISSUE"),

    # DISCOUNT
    ("scheme discount not applied",                     "DISCOUNT"),
    ("trade discount missing",                          "DISCOUNT"),
    ("promotional deduction claimed",                   "DISCOUNT"),
    ("cash discount as per agreement",                  "DISCOUNT"),
    ("festival discount not reflected",                 "DISCOUNT"),
    ("quarterly scheme not adjusted",                   "DISCOUNT"),
    ("special offer discount",                          "DISCOUNT"),
    ("discount as per trade terms",                     "DISCOUNT"),
    ("annual scheme deduction",                         "DISCOUNT"),
    ("agreed rebate not applied",                       "DISCOUNT"),
]

CONFIDENCE_THRESHOLD = 0.60  # Below this → route to human review


class ClaimClassifier:
    """
    NLP pipeline for classifying retailer claim descriptions.

    Pipeline:
        Text input → TF-IDF features → Logistic Regression → Category + Confidence
    """

    def __init__(self):
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                ngram_range=(1, 2),   # unigrams and bigrams for better context
                min_df=1,
                lowercase=True,
                strip_accents="unicode"
            )),
            ("clf", LogisticRegression(
                max_iter=500,
                C=1.0,
                solver="lbfgs"
            ))
        ])
        self.trained = False
        self._train()

    def _train(self):
        texts  = [item[0] for item in TRAINING_DATA]
        labels = [item[1] for item in TRAINING_DATA]
        self.pipeline.fit(texts, labels)
        self.trained = True

    def classify(self, claim_text: str) -> dict:
        """
        Classify a claim description.

        Returns:
            {
                "category":   str,   # predicted class
                "confidence": float, # probability of predicted class
                "review":     bool   # True if confidence is below threshold
            }
        """
        if not claim_text or not claim_text.strip():
            return {
                "category":   "NO_CLAIM",
                "confidence": 1.0,
                "review":     False
            }

        proba  = self.pipeline.predict_proba([claim_text])[0]
        classes = self.pipeline.classes_
        best_idx = int(np.argmax(proba))

        category   = classes[best_idx]
        confidence = round(float(proba[best_idx]), 3)
        needs_review = confidence < CONFIDENCE_THRESHOLD

        return {
            "category":   category,
            "confidence": confidence,
            "review":     needs_review
        }
