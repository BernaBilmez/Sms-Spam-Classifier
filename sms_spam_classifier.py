"""
SMS Spam Classifier — TF-IDF + Naive Bayes / Logistic Regression
==================================================================
Author: Berna Bilmez

A text classification pipeline that filters spam SMS messages while
avoiding false positives on legitimate ("ham") messages. Compares three
models and addresses class imbalance directly.

Dataset: SMS Spam Collection Dataset (UCI Machine Learning / Kaggle)
Full interactive notebook: https://www.kaggle.com/code/bernabilmezlondon/sms-spam-classifier-tf-idf-naive-bayes-ld5010
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


# ---------------------------------------------------------------------------
# 1. Load and prepare the dataset
# ---------------------------------------------------------------------------
df = pd.read_csv('spam.csv', encoding='latin-1')
df = df[['v1', 'v2']]
df.columns = ['label', 'message']

print(df.head())
print(df['label'].value_counts())

X_train, X_test, y_train, y_test = train_test_split(
    df['message'], df['label'], test_size=0.2, random_state=42
)

# ---------------------------------------------------------------------------
# 2. Preprocessing — TF-IDF vectorisation
# ---------------------------------------------------------------------------
vectorizer = TfidfVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ---------------------------------------------------------------------------
# 3. Model 1 — Naive Bayes (baseline)
# ---------------------------------------------------------------------------
nb_model = MultinomialNB()
nb_model.fit(X_train_vec, y_train)
y_pred_nb = nb_model.predict(X_test_vec)

print("\n--- Naive Bayes ---")
print("Accuracy:", accuracy_score(y_test, y_pred_nb))
print(confusion_matrix(y_test, y_pred_nb, labels=['ham', 'spam']))
print(classification_report(y_test, y_pred_nb))

# ---------------------------------------------------------------------------
# 4. Model 2 — Logistic Regression (default settings)
# ---------------------------------------------------------------------------
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_vec, y_train)
y_pred_lr = lr_model.predict(X_test_vec)

print("\n--- Logistic Regression (default) ---")
print("Accuracy:", accuracy_score(y_test, y_pred_lr))
print(confusion_matrix(y_test, y_pred_lr, labels=['ham', 'spam']))
print(classification_report(y_test, y_pred_lr))

# ---------------------------------------------------------------------------
# 5. Model 3 — Logistic Regression with class_weight='balanced'
#    Addresses the ~13% spam / 87% ham class imbalance directly.
# ---------------------------------------------------------------------------
lr_balanced = LogisticRegression(max_iter=1000, class_weight='balanced')
lr_balanced.fit(X_train_vec, y_train)
y_pred_lr_bal = lr_balanced.predict(X_test_vec)

print("\n--- Logistic Regression (balanced) ---")
print("Accuracy:", accuracy_score(y_test, y_pred_lr_bal))
print(confusion_matrix(y_test, y_pred_lr_bal, labels=['ham', 'spam']))
print(classification_report(y_test, y_pred_lr_bal))

# ---------------------------------------------------------------------------
# 6. Visualise: confusion matrices for all three models side by side
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
results = [
    (y_pred_nb, "Naive Bayes"),
    (y_pred_lr, "Logistic Regression (default)"),
    (y_pred_lr_bal, "Logistic Regression (balanced)")
]
for ax, (preds, title) in zip(axes, results):
    cm = confusion_matrix(y_test, preds, labels=['ham', 'spam'])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['ham', 'spam'])
    disp.plot(cmap='Blues', ax=ax, colorbar=False)
    ax.set_title(title, fontsize=12)
plt.tight_layout()
plt.savefig('images/three_model_comparison.png', dpi=150)
plt.show()

# ---------------------------------------------------------------------------
# 7. Deployment-style test — inference on brand-new, unseen messages
# ---------------------------------------------------------------------------
def predict_spam(message):
    vec = vectorizer.transform([message])
    prediction = lr_balanced.predict(vec)[0]
    prob = lr_balanced.predict_proba(vec)[0]
    confidence = max(prob) * 100
    return f"Prediction: {prediction.upper()} ({confidence:.1f}% confidence)"

test_messages = [
    "Congratulations! You've won a free prize, call now!",
    "Hey, are we still meeting for lunch tomorrow?",
    "URGENT: Your account has been suspended, click here immediately",
    "Can you send me the report when you get a chance?"
]

print("\n--- Inference test on unseen messages ---")
for msg in test_messages:
    print(f"Input: {msg}")
    print(f"-> {predict_spam(msg)}\n")
