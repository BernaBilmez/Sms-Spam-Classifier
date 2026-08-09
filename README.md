# SMS Spam Classifier — TF-IDF + Naive Bayes / Logistic Regression

A text classification pipeline that filters spam SMS messages while avoiding false positives on legitimate messages, with a focus on **evaluating models honestly** rather than optimising for a single misleading metric.

**Live notebook:** [kaggle.com/code/bernabilmezlondon/sms-spam-classifier-tf-idf-naive-bayes-ld5010](https://www.kaggle.com/code/bernabilmezlondon/sms-spam-classifier-tf-idf-naive-bayes-ld5010)

---

## The problem

Messaging platforms need to automatically filter spam SMS messages without blocking legitimate ones. This is a binary classification task with a real-world constraint most tutorials skip: the dataset is **imbalanced** — only ~13% of messages are spam — which makes accuracy alone a misleading measure of success.

## Approach

1. **Preprocessing** — raw SMS text converted into numerical features using TF-IDF vectorisation
2. **Modelling** — trained and compared three classifiers:
   - Naive Bayes (baseline)
   - Logistic Regression (default settings)
   - Logistic Regression with `class_weight='balanced'`
3. **Evaluation** — assessed on accuracy, precision, recall, and confusion matrices, not accuracy alone
4. **Inference test** — the best model wrapped in a callable function and tested on entirely new, unseen messages, simulating basic deployment behaviour

## Results

| Model | Accuracy | Spam Recall | Spam Missed (of 150) |
|---|---|---|---|
| Naive Bayes | 96.7% | 75% | 37 |
| Logistic Regression (default) | 95.2% | 67% | 50 |
| **Logistic Regression (balanced)** | **97.8%** | **91%** | **14** |

![Model comparison](images/three_model_comparison.png)

**Key finding:** the first model looked strong at 96.7% accuracy, but was actually missing a quarter of real spam messages. Simply trying a more complex algorithm (Logistic Regression) made this *worse*, not better. The fix was addressing the underlying class imbalance directly — reducing missed spam from 37 to 14 while also achieving the highest overall accuracy.

## Tech stack

`Python` · `pandas` · `scikit-learn` · `matplotlib` · TF-IDF vectorisation · Naive Bayes · Logistic Regression

## Dataset

[SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset) (UCI Machine Learning / Kaggle) — 5,572 labelled SMS messages.

## Run it yourself

```bash
pip install -r requirements.txt
python sms_spam_classifier.py
```

(Requires `spam.csv` from the dataset link above, placed in the same directory.)

---

*Built as part of an AI/ML Engineering portfolio project. Full write-up and reflection available on request.*
