# ==============================
# Spam Classifier using Naive Bayes
# Dataset: Apache SpamAssassin public corpus (raw email text)
# Includes: ROC, Top spam words, Heatmaps, Cross-validation
# ==============================

import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_curve, roc_auc_score
)

# -----------------------------
# Step 0: Preprocessing function
# -----------------------------
def preprocess_email(text):
    """Clean email text: replace numbers and URLs with tokens."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r'\b\d+\b', 'NUMBER', text)                 # Replace numbers
    text = re.sub(r'(http[s]?://\S+|www\.\S+)', 'URL', text)  # Replace URLs
    return text

# -----------------------------
# Step 1: Load dataset
# -----------------------------
df = pd.read_csv("emails_dataset.csv")  # Update with your CSV path

print(f"Dataset shape: {df.shape}")
print("First 5 rows:\n", df.head())

# Drop missing emails
df = df.dropna(subset=['email']).reset_index(drop=True)

# Preprocess emails
df['email'] = df['email'].apply(preprocess_email)

# -----------------------------
# Step 2: Prepare features and labels
# -----------------------------
X_text = df['email']
y = df['label'].values

# Bag-of-words vectorization
vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(X_text)

print(f"Feature matrix shape: {X.shape}")
print(f"Label array shape: {y.shape}")

# -----------------------------
# Step 3: Split dataset
# -----------------------------
X_train, X_test, y_train, y_test, text_train, text_test = train_test_split(
    X, y, X_text, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------
# Step 4: Train Naive Bayes
# -----------------------------
model = MultinomialNB()
model.fit(X_train, y_train)

# -----------------------------
# Step 5: Evaluate model
# -----------------------------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.4f}")

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", cm)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# -----------------------------
# Step 6: Confusion Matrix Heatmap
# -----------------------------
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Not Spam", "Spam"],
            yticklabels=["Not Spam", "Spam"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

# -----------------------------
# Step 7: Top 10 Words Indicating Spam
# -----------------------------
feature_names = np.array(vectorizer.get_feature_names_out())
spam_index = 1  # index for spam class

top_indices = np.argsort(model.feature_log_prob_[spam_index])[-10:]
top_words = feature_names[top_indices]
top_probs = model.feature_log_prob_[spam_index][top_indices]

print("\nTop 10 words indicating spam:", top_words.tolist())

# Bar chart
plt.figure(figsize=(8, 4))
sns.barplot(x=top_words, y=top_probs, color="red")
plt.xticks(rotation=45)
plt.ylabel("Log Probability")
plt.title("Top 10 Words Indicating Spam")
plt.tight_layout()
plt.savefig("top_spam_words.png")
plt.show()

# -----------------------------
# Step 8: ROC Curve & AUC
# -----------------------------
y_prob = model.predict_proba(X_test)[:, 1]  # spam probabilities
fpr, tpr, _ = roc_curve(y_test, y_prob)
auc_score = roc_auc_score(y_test, y_prob)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {auc_score:.3f})", color="darkorange", lw=2)
plt.plot([0, 1], [0, 1], "k--", lw=1)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("roc_curve.png")
plt.show()

print(f"\nAUC Score: {auc_score:.4f}")

# -----------------------------
# Step 9: 5-Fold Cross-Validation
# -----------------------------
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X, y, cv=skf, scoring="accuracy")
print(f"5-Fold Cross-Validation Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# -----------------------------
# Step 10: Predict a single email (raw text)
# -----------------------------
sample_email = text_test.iloc[0]
sample_vector = vectorizer.transform([sample_email])
prediction = model.predict(sample_vector)[0]
print(f"\nSample email:\n{sample_email[:200]}...")  # show first 200 chars
print(f"Prediction: {'Spam' if prediction == 1 else 'Not Spam'}")
