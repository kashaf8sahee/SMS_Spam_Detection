# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Import necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_curve, auc
)


# Config (set local CSV path)
# Local (Windows) example:
file_path = r"E:\SMS_Spam_Detection\spam.csv"   # <- change if needed

# Choose model: 'svm' (recommended), 'nb' (baseline), or 'logreg'
MODEL_CHOICE = 'svm'

# Load the dataset
# The original CSV has extra unnamed columns; we’ll select v1 (label) and v2 (message)
df = pd.read_csv(file_path, encoding='ISO-8859-1')
df = df[['v1', 'v2']]
df.columns = ['label', 'message']

# Display the first few rows of the dataset
df.head()

 
# Basic cleaning (non-destructive, keeps theme)
 
# Strip whitespace and drop empty messages
df['message'] = df['message'].astype(str).str.strip()
df = df[df['message'].str.len() > 0].reset_index(drop=True)

# Check for missing values
df.isnull().sum()

 
# Plot the distribution of spam and ham messages
 
sns.countplot(x='label', data=df)
plt.title('Distribution of Spam and Ham Messages')
plt.xlabel('Label')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

 
# Convert text data to numerical data
USE_TFIDF = True

if USE_TFIDF:
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True
    )
else:
    vectorizer = CountVectorizer()

X = vectorizer.fit_transform(df['message'])
y = df['label'].map({'ham': 0, 'spam': 1}).values

 
# Split the data into training and testing sets
 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

 
# Train a classifier (better and efficient by default)
 
if MODEL_CHOICE == 'svm':
    # Fast linear SVM (strong baseline for sparse text)
    model = LinearSVC()  # hinge loss, decision_function available
elif MODEL_CHOICE == 'logreg':
    # Also strong; liblinear handles sparse, smaller datasets well
    model = LogisticRegression(max_iter=2000, solver='liblinear')
else:
    # Baseline Naive Bayes
    model = MultinomialNB()

model.fit(X_train, y_train)

 
# Make predictions on the test data
 
y_pred = model.predict(X_test)

 
# Evaluate the model
 
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred, target_names=['ham', 'spam'])

print(f"Accuracy: {accuracy:.4f}")
print("\nConfusion Matrix:\n", conf_matrix)
print("\nClassification Report:\n", class_report)

 
# Confusion Matrix plot
 
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['ham', 'spam'], yticklabels=['ham', 'spam'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()

 
# ROC Curve and AUC (binary)
 
# We can build ROC using decision scores (preferred) or probabilities.
# LinearSVC: use decision_function; NB/LogReg: use predict_proba (or decision_function for LogReg).
if hasattr(model, "decision_function"):
    y_scores = model.decision_function(X_test)
elif hasattr(model, "predict_proba"):
    # use probability of positive class
    y_scores = model.predict_proba(X_test)[:, 1]
else:
    y_scores = None

if y_scores is not None:
    fpr, tpr, _ = roc_curve(y_test, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve (Spam vs Ham)')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()

 
# Inference helper (optional)
 
def predict_sms(texts):
    x_new = vectorizer.transform(texts)
    preds = model.predict(x_new)
    return ['spam' if p == 1 else 'ham' for p in preds]
