# Spam-Email-Detection


import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import seaborn as sns

import matplotlib.pyplot as plt

# Sample dataset

data = {
    'Email': [
        "Congratulations! You have won a lottery. Claim now.",
        "Reminder: Your meeting with the team is tomorrow.",
        "Exclusive offer just for you. Click here to claim your prize.",
        "Project deadline extended. Please review the updated timeline.",
        "Win a free iPhone by clicking this link.",
        "Don't forget to submit your report by end of day.",
        "Claim your cash reward today! Limited time only.",
        "Meeting agenda for next week attached."
    ],
    'Label': ['spam', 'ham', 'spam', 'ham', 'spam', 'ham', 'spam', 'ham']
}

# Load dataset into a DataFrame
df = pd.DataFrame(data)

# Preprocessing: Convert labels to binary (1 for spam, 0 for ham)
df['Label'] = df['Label'].map({'spam': 1, 'ham': 0})

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['Email'], df['Label'], test_size=0.3, random_state=42)

# Vectorize the text data
vectorizer = CountVectorizer(stop_words='english')

X_train_vec = vectorizer.fit_transform(X_train)

X_test_vec = vectorizer.transform(X_test)

# Train the Naive Bayes classifier

model = MultinomialNB()

model.fit(X_train_vec, y_train)

# Predictions

y_pred = model.predict(X_test_vec)

# Evaluation

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix

cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])

plt.xlabel('Predicted')

plt.ylabel('Actual')

plt.title('Confusion Matrix')

plt.show()

# Classify new emails
new_emails = [
    "Congratulations! You are the lucky winner of a gift card.",
    "Please find the meeting notes attached."
]

new_emails_vec = vectorizer.transform(new_emails)

predictions = model.predict(new_emails_vec)

for email, pred in zip(new_emails, predictions):

print(f"Email: '{email}' -> Prediction: {'Spam' if pred == 1 else 'Ham'}")
