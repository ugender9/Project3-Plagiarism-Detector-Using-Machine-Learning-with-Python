import pandas as pd
import pickle
import string
from nltk.corpus import stopwords
import pandas as pd
import pickle
import string
from nltk.corpus import stopwords
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer

# Load dataset
data = pd.read_csv("dataset.csv")

def preprocess_text(text):
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Convert to lowercase
    text = text.lower()
    # Remove stop words
    stop_words = set(stopwords.words("english"))
    text = " ".join(word for word in text.split() if word not in stop_words)
    return text

# Preprocess
data["source_text"] = data["source_text"].apply(preprocess_text)
data["plagiarized_text"] = data["plagiarized_text"].apply(preprocess_text)

# Prepare training data: individual texts with labels
texts = data["source_text"].tolist() + data["plagiarized_text"].tolist()
labels = []
for i in range(len(data)):
    labels.append(0)  # source_text is original (0)
    labels.append(1 if data.loc[i, 'label'] == 1 else 0)  # plagiarized_text is 1 if pair is plagiarized, else 0

# Vectorize
tfidf_vectorizer = TfidfVectorizer()
X = tfidf_vectorizer.fit_transform(texts)
y = labels

# Train model (using SVM as in original notebook)
model = SVC(kernel='linear', random_state=42)
model.fit(X, y)

# Save
pickle.dump(model, open("model.pkl", 'wb'))
pickle.dump(tfidf_vectorizer, open('tfidf_vectorizer.pkl', 'wb'))

print("Model retrained and saved.")
