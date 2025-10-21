import pandas as pd
import requests
import pickle
import string
from nltk.corpus import stopwords

# Load dataset
data = pd.read_csv('dataset.csv')

# Load model and vectorizer for local prediction
model = pickle.load(open('model.pkl', 'rb'))
tfidf_vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))

def preprocess_text(text):
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = text.lower()
    stop_words = set(stopwords.words("english"))
    text = " ".join(word for word in text.split() if word not in stop_words)
    return text

def local_detect(input_text):
    preprocessed_text = preprocess_text(input_text)
    vectorized_text = tfidf_vectorizer.transform([preprocessed_text])
    result = model.predict(vectorized_text)
    return "Plagiarism Detected" if result[0] == 1 else "No Plagiarism Detected"

# Test first 10 rows
print("Testing first 10 rows from dataset:")
for i in range(min(10, len(data))):
    source = data.loc[i, 'source_text']
    plag = data.loc[i, 'plagiarized_text']
    label = data.loc[i, 'label']
    print(f"Row {i}: Label {label}")
    print(f"  Source: {source}")
    print(f"  Local prediction: {local_detect(source)}")
    print(f"  Plagiarized: {plag}")
    print(f"  Local prediction: {local_detect(plag)}")
    # Test via app
    response_source = requests.post('http://127.0.0.1:5000/detect', data={'text': source})
    response_plag = requests.post('http://127.0.0.1:5000/detect', data={'text': plag})
    app_result_source = "Plagiarism Detected" if "Plagiarism Detected" in response_source.text else "No Plagiarism Detected"
    app_result_plag = "Plagiarism Detected" if "Plagiarism Detected" in response_plag.text else "No Plagiarism Detected"
    print(f"  App prediction source: {app_result_source}")
    print(f"  App prediction plag: {app_result_plag}")
    print()

# Test unseen text
print("Testing unseen text:")
unseen_texts = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning is a subset of artificial intelligence.",
    "This is a completely original sentence not in the dataset."
]
for text in unseen_texts:
    print(f"Text: {text}")
    print(f"  Local prediction: {local_detect(text)}")
    response = requests.post('http://127.0.0.1:5000/detect', data={'text': text})
    app_result = "Plagiarism Detected" if "Plagiarism Detected" in response.text else "No Plagiarism Detected"
    print(f"  App prediction: {app_result}")
    print()

# Edge cases
print("Testing edge cases:")
edge_cases = [
    "",  # empty
    "a",  # short
    "This is a very long sentence that goes on and on and on to test how the model handles lengthy input text that might be considered for plagiarism detection in various contexts.",  # long
    "Hello! @#$%^&*()",  # special chars
    "1234567890",  # numbers
]
for text in edge_cases:
    print(f"Text: '{text}'")
    try:
        print(f"  Local prediction: {local_detect(text)}")
        response = requests.post('http://127.0.0.1:5000/detect', data={'text': text})
        app_result = "Plagiarism Detected" if "Plagiarism Detected" in response.text else "No Plagiarism Detected"
        print(f"  App prediction: {app_result}")
    except Exception as e:
        print(f"  Error: {e}")
    print()
