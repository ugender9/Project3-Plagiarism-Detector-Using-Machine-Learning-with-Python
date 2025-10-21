from flask import Flask, render_template, request, redirect, url_for
import pickle
import string
from nltk.corpus import stopwords

app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))
tfidf_vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))

def preprocess_text(text):
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Convert to lowercase
    text = text.lower()
    # Remove stop words
    stop_words = set(stopwords.words("english"))
    text = " ".join(word for word in text.split() if word not in stop_words)
    return text

def detect(input_text):
    if not input_text.strip():
        return "No Plagiarism Detected"  # Handle empty input
    preprocessed_text = preprocess_text(input_text)
    if not preprocessed_text.strip():
        return "No Plagiarism Detected"  # Handle input that becomes empty after preprocessing
    try:
        vectorized_text = tfidf_vectorizer.transform([preprocessed_text])
        result = model.predict(vectorized_text)
        return "Plagiarism Detected" if result[0] == 1 else "No Plagiarism Detected"
    except Exception as e:
        print(f"Error in detection: {e}")
        return "No Plagiarism Detected"  # Fallback

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/detect', methods=['GET', 'POST'])
def detect_plagiarism():
    if request.method == 'POST':
        input_text = request.form.get('text', '').strip()
        if not input_text:
            detection_result = "No Plagiarism Detected"  # Handle empty form submission
        else:
            detection_result = detect(input_text)
        return render_template('index.html', result=detection_result)
    else:
        return redirect(url_for('home'))

if __name__ == "__main__":
    app.run(debug=True)
