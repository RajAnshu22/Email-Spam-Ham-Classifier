import numpy as np
import pandas as pd
import nltk
import re
import gdown
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import gradio as gr
import io

# Download necessary nltk resources
nltk.download('stopwords')
nltk.download('wordnet')

# Step 1: Download dataset from Google Drive using gdown
url = 'https://drive.google.com/uc?id=1xu6FOMPERE0uflcAfjXDxV6W45IYOmT3'
gdown.download(url, 'spam_ham.csv', quiet=False)

# Step 2: Load the dataset
data = pd.read_csv('spam_ham.csv')

# Step 3: Identify and rename text column to 'EmailText'
text_column = None
for col in data.columns:
    if data[col].dtype == object:
        avg_text_len = data[col].apply(lambda x: len(str(x)) if isinstance(x, str) else 0).mean()
        if avg_text_len > 20:
            text_column = col
            break

if text_column:
    data.rename(columns={text_column: 'EmailText'}, inplace=True)
else:
    raise ValueError("No suitable text column found.")

# Preprocess function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\d', '', text)
    tokens = text.split()
    stop_words = set(nltk.corpus.stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = nltk.stem.WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

data['ProcessedText'] = data['EmailText'].apply(preprocess_text)

# Vectorize the 'ProcessedText' using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(data['ProcessedText']).toarray()

# Target variable (assuming 'spam' column contains spam/ham labels)
y = data['spam']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models
nb_model = MultinomialNB().fit(X_train, y_train)
svm_model = SVC(kernel='linear').fit(X_train, y_train)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)

# Define classification function for single email
def classify_text_input(text):
    processed_text = preprocess_text(text)
    vectorized_text = vectorizer.transform([processed_text]).toarray()
    predictions = {
        "Naive Bayes": "Spam" if nb_model.predict(vectorized_text)[0] == 1 else "Ham",
        "SVM": "Spam" if svm_model.predict(vectorized_text)[0] == 1 else "Ham",
        "Random Forest": "Spam" if rf_model.predict(vectorized_text)[0] == 1 else "Ham",
    }
    return predictions

# Define classification function for multiple emails
def classify_csv_file(file):
    # Read the uploaded CSV file
    data = pd.read_csv(file.name)

    # Check and rename the text column to 'EmailText' if it's not present
    if 'EmailText' not in data.columns:
        potential_text_columns = {
            col: data[col].apply(lambda x: len(str(x)) if isinstance(x, str) else 0).mean()
            for col in data.columns
        }
        text_column = max(potential_text_columns, key=potential_text_columns.get)
        data.rename(columns={text_column: 'EmailText'}, inplace=True)

    if 'EmailText' not in data.columns:
        return "Could not identify a suitable text column."

    # Preprocess and classify
    data['ProcessedText'] = data['EmailText'].apply(preprocess_text)
    vectorized_data = vectorizer.transform(data['ProcessedText']).toarray()
    nb_pred = nb_model.predict(vectorized_data)
    svm_pred = svm_model.predict(vectorized_data)
    rf_pred = rf_model.predict(vectorized_data)

    # Majority vote
    data['NaiveBayesPrediction'] = nb_pred
    data['SVMPrediction'] = svm_pred
    data['RandomForestPrediction'] = rf_pred
    data['FinalPrediction'] = data[['NaiveBayesPrediction', 'SVMPrediction', 'RandomForestPrediction']].mode(axis=1)[0]

    # Select relevant columns for the output
    output_data = data[['EmailText', 'NaiveBayesPrediction', 'SVMPrediction', 'RandomForestPrediction', 'FinalPrediction']]

    # Convert DataFrame to a list of dictionaries for JSON output
    return output_data.to_dict(orient='records')

# Gradio Interface
import gradio as gr

# Gradio Interface with Enhanced Title Styling and Layout
interface = gr.Interface(
    fn=lambda text, file: classify_text_input(text) if text else classify_csv_file(file),
    inputs=[
        gr.Textbox(label="Enter Email Text (leave blank for file upload)", placeholder="Type your email content here..."),
        gr.File(label="Upload CSV File (leave blank for single email text)")
    ],
    outputs="json",
    title="Spam/Ham Email Classifier",  # We'll override this title styling with CSS
    description="Classify emails as Spam (1) or Ham (0) in 3 different models.<br><br>"
                "Let's Go 🔥",
    theme="compact",  # Keep the compact theme for better spacing
    css="""
        body {
            font-family: 'Roboto', sans-serif;
            background-color: #1f1f1f;  /* Dark Background */
            color: #e0e0e0;  /* Light text for readability */
        }
        h1 {
            font-family: 'Nave', sans-serif;  /* Using Nave Bold Italic font */
            font-weight: bold;
            font-style: italic;
            font-size: 36px;
            text-align: center;
            background: linear-gradient(90deg, #fc6601 33%, #682bd7 66%, #ffffff 100%);  /* Gradient effect for title */
            -webkit-background-clip: text;  /* Clip the gradient background to text */
            color: transparent;  /* Make the text transparent to show the gradient */
        }
        .container {
            border: 2px solid #3a3a3a;  /* Slightly lighter border for contrast */
            border-radius: 10px;
            background-color: #2c2c2c;  /* Dark background for container */
            padding: 20px;
            box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.5);  /* Subtle shadow for depth */
        }
        .gr-button {
            background-color: #896deb;  /* Purple button */
            color: #fefae0;  /* Off-white text */
            border: 1px solid #896deb;
            border-radius: 5px;
            padding: 10px 20px;
            font-weight: bold;
        }
        .gr-button:hover {
            background-color: #7a4bb6;  /* Darker shade of purple on hover */
            color: white;
        }
        .gr-textbox {
            background-color: #333333;  /* Dark textbox background */
            color: #e0e0e0;  /* Light text in textboxes */
            border: 1px solid #444444;  /* Slightly lighter border */
        }
        .gr-textbox:focus {
            border-color: #896deb;  /* Purple border on focus */
        }
        .gr-file {
            background-color: #333333;
            color: #e0e0e0;
            border: 1px solid #444444;
        }
        .gr-file:focus {
            border-color: #896deb;
        }
        .description {
            font-size: 18px;
            font-style: italic;
            color: #b3b3b3;
        }
    """
)

interface.launch()