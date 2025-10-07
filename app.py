import streamlit as st
import numpy as np
import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# =======================
# Ensure required NLTK data is downloaded
# =======================
required_packages = ['punkt', 'stopwords']
for pkg in required_packages:
    try:
        nltk.data.find(f'tokenizers/{pkg}' if pkg == 'punkt' else f'corpora/{pkg}')
    except LookupError:
        nltk.download(pkg)

# =======================
# Load and preprocess data
# =======================
news_df = pd.read_csv('train.csv')
news_df = news_df.fillna(' ')
news_df['content'] = news_df['author'] + ' ' + news_df['title']

# Stemming function
ps = PorterStemmer()
def stemming(content):
    # Remove non-alphabetic characters
    content = re.sub('[^a-zA-Z]', ' ', content)
    # Convert to lowercase and split
    words = content.lower().split()
    # Remove stopwords and stem
    words = [ps.stem(word) for word in words if word not in stopwords.words('english')]
    return ' '.join(words)

news_df['content'] = news_df['content'].apply(stemming)

# Vectorize
X = news_df['content'].values
y = news_df['label'].values
vector = TfidfVectorizer()
X = vector.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=2
)

# Fit model
model = LogisticRegression()
model.fit(X_train, y_train)

# =======================
# Streamlit Web App
# =======================
st.title('Fake News Detection')

input_text = st.text_input('Enter news article to be verified:')

def predict_news(text):
    input_vec = vector.transform([text])
    pred = model.predict(input_vec)[0]
    return 'Fake' if pred == 1 else 'Real'

if input_text:
    result = predict_news(input_text)
    st.write(f'The news is: **{result}**')
