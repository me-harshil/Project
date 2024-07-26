import string
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
import spacy

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize SpaCy model for NLP tasks such as POS tagging and NER
nlp = spacy.load("en_core_web_sm")


def preprocess_text(text):
    # Step 1: Convert to lowercase
    text = text.lower()

    # Step 2: Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Step 3: Tokenization
    tokens = word_tokenize(text)

    # Step 4: Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    # Step 5: Stemming and Lemmatization
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(stemmer.stem(word)) for word in tokens]

    # Step 6: Remove numbers
    tokens = [word for word in tokens if not word.isdigit()]

    # Join tokens back into a string
    cleaned_text = ' '.join(tokens)

    # Step 7: Part-of-Speech Tagging and Named Entity Recognition using SpaCy
    doc = nlp(cleaned_text)
    pos_tags = [(token.text, token.pos_) for token in doc]
    named_entities = [(ent.text, ent.label_) for ent in doc.ents]

    return {
        'cleaned_text': cleaned_text,
        'tokens': tokens,
        'pos_tags': pos_tags,
        'named_entities': named_entities
    }


def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)

    df['preprocessed'] = df['sentence'].apply(preprocess_text)

    return df


def main():
    input_file_path = 'data/sentiment140.csv'

    data = load_and_preprocess_data(input_file_path)

    output_file_path = 'data/preprocessed_sentiment_data.csv'

    data.to_csv(output_file_path, index=False)
    print(f"Preprocessing completed and saved to {output_file_path}")

if __name__ == "__main__":
    main()
