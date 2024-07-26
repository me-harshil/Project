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
    """
    Preprocess the input text by performing the following steps:
    1. Convert to lowercase
    2. Remove punctuation
    3. Tokenize the text
    4. Remove stop words
    5. Apply stemming and lemmatization
    6. Remove numbers
    7. Perform POS tagging and NER using SpaCy

    Args:
    text (str): Input text to preprocess.

    Returns:
    dict: Dictionary containing cleaned text, tokens, POS tags, and named entities.
    """
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
    # Load dataset
    df = pd.read_csv(file_path)

    # Apply preprocessing to each sentence in the dataset
    df['preprocessed'] = df['sentence'].apply(preprocess_text)

    return df


def main():
    # Define the path to the input CSV file
    input_file_path = 'data/sentiment140.csv'

    # Load and preprocess data
    data = load_and_preprocess_data(input_file_path)

    # Define the path to the output CSV file
    output_file_path = 'data/preprocessed_sentiment_data.csv'

    # Save preprocessed data to a new CSV file
    data.to_csv(output_file_path, index=False)
    print(f"Preprocessing completed and saved to {output_file_path}")


# Entry point for script execution
if __name__ == "__main__":
    main()
