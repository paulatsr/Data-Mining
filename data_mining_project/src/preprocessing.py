#!/usr/bin/env python3
"""
Modul pentru preprocesarea textului
Include: tokenizare, eliminare stop words, stemming, vectorizare
"""

import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split

# Descarca resursele NLTK necesare (doar prima data)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)


class TextPreprocessor:
    """Clasă pentru preprocesarea textului"""
    
    def __init__(self, use_stemming=True, use_stopwords=True, language='english'):
        """
        Initializeaza preprocesorul
        
        Args:
            use_stemming: Daca sa foloseasca stemming
            use_stopwords: Daca sa elimine stop words
            language: Limba pentru stop words
        """
        self.use_stemming = use_stemming
        self.use_stopwords = use_stopwords
        self.stemmer = PorterStemmer() if use_stemming else None
        self.stop_words = set(stopwords.words(language)) if use_stopwords else set()
    
    def clean_text(self, text):
        """
        Curata textul: elimina caractere speciale, lowercase, etc.
        
        Args:
            text: Textul de curatat
            
        Returns:
            Text curatat
        """
        if pd.isna(text) or text == '':
            return ''
        
        # Convert la string dacă nu este
        text = str(text)
        
        # Convert la lowercase
        text = text.lower()
        
        # Elimina caractere speciale, pastreaza doar litere, cifre si spatii
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        
        # Elimina spatii multiple
        text = re.sub(r'\s+', ' ', text)
        
        # Elimina spatii de la inceput si sfarsit
        text = text.strip()
        
        return text
    
    def tokenize(self, text):
        """
        Tokenizeaza textul
        
        Args:
            text: Textul de tokenizat
            
        Returns:
            Lista de tokeni
        """
        text = self.clean_text(text)
        if not text:
            return []
        
        # Tokenizare simpla (poti folosi si nltk.word_tokenize)
        tokens = text.split()
        
        # Elimina stop words
        if self.use_stopwords:
            tokens = [token for token in tokens if token not in self.stop_words]
        
        # Stemming
        if self.use_stemming and self.stemmer:
            tokens = [self.stemmer.stem(token) for token in tokens]
        
        return tokens
    
    def preprocess_text(self, text):
        """
        Preproceseaza un text complet
        
        Args:
            text: Textul de preprocesat
            
        Returns:
            Text preprocesat (string)
        """
        tokens = self.tokenize(text)
        return ' '.join(tokens)
    
    def preprocess_dataframe(self, df, text_column='text'):
        """
        Preproceseaza un DataFrame intreg
        
        Args:
            df: DataFrame cu textul
            text_column: Numele coloanei cu textul
            
        Returns:
            DataFrame cu textul preprocesat
        """
        df_processed = df.copy()
        df_processed[f'{text_column}_processed'] = df_processed[text_column].apply(
            self.preprocess_text
        )
        return df_processed


def prepare_data_for_training(df, text_column='text', label_column='category_id', 
                              test_size=0.2, random_state=42, vectorizer_type='tfidf',
                              max_features=10000):
    """
    Pregateste datele pentru antrenare: preprocesare si vectorizare
    
    Args:
        df: DataFrame cu datele
        text_column: Coloana cu textul
        label_column: Coloana cu etichetele
        test_size: Procentaj pentru test set
        random_state: Seed pentru reproducibilitate
        vectorizer_type: 'tfidf' sau 'count'
        max_features: Numarul maxim de features pentru vectorizare
        
    Returns:
        X_train, X_test, y_train, y_test, vectorizer
    """
    # Preprocesare
    preprocessor = TextPreprocessor(use_stemming=True, use_stopwords=True)
    df_processed = preprocessor.preprocess_dataframe(df, text_column)
    
    # Vectorizare
    if vectorizer_type == 'tfidf':
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),  # unigrams si bigrams
            min_df=2,  # minim 2 aparitii
            max_df=0.95  # maxim 95% din documente
        )
    else:
        vectorizer = CountVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )
    
    # Vectorizare text
    X = vectorizer.fit_transform(df_processed[f'{text_column}_processed'])
    y = df_processed[label_column].values
    
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"Date preprocesate:")
    print(f"   Train: {X_train.shape[0]} documente, {X_train.shape[1]} features")
    print(f"   Test: {X_test.shape[0]} documente")
    print(f"   Categorii: {len(set(y))}")
    
    return X_train, X_test, y_train, y_test, vectorizer


if __name__ == "__main__":
    # Test preprocesare
    sample_text = "This is a sample text for testing! It contains some words and punctuation."
    
    preprocessor = TextPreprocessor()
    processed = preprocessor.preprocess_text(sample_text)
    
    print("Text original:", sample_text)
    print("Text preprocesat:", processed)

