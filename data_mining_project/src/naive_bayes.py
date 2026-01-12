#!/usr/bin/env python3
"""
Implementare Naive Bayes pentru clasificare text
"""

import time
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd


class NaiveBayesClassifier:
    """Clasificator Naive Bayes pentru text"""
    
    def __init__(self, alpha=1.0):
        """
        Initializeaza clasificatorul
        
        Args:
            alpha: Parametru de smoothing (Laplace smoothing)
        """
        self.model = MultinomialNB(alpha=alpha)
        self.training_time = None
        self.prediction_time = None
    
    def train(self, X_train, y_train):
        """
        Antreneaza modelul
        
        Args:
            X_train: Features de antrenare
            y_train: Etichete de antrenare
            
        Returns:
            self
        """
        print("Antrenare Naive Bayes...")
        start_time = time.time()
        
        self.model.fit(X_train, y_train)
        
        self.training_time = time.time() - start_time
        print(f"Antrenare completa in {self.training_time:.6f} secunde")
        
        return self
    
    def predict(self, X_test):
        """
        Face predictii
        
        Args:
            X_test: Features de test
            
        Returns:
            Predictii
        """
        start_time = time.time()
        predictions = self.model.predict(X_test)
        self.prediction_time = time.time() - start_time
        
        return predictions
    
    def evaluate(self, X_test, y_test):
        """
        Evalueaza modelul
        
        Args:
            X_test: Features de test
            y_test: Etichete reale
            
        Returns:
            Dictionar cu metrici
        """
        predictions = self.predict(X_test)
        
        accuracy = accuracy_score(y_test, predictions)
        
        metrics = {
            'algorithm': 'Naive Bayes',
            'accuracy': accuracy,
            'training_time': self.training_time,
            'prediction_time': self.prediction_time,
            'predictions': predictions,
            'classification_report': classification_report(y_test, predictions, output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, predictions)
        }
        
        print(f"\nRezultate Naive Bayes:")
        print(f"   Accuracy: {accuracy:.4f}")
        print(f"   Timp antrenare: {self.training_time:.6f}s")
        print(f"   Timp predictie: {self.prediction_time:.6f}s")
        
        return metrics

