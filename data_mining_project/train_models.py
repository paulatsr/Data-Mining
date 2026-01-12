#!/usr/bin/env python3
"""
Script pentru antrenarea si salvarea modelelor pentru UI
"""

import os
import sys
import json
import pickle
import pandas as pd
from datetime import datetime

# Adauga directorul src la path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from preprocessing import prepare_data_for_training
from naive_bayes import NaiveBayesClassifier
from svm_classifier import SVMClassifier
from random_forest import RandomForestTextClassifier


def train_and_save_models():
    """Antreneaza si salveaza modelele"""
    print("="*80)
    print("ANTRENARE MODELE PENTRU UI")
    print("="*80)
    
    base_dir = os.path.dirname(__file__)
    dataset_path = os.path.join(base_dir, 'data', 'processed', 'selected_6categories_dataset.csv')
    models_dir = os.path.join(base_dir, 'models')
    
    # Verifica dataset
    if not os.path.exists(dataset_path):
        print(f"Eroare: Dataset-ul nu exista la {dataset_path}")
        print("\nRuleaza mai intai:")
        print("   python3 scripts/select_categories.py")
        return False
    
    # Creeaza directorul pentru modele
    os.makedirs(models_dir, exist_ok=True)
    
    # Incarca dataset
    print(f"\nIncarcare dataset...")
    df = pd.read_csv(dataset_path)
    print(f"Dataset incarcat: {len(df)} documente")
    
    # Pregateste datele
    print("\nPreprocesare date...")
    X_train, X_test, y_train, y_test, vectorizer = prepare_data_for_training(
        df,
        text_column='text',
        label_column='category_id',
        test_size=0.2,
        random_state=42,
        vectorizer_type='tfidf',
        max_features=10000
    )
    
    # Antreneaza modelele
    print("\n" + "="*80)
    print("ANTRENARE ALGORITMI")
    print("="*80)
    
    # 1. Naive Bayes
    print("\n1. Naive Bayes...")
    nb_classifier = NaiveBayesClassifier(alpha=1.0)
    nb_classifier.train(X_train, y_train)
    nb_accuracy = nb_classifier.model.score(X_test, y_test)
    print(f"   Accuracy: {nb_accuracy:.4f}")
    
    # 2. SVM
    print("\n2. SVM...")
    svm_classifier = SVMClassifier(kernel='linear', C=1.0, max_iter=2000)
    svm_classifier.train(X_train, y_train)
    svm_accuracy = svm_classifier.model.score(X_test, y_test)
    print(f"   Accuracy: {svm_accuracy:.4f}")
    
    # 3. Random Forest
    print("\n3. Random Forest...")
    rf_classifier = RandomForestTextClassifier(n_estimators=100, max_depth=None, random_state=42)
    rf_classifier.train(X_train, y_train)
    rf_accuracy = rf_classifier.model.score(X_test, y_test)
    print(f"   Accuracy: {rf_accuracy:.4f}")
    
    # Salveaza modelele
    print("\n" + "="*80)
    print("SALVARE MODELE")
    print("="*80)
    
    # Salveaza vectorizer
    vectorizer_path = os.path.join(models_dir, 'vectorizer.pkl')
    with open(vectorizer_path, 'wb') as f:
        pickle.dump(vectorizer, f)
    print(f"Vectorizer salvat: {vectorizer_path}")
    
    # Salveaza Naive Bayes
    nb_path = os.path.join(models_dir, 'naive_bayes.pkl')
    with open(nb_path, 'wb') as f:
        pickle.dump(nb_classifier.model, f)
    print(f"Naive Bayes salvat: {nb_path}")
    
    # Salveaza SVM
    svm_path = os.path.join(models_dir, 'svm.pkl')
    with open(svm_path, 'wb') as f:
        pickle.dump(svm_classifier.model, f)
    print(f"SVM salvat: {svm_path}")
    
    # Salveaza Random Forest
    rf_path = os.path.join(models_dir, 'random_forest.pkl')
    with open(rf_path, 'wb') as f:
        pickle.dump(rf_classifier.model, f)
    print(f"Random Forest salvat: {rf_path}")
    
    # Salveaza informatii despre antrenare
    training_info = {
        'training_date': datetime.now().isoformat(),
        'dataset': {
            'name': '20 Newsgroups (Selected Categories)',
            'path': dataset_path,
            'total_documents': len(df),
            'train_documents': X_train.shape[0],
            'test_documents': X_test.shape[0],
            'categories': sorted(df['category_name'].unique().tolist()),
            'num_categories': len(df['category_name'].unique()),
            'features': X_train.shape[1]
        },
        'preprocessing': {
            'vectorizer_type': 'TF-IDF',
            'max_features': 10000,
            'ngram_range': '(1, 2)',
            'use_stemming': True,
            'use_stopwords': True
        },
        'algorithms': {
            'naive_bayes': {
                'name': 'Naive Bayes (MultinomialNB)',
                'parameters': {'alpha': 1.0},
                'training_time': float(nb_classifier.training_time),
                'training_time_formatted': f"{nb_classifier.training_time:.6f}s",
                'accuracy': float(nb_accuracy),
                'test_accuracy': float(nb_classifier.model.score(X_test, y_test))
            },
            'svm': {
                'name': 'Support Vector Machine (SVM)',
                'parameters': {'kernel': 'linear', 'C': 1.0, 'max_iter': 2000},
                'training_time': float(svm_classifier.training_time),
                'training_time_formatted': f"{svm_classifier.training_time:.6f}s",
                'accuracy': float(svm_accuracy),
                'test_accuracy': float(svm_classifier.model.score(X_test, y_test))
            },
            'random_forest': {
                'name': 'Random Forest',
                'parameters': {'n_estimators': 100, 'max_depth': None, 'random_state': 42},
                'training_time': float(rf_classifier.training_time),
                'training_time_formatted': f"{rf_classifier.training_time:.6f}s",
                'accuracy': float(rf_accuracy),
                'test_accuracy': float(rf_classifier.model.score(X_test, y_test))
            }
        },
        'summary': {
            'best_accuracy_algorithm': 'naive_bayes' if nb_accuracy >= max(svm_accuracy, rf_accuracy) else ('svm' if svm_accuracy >= rf_accuracy else 'random_forest'),
            'fastest_training_algorithm': 'naive_bayes' if nb_classifier.training_time <= min(svm_classifier.training_time, rf_classifier.training_time) else ('svm' if svm_classifier.training_time <= rf_classifier.training_time else 'random_forest')
        }
    }
    
    info_path = os.path.join(models_dir, 'training_info.json')
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(training_info, f, indent=2, ensure_ascii=False)
    print(f"Informatii antrenare salvate: {info_path}")
    
    print("\n" + "="*80)
    print("MODELE ANTRENATE SI SALVATE CU SUCCES!")
    print("="*80)
    print(f"\nModele salvate in: {models_dir}")
    print("\nAcum poti rula UI-ul:")
    print("   python3 app.py")
    
    return True


if __name__ == "__main__":
    train_and_save_models()

