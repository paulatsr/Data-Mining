#!/usr/bin/env python3
"""
Script principal pentru proiectul de data mining
Antreneaza 3 algoritmi si compara rezultatele
"""

import os
import sys
import json
import pandas as pd

# Adauga directorul src la path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from preprocessing import prepare_data_for_training
from naive_bayes import NaiveBayesClassifier
from svm_classifier import SVMClassifier
from random_forest import RandomForestTextClassifier
from evaluation import compare_algorithms, plot_confusion_matrices, save_detailed_results


def load_dataset(dataset_path):
    """
    Incarca dataset-ul
    
    Args:
        dataset_path: Calea catre fisierul CSV
        
    Returns:
        DataFrame cu datele
    """
    print(f"Incarcare dataset din {dataset_path}...")
    df = pd.read_csv(dataset_path)
    
    print(f"Dataset incarcat: {len(df)} documente, {df['category_name'].nunique()} categorii")
    print(f"\nDistributie categorii:")
    category_counts = df['category_name'].value_counts()
    for category, count in category_counts.items():
        print(f"   {category}: {count} documente")
    
    return df


def main():
    """Functie principala"""
    print("="*80)
    print("PROIECT DATA MINING - CLASIFICARE TEXT")
    print("="*80)
    print("\nAlgoritmi: Naive Bayes, SVM, Random Forest")
    print("Dataset: 20 Newsgroups (subset selectat)\n")
    
    # Cai catre fisiere
    base_dir = os.path.dirname(__file__)
    dataset_path = os.path.join(base_dir, 'data', 'processed', 'selected_6categories_dataset.csv')
    mapping_path = os.path.join(base_dir, 'data', 'processed', 'category_mapping.json')
    results_dir = os.path.join(base_dir, 'results')
    
    # Verifica daca dataset-ul exista
    if not os.path.exists(dataset_path):
        print(f"Eroare: Dataset-ul nu exista la {dataset_path}")
        print("\nRuleaza mai intai:")
        print("   python3 scripts/select_categories.py")
        return
    
    # Incarca dataset
    df = load_dataset(dataset_path)
    
    # Incarca mapping-ul categoriilor
    if os.path.exists(mapping_path):
        with open(mapping_path, 'r', encoding='utf-8') as f:
            category_mapping = json.load(f)
        # Inverseaza mapping-ul pentru a obtine numele categoriilor
        id_to_name = {v: k for k, v in category_mapping.items()}
        category_names = [id_to_name[i] for i in sorted(id_to_name.keys())]
    else:
        # Daca nu exista mapping, foloseste categoriile din dataset
        category_names = sorted(df['category_name'].unique().tolist())
    
    print(f"\nCategorii: {', '.join(category_names)}")
    
    # Pregateste datele pentru antrenare
    print("\n" + "="*80)
    print("PREPROCESARE DATE")
    print("="*80)
    X_train, X_test, y_train, y_test, vectorizer = prepare_data_for_training(
        df, 
        text_column='text',
        label_column='category_id',
        test_size=0.2,
        random_state=42,
        vectorizer_type='tfidf',
        max_features=10000
    )
    
    # Lista pentru rezultate
    results_list = []
    
    # 1. Naive Bayes
    print("\n" + "="*80)
    print("ALGORITM 1: NAIVE BAYES")
    print("="*80)
    nb_classifier = NaiveBayesClassifier(alpha=1.0)
    nb_classifier.train(X_train, y_train)
    nb_results = nb_classifier.evaluate(X_test, y_test)
    nb_results['y_test'] = y_test  # Adauga y_test pentru evaluare detaliata
    results_list.append(nb_results)
    
    # 2. SVM
    print("\n" + "="*80)
    print("ALGORITM 2: SUPPORT VECTOR MACHINE (SVM)")
    print("="*80)
    svm_classifier = SVMClassifier(kernel='linear', C=1.0, max_iter=1000)
    svm_classifier.train(X_train, y_train)
    svm_results = svm_classifier.evaluate(X_test, y_test)
    svm_results['y_test'] = y_test
    results_list.append(svm_results)
    
    # 3. Random Forest
    print("\n" + "="*80)
    print("ALGORITM 3: RANDOM FOREST")
    print("="*80)
    rf_classifier = RandomForestTextClassifier(n_estimators=100, max_depth=None, random_state=42)
    rf_classifier.train(X_train, y_train)
    rf_results = rf_classifier.evaluate(X_test, y_test)
    rf_results['y_test'] = y_test
    results_list.append(rf_results)
    
    # Compara algoritmii
    print("\n" + "="*80)
    print("COMPARATIE SI EVALUARE")
    print("="*80)
    
    comparison_df = compare_algorithms(results_list, output_dir=results_dir)
    
    # Creeaza confusion matrices
    plot_confusion_matrices(results_list, category_names, output_dir=results_dir)
    
    # Salveaza rezultate detaliate
    save_detailed_results(results_list, category_names, output_dir=results_dir)
    
    print("\n" + "="*80)
    print("PROIECT COMPLET!")
    print("="*80)
    print(f"\nRezultate salvate in: {results_dir}")
    print("   - algorithm_comparison.csv")
    print("   - algorithm_comparison.png")
    print("   - detailed_results.json")
    print("   - confusion_matrices/")
    
    # Concluzie
    print("\n" + "="*80)
    print("CONCLUZII")
    print("="*80)
    best_accuracy = comparison_df.loc[comparison_df['Accuracy'].idxmax()]
    fastest_training = comparison_df.loc[comparison_df['Training Time (s)'].idxmin()]
    best_f1 = comparison_df.loc[comparison_df['F1-Score (Macro)'].idxmax()]
    
    print(f"\nCel mai bun accuracy: {best_accuracy['Algorithm']} ({best_accuracy['Accuracy']:.4f})")
    print(f"Cel mai rapid antrenare: {fastest_training['Algorithm']} ({fastest_training['Training Time (s)']:.2f}s)")
    print(f"Cel mai bun F1-Score: {best_f1['Algorithm']} ({best_f1['F1-Score (Macro)']:.4f})")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()

