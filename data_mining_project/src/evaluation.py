#!/usr/bin/env python3
"""
Modul pentru evaluare si comparatie a algoritmilor
"""

import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_fscore_support
import numpy as np


def calculate_detailed_metrics(y_true, y_pred, labels):
    """
    Calculeaza metrici detaliate pentru fiecare clasa
    
    Args:
        y_true: Etichete reale
        y_pred: Predictii
        labels: Lista de etichete
        
    Returns:
        Dictionar cu metrici
    """
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    
    # Metrici macro (media aritmetica)
    macro_precision = np.mean(precision)
    macro_recall = np.mean(recall)
    macro_f1 = np.mean(f1)
    
    # Metrici micro (calculat pe toate predictiile)
    micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average='micro', zero_division=0
    )
    
    # Metrici weighted (ponderat dupa support)
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average='weighted', zero_division=0
    )
    
    metrics = {
        'per_class': {
            'precision': precision.tolist(),
            'recall': recall.tolist(),
            'f1': f1.tolist(),
            'support': support.tolist()
        },
        'macro': {
            'precision': float(macro_precision),
            'recall': float(macro_recall),
            'f1': float(macro_f1)
        },
        'micro': {
            'precision': float(micro_precision),
            'recall': float(micro_recall),
            'f1': float(micro_f1)
        },
        'weighted': {
            'precision': float(weighted_precision),
            'recall': float(weighted_recall),
            'f1': float(weighted_f1)
        }
    }
    
    return metrics


def compare_algorithms(results_list, output_dir='results'):
    """
    Compara rezultatele mai multor algoritmi
    
    Args:
        results_list: Lista de dictionare cu rezultate pentru fiecare algoritm
        output_dir: Directorul pentru salvare rezultate
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Creeaza DataFrame pentru comparatie
    comparison_data = []
    
    for result in results_list:
        algorithm = result['algorithm']
        accuracy = result['accuracy']
        training_time = result['training_time']
        prediction_time = result['prediction_time']
        
        # Extrage metrici din classification report
        report = result['classification_report']
        macro_avg = report.get('macro avg', {})
        weighted_avg = report.get('weighted avg', {})
        
        comparison_data.append({
            'Algorithm': algorithm,
            'Accuracy': accuracy,
            'Precision (Macro)': macro_avg.get('precision', 0),
            'Recall (Macro)': macro_avg.get('recall', 0),
            'F1-Score (Macro)': macro_avg.get('f1-score', 0),
            'Precision (Weighted)': weighted_avg.get('precision', 0),
            'Recall (Weighted)': weighted_avg.get('recall', 0),
            'F1-Score (Weighted)': weighted_avg.get('f1-score', 0),
            'Training Time (s)': training_time,
            'Prediction Time (s)': prediction_time
        })
    
    df_comparison = pd.DataFrame(comparison_data)
    
    # Salveaza comparatia
    csv_path = os.path.join(output_dir, 'algorithm_comparison.csv')
    df_comparison.to_csv(csv_path, index=False)
    print(f"\nComparatie salvata: {csv_path}")
    
    # Afiseaza tabel
    print("\n" + "="*80)
    print("COMPARATIE ALGORITMI")
    print("="*80)
    print(df_comparison.to_string(index=False))
    print("="*80)
    
    # Creeaza grafice
    create_comparison_plots(df_comparison, output_dir)
    
    return df_comparison


def create_comparison_plots(df_comparison, output_dir):
    """
    Creeaza grafice pentru comparatie
    
    Args:
        df_comparison: DataFrame cu comparatia
        output_dir: Directorul pentru salvare
    """
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Accuracy
    axes[0, 0].bar(df_comparison['Algorithm'], df_comparison['Accuracy'], color='skyblue')
    axes[0, 0].set_title('Accuracy Comparison', fontsize=14, fontweight='bold')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # F1-Score (Macro)
    axes[0, 1].bar(df_comparison['Algorithm'], df_comparison['F1-Score (Macro)'], color='lightgreen')
    axes[0, 1].set_title('F1-Score (Macro) Comparison', fontsize=14, fontweight='bold')
    axes[0, 1].set_ylabel('F1-Score')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # Training Time
    axes[1, 0].bar(df_comparison['Algorithm'], df_comparison['Training Time (s)'], color='coral')
    axes[1, 0].set_title('Training Time Comparison', fontsize=14, fontweight='bold')
    axes[1, 0].set_ylabel('Time (seconds)')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # Prediction Time
    axes[1, 1].bar(df_comparison['Algorithm'], df_comparison['Prediction Time (s)'], color='plum')
    axes[1, 1].set_title('Prediction Time Comparison', fontsize=14, fontweight='bold')
    axes[1, 1].set_ylabel('Time (seconds)')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, 'algorithm_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Grafic comparatie salvat: {plot_path}")
    plt.close()


def plot_confusion_matrices(results_list, category_names, output_dir='results'):
    """
    Creeaza confusion matrices pentru fiecare algoritm
    
    Args:
        results_list: Lista de rezultate
        category_names: Numele categoriilor
        output_dir: Directorul pentru salvare
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'confusion_matrices'), exist_ok=True)
    
    for result in results_list:
        algorithm = result['algorithm']
        cm = result['confusion_matrix']
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=category_names, yticklabels=category_names)
        plt.title(f'Confusion Matrix - {algorithm}', fontsize=14, fontweight='bold')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        plot_path = os.path.join(output_dir, 'confusion_matrices', f'{algorithm.lower().replace(" ", "_")}_confusion_matrix.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Confusion matrices salvate in: {os.path.join(output_dir, 'confusion_matrices')}")


def save_detailed_results(results_list, category_names, output_dir='results'):
    """
    Salveaza rezultate detaliate in JSON
    
    Args:
        results_list: Lista de rezultate
        category_names: Numele categoriilor
        output_dir: Directorul pentru salvare
    """
    os.makedirs(output_dir, exist_ok=True)
    
    detailed_results = []
    
    for result in results_list:
        algorithm = result['algorithm']
        y_test = result.get('y_test', [])
        y_pred = result['predictions']
        
        # Calculeaza metrici detaliate
        labels = list(range(len(category_names)))
        detailed_metrics = calculate_detailed_metrics(y_test, y_pred, labels)
        
        result_dict = {
            'algorithm': algorithm,
            'accuracy': float(result['accuracy']),
            'training_time': float(result['training_time']),
            'prediction_time': float(result['prediction_time']),
            'metrics': detailed_metrics,
            'category_names': category_names
        }
        
        detailed_results.append(result_dict)
    
    # Salveaza JSON
    json_path = os.path.join(output_dir, 'detailed_results.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(detailed_results, f, indent=2, ensure_ascii=False)
    
    print(f"Rezultate detaliate salvate: {json_path}")

