#!/usr/bin/env python3
"""
Script pentru descarcarea si exportul setului de date 20 Newsgroups
Exporta datele in format CSV si JSON pentru a le avea local
"""

import os
import json
import csv
from sklearn.datasets import fetch_20newsgroups
import pandas as pd


def download_and_export_20newsgroups():
    """
    Descarca setul de date 20 Newsgroups si il exporta in CSV si JSON
    """
    print("Descarcare set de date 20 Newsgroups...")
    
    # Descarca toate categoriile
    newsgroups_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
    newsgroups_test = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'))
    
    print(f"Date antrenare: {len(newsgroups_train.data)} documente")
    print(f"Date test: {len(newsgroups_test.data)} documente")
    
    # Combina datele de antrenare si test
    all_data = list(newsgroups_train.data) + list(newsgroups_test.data)
    all_targets = list(newsgroups_train.target) + list(newsgroups_test.target)
    all_target_names = newsgroups_train.target_names
    
    print(f"\nCategorii disponibile ({len(all_target_names)}):")
    for i, name in enumerate(all_target_names):
        count = all_targets.count(i)
        print(f"  {i+1}. {name}: {count} documente")
    
    # Creeaza DataFrame
    df = pd.DataFrame({
        'text': all_data,
        'category_id': all_targets,
        'category_name': [all_target_names[target] for target in all_targets]
    })
    
    # Elimina documentele goale
    df = df[df['text'].str.strip().str.len() > 0]
    
    print(f"\nTotal documente valide: {len(df)}")
    
    # Creeaza directoarele daca nu exista
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
    os.makedirs(output_dir, exist_ok=True)
    
    # Export CSV
    csv_path = os.path.join(output_dir, '20newsgroups_dataset.csv')
    df.to_csv(csv_path, index=False, encoding='utf-8')
    print(f"\nExportat CSV: {csv_path}")
    
    # Export JSON (format mai structurat)
    json_path = os.path.join(output_dir, '20newsgroups_dataset.json')
    df.to_json(json_path, orient='records', indent=2, force_ascii=False)
    print(f"Exportat JSON: {json_path}")
    
    # Export statistici
    stats = {
        'total_documents': len(df),
        'total_categories': len(all_target_names),
        'categories': {}
    }
    
    for category_id, category_name in enumerate(all_target_names):
        count = len(df[df['category_id'] == category_id])
        stats['categories'][category_name] = {
            'id': category_id,
            'count': count,
            'percentage': round((count / len(df)) * 100, 2)
        }
    
    stats_path = os.path.join(output_dir, 'dataset_statistics.json')
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"Statistici exportate: {stats_path}")
    
    # Export lista de categorii
    categories_path = os.path.join(output_dir, 'categories_list.txt')
    with open(categories_path, 'w', encoding='utf-8') as f:
        for i, name in enumerate(all_target_names):
            f.write(f"{i}: {name}\n")
    print(f"Lista categorii: {categories_path}")
    
    print("\nExport complet! Datele sunt in data/raw/")
    print("\nSfat: Poti selecta 5-6 categorii relevante pentru proiectul tau")
    
    return df, stats


if __name__ == "__main__":
    download_and_export_20newsgroups()

