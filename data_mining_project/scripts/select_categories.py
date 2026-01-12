#!/usr/bin/env python3
"""
Script pentru selectarea unui subset de categorii din 20 Newsgroups
Permite alegerea a 5-6 categorii pentru proiect
"""

import os
import pandas as pd
import json


def select_categories(category_names, output_filename='selected_dataset.csv'):
    """
    Selecteaza anumite categorii din dataset si creeaza un subset
    
    Args:
        category_names: Lista cu numele categoriilor de selectat
        output_filename: Numele fisierului de output
    """
    # Citeste dataset-ul complet
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
    input_file = os.path.join(data_dir, '20newsgroups_dataset.csv')
    
    if not os.path.exists(input_file):
        print(f"Eroare: Fisierul {input_file} nu exista!")
        print("   Ruleaza mai intai: python scripts/download_20newsgroups.py")
        return
    
    print(f"Citire dataset din {input_file}...")
    df = pd.read_csv(input_file)
    
    print(f"\nDataset original: {len(df)} documente, {df['category_name'].nunique()} categorii")
    
    # Filtreaza doar categoriile selectate
    selected_df = df[df['category_name'].isin(category_names)].copy()
    
    if len(selected_df) == 0:
        print(f"Eroare: Nu s-au gasit documente pentru categoriile: {category_names}")
        print("\nCategorii disponibile:")
        for cat in sorted(df['category_name'].unique()):
            print(f"  - {cat}")
        return
    
    # Reindexeaza categoriile (0, 1, 2, ...)
    category_mapping = {name: idx for idx, name in enumerate(sorted(selected_df['category_name'].unique()))}
    selected_df['category_id'] = selected_df['category_name'].map(category_mapping)
    
    print(f"\nDataset selectat: {len(selected_df)} documente, {len(category_mapping)} categorii")
    print("\nCategorii selectate:")
    for name, idx in sorted(category_mapping.items(), key=lambda x: x[1]):
        count = len(selected_df[selected_df['category_name'] == name])
        print(f"  {idx}: {name} ({count} documente)")
    
    # Salveaza dataset-ul selectat
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, output_filename)
    selected_df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"\nDataset salvat: {output_path}")
    
    # Salveaza mapping-ul categoriilor
    mapping_path = os.path.join(output_dir, 'category_mapping.json')
    with open(mapping_path, 'w', encoding='utf-8') as f:
        json.dump(category_mapping, f, indent=2, ensure_ascii=False)
    print(f"Mapping categorii: {mapping_path}")
    
    return selected_df


if __name__ == "__main__":
    # EXEMPLU: Selecteaza 6 categorii diverse
    # Poti modifica aceste categorii dupa preferinte
    
    selected_categories = [
        'sci.space',           # Stiinta - spatiu
        'rec.sport.hockey',    # Sport
        'comp.graphics',       # Tehnologie
        'talk.politics.mideast',  # Politica
        'rec.autos',           # Auto
        'sci.med'              # Medicina
    ]
    
    print("=" * 60)
    print("SELECTARE CATEGORII DIN 20 NEWSCROUPS")
    print("=" * 60)
    print("\nModifica lista 'selected_categories' in script pentru a alege alte categorii")
    print(f"\nCategorii selectate: {len(selected_categories)}")
    
    select_categories(selected_categories, 'selected_6categories_dataset.csv')

