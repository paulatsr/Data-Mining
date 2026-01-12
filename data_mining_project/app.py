#!/usr/bin/env python3
"""
Flask app pentru UI de clasificare text
"""

import os
import sys
import json
import pickle
import time
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import pandas as pd

# Import pentru citirea PDF-urilor
try:
    import PyPDF2
    PDF_SUPPORT = True
    PDF_LIBRARY = 'PyPDF2'
except ImportError:
    try:
        import pdfplumber
        PDF_SUPPORT = True
        PDF_LIBRARY = 'pdfplumber'
    except ImportError:
        PDF_SUPPORT = False
        PDF_LIBRARY = None

# Adauga directorul src la path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from preprocessing import TextPreprocessor, prepare_data_for_training
from naive_bayes import NaiveBayesClassifier
from svm_classifier import SVMClassifier
from random_forest import RandomForestTextClassifier

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'txt', 'csv', 'json', 'pdf'}

# Creeaza directorul pentru uploads
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Variabile globale pentru modele și vectorizer
models = {}
vectorizer = None
category_names = []


def allowed_file(filename):
    """Verifica daca fisierul este permis"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def extract_text_from_pdf(filepath):
    """
    Extrage textul dintr-un fisier PDF
    
    Args:
        filepath: Calea catre fisierul PDF
        
    Returns:
        Textul extras din PDF
    """
    text = ""
    
    try:
        if PDF_LIBRARY == 'PyPDF2':
            with open(filepath, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
        
        elif PDF_LIBRARY == 'pdfplumber':
            with pdfplumber.open(filepath) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        
        return text.strip()
    
    except Exception as e:
        raise Exception(f"Eroare la extragerea textului din PDF: {str(e)}")


def format_category_name(category):
    """
    Formateaza numele categoriei intr-un format prietenos
    
    Args:
        category: Numele categoriei (ex: 'rec.sport.hockey')
        
    Returns:
        Numele formatat (ex: 'Sport - Hockey')
    """
    # Mapping complet pentru categorii (prioritate pentru mapping-uri complete)
    full_category_map = {
        'alt.atheism': 'Ateism',
        'comp.graphics': 'Tehnologie - Grafică',
        'comp.os.ms-windows.misc': 'Tehnologie - Windows',
        'comp.sys.ibm.pc.hardware': 'Tehnologie - PC Hardware',
        'comp.sys.mac.hardware': 'Tehnologie - Mac Hardware',
        'comp.windows.x': 'Tehnologie - Windows X',
        'misc.forsale': 'Vânzări',
        'rec.autos': 'Auto',
        'rec.motorcycles': 'Motociclete',
        'rec.sport.baseball': 'Sport - Baseball',
        'rec.sport.hockey': 'Sport - Hockey',
        'sci.crypt': 'Stiinta - Criptografie',
        'sci.electronics': 'Stiinta - Electronica',
        'sci.med': 'Stiinta - Medicina',
        'sci.space': 'Stiinta - Spatiu',
        'soc.religion.christian': 'Religie - Crestinism',
        'talk.politics.guns': 'Politică - Arme',
        'talk.politics.mideast': 'Politică - Orientul Mijlociu',
        'talk.politics.misc': 'Politică - Diverse',
        'talk.religion.misc': 'Religie - Diverse'
    }
    
    # Verifică dacă există mapping complet
    if category in full_category_map:
        return full_category_map[category]
    
    # Dacă nu există mapping complet, formatează manual
    parts = category.split('.')
    
    # Mapping pentru prefixe principale
    prefix_map = {
        'alt': 'Alte',
        'comp': 'Tehnologie',
        'misc': 'Diverse',
        'rec': 'Recreere',
        'sci': 'Stiinta',
        'soc': 'Social',
        'talk': 'Discuții'
    }
    
    if len(parts) >= 2:
        prefix = parts[0]
        suffix_parts = parts[1:]
        
        # Formateaza prefixul
        prefix_name = prefix_map.get(prefix, prefix.capitalize())
        
        # Formateaza sufixul
        if len(suffix_parts) == 1:
            suffix = suffix_parts[0].replace('-', ' ').title()
            return f"{prefix_name} - {suffix}"
        else:
            # Pentru cazuri cu multiple subcategorii
            suffix = ' - '.join([p.replace('-', ' ').title() for p in suffix_parts])
            return f"{prefix_name} - {suffix}"
    
    # Daca nu se potriveste niciun pattern, returneaza formatat
    return category.replace('.', ' ').replace('-', ' ').title()


def load_models():
    """Incarca modelele antrenate"""
    global models, vectorizer, category_names
    
    base_dir = os.path.dirname(__file__)
    models_dir = os.path.join(base_dir, 'models')
    
    # Verifica daca modelele exista
    if not os.path.exists(models_dir):
        print("Modelele nu sunt antrenate. Antreneaza-le mai intai cu: python3 train_models.py")
        return False
    
    try:
        # Incarca vectorizer
        with open(os.path.join(models_dir, 'vectorizer.pkl'), 'rb') as f:
            vectorizer = pickle.load(f)
        
        # Incarca modelele
        with open(os.path.join(models_dir, 'naive_bayes.pkl'), 'rb') as f:
            models['naive_bayes'] = pickle.load(f)
        
        with open(os.path.join(models_dir, 'svm.pkl'), 'rb') as f:
            models['svm'] = pickle.load(f)
        
        with open(os.path.join(models_dir, 'random_forest.pkl'), 'rb') as f:
            models['random_forest'] = pickle.load(f)
        
        # Incarca numele categoriilor
        with open(os.path.join(base_dir, 'data', 'processed', 'category_mapping.json'), 'r', encoding='utf-8') as f:
            category_mapping = json.load(f)
            id_to_name = {v: k for k, v in category_mapping.items()}
            category_names = [id_to_name[i] for i in sorted(id_to_name.keys())]
        
        print("Modele incarcate cu succes!")
        return True
    except Exception as e:
        print(f"Eroare la incarcarea modelelor: {e}")
        return False


def predict_text(text):
    """Face predictii pentru un text folosind toti algoritmii"""
    if vectorizer is None or len(models) == 0:
        return None
    
    performance_metrics = {
        'preprocessing_time': 0,
        'vectorization_time': 0,
        'total_time': 0,
        'algorithms': {}
    }
    
    total_start = time.time()
    
    # Preprocesare text
    preprocess_start = time.time()
    preprocessor = TextPreprocessor(use_stemming=True, use_stopwords=True)
    processed_text = preprocessor.preprocess_text(text)
    performance_metrics['preprocessing_time'] = time.time() - preprocess_start
    
    # Vectorizare
    vectorize_start = time.time()
    X = vectorizer.transform([processed_text])
    performance_metrics['vectorization_time'] = time.time() - vectorize_start
    
    results = {}
    
    # Predictii pentru fiecare algoritm
    for name, model in models.items():
        algo_start = time.time()
        
        prediction_id = model.predict(X)[0]
        prediction_name = category_names[prediction_id]
        
        # Probabilitati (daca sunt disponibile)
        try:
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(X)[0]
                prob_dict = {category_names[i]: float(prob) for i, prob in enumerate(probabilities)}
            else:
                prob_dict = {prediction_name: 1.0}
        except:
            prob_dict = {prediction_name: 1.0}
        
        prediction_time = time.time() - algo_start
        
        # Formateaza numele categoriei
        formatted_prediction = format_category_name(prediction_name)
        
        # Formateaza probabilitatile
        formatted_probs = {format_category_name(k): v for k, v in prob_dict.items()}
        
        results[name] = {
            'prediction': formatted_prediction,
            'prediction_original': prediction_name,  # Pastreaza originalul pentru referinta
            'prediction_id': int(prediction_id),
            'probabilities': formatted_probs,
            'confidence': float(max(prob_dict.values())),
            'prediction_time': prediction_time,
            'prediction_time_ms': prediction_time * 1000  # In milisecunde
        }
        
        performance_metrics['algorithms'][name] = {
            'prediction_time': prediction_time,
            'prediction_time_ms': prediction_time * 1000
        }
    
    performance_metrics['total_time'] = time.time() - total_start
    
    return results, performance_metrics


@app.route('/')
def index():
    """Pagina principala"""
    return render_template('index.html')


@app.route('/details')
def details():
    """Pagina cu detalii despre antrenare"""
    return render_template('details.html')


@app.route('/api/training-info')
def get_training_info():
    """Endpoint pentru informatii despre antrenare"""
    try:
        base_dir = os.path.dirname(__file__)
        info_path = os.path.join(base_dir, 'models', 'training_info.json')
        
        if not os.path.exists(info_path):
            return jsonify({'error': 'Informatiile de antrenare nu sunt disponibile. Antreneaza modelele mai intai.'}), 404
        
        with open(info_path, 'r', encoding='utf-8') as f:
            training_info = json.load(f)
        
        # Formateaza categoriile
        if 'dataset' in training_info and 'categories' in training_info['dataset']:
            training_info['dataset']['categories_formatted'] = [
                format_category_name(cat) for cat in training_info['dataset']['categories']
            ]
        
        return jsonify(training_info)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/predict', methods=['POST'])
def predict():
    """Endpoint pentru predictie"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if not text:
            return jsonify({'error': 'Textul este gol'}), 400
        
        prediction_result = predict_text(text)
        
        if prediction_result is None:
            return jsonify({'error': 'Modelele nu sunt încărcate. Antrenează-le mai întâi.'}), 500
        
        results, performance_metrics = prediction_result
        
        # Formateaza numele categoriilor pentru afisare
        formatted_category_names = [format_category_name(cat) for cat in category_names]
        
        return jsonify({
            'success': True,
            'results': results,
            'category_names': formatted_category_names,
            'category_names_original': category_names,  # Pentru referinta
            'performance': performance_metrics,
            'text_length': len(text),
            'processed_text_length': len(text.split())  # Numar de cuvinte aproximativ
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Endpoint pentru upload fisier"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'Nu s-a incarcat niciun fisier'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'Nu s-a selectat niciun fisier'}), 400
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Citeste continutul fisierului
            try:
                if filename.endswith('.txt'):
                    with open(filepath, 'r', encoding='utf-8') as f:
                        text = f.read()
                elif filename.endswith('.csv'):
                    df = pd.read_csv(filepath)
                    # Ia prima coloana sau combina toate coloanele text
                    text = ' '.join(df.iloc[:, 0].astype(str).tolist())
                elif filename.endswith('.json'):
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if isinstance(data, dict):
                            text = ' '.join(str(v) for v in data.values() if isinstance(v, str))
                        else:
                            text = str(data)
                elif filename.endswith('.pdf'):
                    if not PDF_SUPPORT:
                        return jsonify({'error': 'Suportul pentru PDF nu este disponibil. Instaleaza PyPDF2 sau pdfplumber: pip install PyPDF2'}), 500
                    
                    text = extract_text_from_pdf(filepath)
                    if not text or len(text.strip()) == 0:
                        return jsonify({'error': 'Nu s-a putut extrage text din PDF. Fisierul poate fi scanat sau corupt.'}), 500
                
                # Sterge fisierul dupa citire
                os.remove(filepath)
                
                # Face predictii
                prediction_result = predict_text(text)
                
                if prediction_result is None:
                    return jsonify({'error': 'Modelele nu sunt incarcate. Antreneaza-le mai intai.'}), 500
                
                results, performance_metrics = prediction_result
                
                # Formateaza numele categoriilor pentru afisare
                formatted_category_names = [format_category_name(cat) for cat in category_names]
                
                return jsonify({
                    'success': True,
                    'text': text[:500] + '...' if len(text) > 500 else text,  # Primele 500 caractere
                    'text_length': len(text),
                    'processed_text_length': len(text.split()),  # Numar de cuvinte aproximativ
                    'results': results,
                    'category_names': formatted_category_names,
                    'category_names_original': category_names,  # Pentru referinta
                    'performance': performance_metrics
                })
            
            except Exception as e:
                if os.path.exists(filepath):
                    os.remove(filepath)
                return jsonify({'error': f'Eroare la citirea fisierului: {str(e)}'}), 500
        
        return jsonify({'error': 'Tip de fișier nepermis'}), 400
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    # Incarca modelele la pornire
    if load_models():
        port = 5001  # Foloseste port 5001 pentru a evita conflictul cu AirPlay
        print(f"Server pornit! Acceseaza http://localhost:{port}")
        app.run(debug=True, host='0.0.0.0', port=port)
    else:
        print("Nu s-au putut incarca modelele. Ruleaza mai intai: python3 train_models.py")

