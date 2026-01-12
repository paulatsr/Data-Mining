#!/bin/bash
# Script de setup pentru proiectul de data mining

echo "🔧 Creare virtual environment..."
python3 -m venv venv

echo "✅ Activare virtual environment..."
source venv/bin/activate

echo "📦 Instalare dependențe..."
pip install -r requirements.txt

echo ""
echo "✅ Setup complet!"
echo ""
echo "💡 Pentru a activa virtual environment-ul în viitor, rulează:"
echo "   source venv/bin/activate"
echo ""
echo "🚀 Apoi poți rula:"
echo "   python3 scripts/download_20newsgroups.py"

