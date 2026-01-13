let currentFile = null;
let fileText = '';
let charts = {}; // Store chart instances

// History management
function saveToHistory(data) {
    // Skip if loading from history
    if (data._fromHistory) return;
    
    const history = getHistory();
    const historyItem = {
        id: Date.now(),
        timestamp: new Date().toLocaleString('ro-RO'),
        text: data.text || data.text_preview || 'Text input',
        textLength: data.text_length || 0,
        processedTextLength: data.processed_text_length || 0,
        results: data.results,
        performance: data.performance,
        categoryNames: data.category_names,
        // Extract detailed metrics for easy comparison
        metrics: {
            naive_bayes: {
                prediction: data.results.naive_bayes.prediction,
                confidence: data.results.naive_bayes.confidence,
                predictionTime: data.results.naive_bayes.prediction_time_ms,
                probabilities: data.results.naive_bayes.probabilities
            },
            svm: {
                prediction: data.results.svm.prediction,
                confidence: data.results.svm.confidence,
                predictionTime: data.results.svm.prediction_time_ms,
                probabilities: data.results.svm.probabilities
            },
            random_forest: {
                prediction: data.results.random_forest.prediction,
                confidence: data.results.random_forest.confidence,
                predictionTime: data.results.random_forest.prediction_time_ms,
                probabilities: data.results.random_forest.probabilities
            }
        },
        performanceMetrics: {
            preprocessingTime: data.performance.preprocessing_time,
            vectorizationTime: data.performance.vectorization_time,
            totalTime: data.performance.total_time
        }
    };
    
    history.unshift(historyItem); // Add to beginning
    // Keep only last 50 items
    if (history.length > 50) {
        history.pop();
    }
    
    localStorage.setItem('classification_history', JSON.stringify(history));
    loadHistory();
}

function getHistory() {
    const historyStr = localStorage.getItem('classification_history');
    return historyStr ? JSON.parse(historyStr) : [];
}

function loadHistory() {
    const history = getHistory();
    const historyList = document.getElementById('history-list');
    
    if (history.length === 0) {
        historyList.innerHTML = '<p class="no-history">Nu există istoric. Clasifică un document pentru a începe.</p>';
        return;
    }
    
    historyList.innerHTML = history.map(item => {
        const metrics = item.metrics || {
            naive_bayes: item.results.naive_bayes,
            svm: item.results.svm,
            random_forest: item.results.random_forest
        };
        
        // Get top 3 probabilities for each algorithm
        const getTopProbs = (probs) => {
            return Object.entries(probs || {})
                .sort((a, b) => b[1] - a[1])
                .slice(0, 3)
                .map(([cat, prob]) => `${cat}: ${(prob * 100).toFixed(1)}%`)
                .join(', ');
        };
        
        // Check if algorithms agree
        const predictions = [
            metrics.naive_bayes.prediction,
            metrics.svm.prediction,
            metrics.random_forest.prediction
        ];
        const allAgree = predictions[0] === predictions[1] && predictions[1] === predictions[2];
        const twoAgree = predictions.filter(p => p === predictions[0]).length === 2 || 
                         predictions.filter(p => p === predictions[1]).length === 2;
        
        return `
        <div class="history-item" onclick="showHistoryItem(${item.id})">
            <div class="history-item-header">
                <span class="history-timestamp"><i class="fas fa-clock"></i> ${item.timestamp}</span>
                <div class="history-badges">
                    ${allAgree ? '<span class="history-badge badge-agree"><i class="fas fa-check-circle"></i> Acord total</span>' : ''}
                    ${twoAgree && !allAgree ? '<span class="history-badge badge-partial"><i class="fas fa-exclamation-circle"></i> Acord parțial</span>' : ''}
                    ${!allAgree && !twoAgree ? '<span class="history-badge badge-disagree"><i class="fas fa-times-circle"></i> Discrepanțe</span>' : ''}
                </div>
                <button class="delete-history-btn" onclick="deleteHistoryItem(${item.id}, event)" title="Șterge">
                    <i class="fas fa-times"></i>
                </button>
            </div>
            <div class="history-item-preview">
                <div class="history-text-preview">
                    <strong>Text:</strong> ${item.text.substring(0, 150)}${item.text.length > 150 ? '...' : ''}
                    <span class="history-text-length">(${item.textLength || 0} caractere, ${item.processedTextLength || 0} cuvinte)</span>
                </div>
                
                <div class="history-comparison-table">
                    <table class="history-table">
                        <thead>
                            <tr>
                                <th>Algoritm</th>
                                <th>Categorie</th>
                                <th>Încredere</th>
                                <th>Timp (ms)</th>
                                <th>Top 3 Probabilități</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr>
                                <td><strong>Naive Bayes</strong></td>
                                <td>${metrics.naive_bayes.prediction}</td>
                                <td>${(metrics.naive_bayes.confidence * 100).toFixed(2)}%</td>
                                <td>${metrics.naive_bayes.predictionTime?.toFixed(3) || (item.results?.naive_bayes?.prediction_time_ms?.toFixed(3) || '-')}</td>
                                <td class="history-probs">${getTopProbs(metrics.naive_bayes.probabilities || item.results?.naive_bayes?.probabilities || {})}</td>
                            </tr>
                            <tr>
                                <td><strong>SVM</strong></td>
                                <td>${metrics.svm.prediction}</td>
                                <td>${(metrics.svm.confidence * 100).toFixed(2)}%</td>
                                <td>${metrics.svm.predictionTime?.toFixed(3) || (item.results?.svm?.prediction_time_ms?.toFixed(3) || '-')}</td>
                                <td class="history-probs">${getTopProbs(metrics.svm.probabilities || item.results?.svm?.probabilities || {})}</td>
                            </tr>
                            <tr>
                                <td><strong>Random Forest</strong></td>
                                <td>${metrics.random_forest.prediction}</td>
                                <td>${(metrics.random_forest.confidence * 100).toFixed(2)}%</td>
                                <td>${metrics.random_forest.predictionTime?.toFixed(3) || (item.results?.random_forest?.prediction_time_ms?.toFixed(3) || '-')}</td>
                                <td class="history-probs">${getTopProbs(metrics.random_forest.probabilities || item.results?.random_forest?.probabilities || {})}</td>
                            </tr>
                        </tbody>
                    </table>
                </div>
                
                ${item.performanceMetrics ? `
                <div class="history-performance">
                    <strong>Performanță:</strong>
                    <span>Preprocesare: ${(item.performanceMetrics.preprocessingTime * 1000).toFixed(3)}ms</span>
                    <span>Vectorizare: ${(item.performanceMetrics.vectorizationTime * 1000).toFixed(3)}ms</span>
                    <span>Total: ${(item.performanceMetrics.totalTime * 1000).toFixed(3)}ms</span>
                </div>
                ` : ''}
            </div>
        </div>
    `;
    }).join('');
}

function showHistoryItem(id) {
    const history = getHistory();
    const item = history.find(h => h.id === id);
    
    if (!item) return;
    
    // Create a data object similar to API response
    const data = {
        text: item.text,
        text_length: item.textLength,
        results: item.results,
        performance: item.performance,
        category_names: item.categoryNames,
        _fromHistory: true  // Flag to prevent saving to history again
    };
    
    displayResults(data);
    
    // Scroll to results
    document.getElementById('results-section').scrollIntoView({ behavior: 'smooth' });
}

function deleteHistoryItem(id, event) {
    event.stopPropagation();
    const history = getHistory();
    const filtered = history.filter(h => h.id !== id);
    localStorage.setItem('classification_history', JSON.stringify(filtered));
    loadHistory();
}

function clearHistory() {
    if (confirm('Ești sigur că vrei să ștergi tot istoricul?')) {
        localStorage.removeItem('classification_history');
        loadHistory();
    }
}

// Load history on page load
document.addEventListener('DOMContentLoaded', function() {
    loadHistory();
});

// Switch tabs
function switchTab(tab) {
    // Update buttons
    document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
    event.target.classList.add('active');
    
    // Update content
    document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));
    document.getElementById(`${tab}-tab`).classList.add('active');
    
    // Clear results
    clearResults();
}

// Handle file selection
function handleFileSelect(event) {
    const file = event.target.files[0];
    if (!file) return;
    
    currentFile = file;
    const fileInfo = document.getElementById('file-info');
    const predictBtn = document.getElementById('file-predict-btn');
    
    fileInfo.classList.remove('hidden');
    fileInfo.innerHTML = `
        <strong>📄 ${file.name}</strong><br>
        <span>Mărime: ${(file.size / 1024).toFixed(2)} KB</span>
    `;
    
    predictBtn.disabled = false;
    
    // Read file preview
    const reader = new FileReader();
    reader.onload = function(e) {
        fileText = e.target.result;
    };
    reader.readAsText(file);
}

// Predict from text
async function predictFromText() {
    const text = document.getElementById('text-input').value.trim();
    
    if (!text) {
        showError('Te rog introdu un text!');
        return;
    }
    
    await makePrediction(text);
}

// Predict from file
async function predictFromFile() {
    if (!currentFile) {
        showError('Te rog selectează un fișier!');
        return;
    }
    
    const formData = new FormData();
    formData.append('file', currentFile);
    
    showLoading();
    clearResults();
    
    try {
        const response = await fetch('/api/upload', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.error || 'Eroare la procesare');
        }
        
        displayResults(data);
    } catch (error) {
        showError(error.message);
    } finally {
        hideLoading();
    }
}

// Make prediction API call
async function makePrediction(text) {
    showLoading();
    clearResults();
    
    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ text: text })
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.error || 'Eroare la predicție');
        }
        
        // Add text preview for history
        data.text = text.substring(0, 500) + (text.length > 500 ? '...' : '');
        data.text_preview = text.substring(0, 500) + (text.length > 500 ? '...' : '');
        
        displayResults(data);
    } catch (error) {
        showError(error.message);
    } finally {
        hideLoading();
    }
}

// Display results
function displayResults(data) {
    const resultsSection = document.getElementById('results-section');
    resultsSection.classList.remove('hidden');
    
    // Show text preview
    const textPreview = document.getElementById('text-preview');
    if (data.text) {
        textPreview.textContent = data.text;
        textPreview.classList.remove('hidden');
    } else {
        textPreview.classList.add('hidden');
    }
    
    const results = data.results;
    const categoryNames = data.category_names || [];
    const performance = data.performance || {};
    
    // Save to history (only if not loading from history)
    if (!data._fromHistory) {
        saveToHistory(data);
    }
    
    // Display results for each algorithm
    displayAlgorithmResult('naive_bayes', 'nb', results.naive_bayes, categoryNames);
    displayAlgorithmResult('svm', 'svm', results.svm, categoryNames);
    displayAlgorithmResult('random_forest', 'rf', results.random_forest, categoryNames);
    
    // Display comparison table
    displayComparison(results, categoryNames);
    
    // Display performance metrics
    displayPerformanceMetrics(performance, data.text_length, data.processed_text_length);
    
    // Create charts
    createCharts(results, performance);
    
    // Display detailed stats
    displayDetailedStats(results, performance);
}

// Display algorithm result
function displayAlgorithmResult(algorithmName, prefix, result, categoryNames) {
    // Prediction
    document.getElementById(`${prefix}-prediction`).textContent = result.prediction;
    
    // Confidence
    const confidence = (result.confidence * 100).toFixed(2);
    document.getElementById(`${prefix}-confidence`).style.width = `${confidence}%`;
    document.getElementById(`${prefix}-confidence-text`).textContent = `${confidence}%`;
    
    // Probabilities
    const probabilitiesDiv = document.getElementById(`${prefix}-probabilities`);
    probabilitiesDiv.innerHTML = '<div style="font-size: 0.85em; color: #888; margin-bottom: 8px;">Probabilități:</div>';
    
    // Sort probabilities
    const sortedProbs = Object.entries(result.probabilities)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 5); // Top 5
    
    sortedProbs.forEach(([category, prob]) => {
        const probItem = document.createElement('div');
        probItem.className = 'prob-item';
        probItem.innerHTML = `
            <span class="prob-item-name">${category}</span>
            <span class="prob-item-value">${(prob * 100).toFixed(2)}%</span>
        `;
        probabilitiesDiv.appendChild(probItem);
    });
}

// Display comparison table
function displayComparison(results, categoryNames) {
    const tableDiv = document.getElementById('comparison-table');
    
    const table = document.createElement('table');
    table.innerHTML = `
        <thead>
            <tr>
                <th>Algoritm</th>
                <th>Categorie Prezisă</th>
                <th>Încredere</th>
                <th>Timp Predicție</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td><strong>Naive Bayes</strong></td>
                <td>${results.naive_bayes.prediction}</td>
                <td>${(results.naive_bayes.confidence * 100).toFixed(2)}%</td>
                <td>${results.naive_bayes.prediction_time_ms.toFixed(3)} ms</td>
            </tr>
            <tr>
                <td><strong>SVM</strong></td>
                <td>${results.svm.prediction}</td>
                <td>${(results.svm.confidence * 100).toFixed(2)}%</td>
                <td>${results.svm.prediction_time_ms.toFixed(3)} ms</td>
            </tr>
            <tr>
                <td><strong>Random Forest</strong></td>
                <td>${results.random_forest.prediction}</td>
                <td>${(results.random_forest.confidence * 100).toFixed(2)}%</td>
                <td>${results.random_forest.prediction_time_ms.toFixed(3)} ms</td>
            </tr>
        </tbody>
    `;
    
    tableDiv.innerHTML = '';
    tableDiv.appendChild(table);
}

// Display performance metrics
function displayPerformanceMetrics(performance, textLength, processedLength) {
    if (!performance) return;
    
    // Format times
    const formatTime = (seconds) => {
        if (seconds < 0.001) return `${(seconds * 1000).toFixed(3)} ms`;
        return `${seconds.toFixed(6)}s`;
    };
    
    document.getElementById('preprocessing-time').textContent = formatTime(performance.preprocessing_time || 0);
    document.getElementById('vectorization-time').textContent = formatTime(performance.vectorization_time || 0);
    document.getElementById('total-time').textContent = formatTime(performance.total_time || 0);
    
    if (textLength) {
        document.getElementById('text-length').textContent = `${textLength.toLocaleString()} caractere`;
    }
}

// Create charts
function createCharts(results, performance) {
    // Destroy existing charts
    Object.values(charts).forEach(chart => {
        if (chart) chart.destroy();
    });
    charts = {};
    
    const algorithmNames = ['Naive Bayes', 'SVM', 'Random Forest'];
    const algorithmKeys = ['naive_bayes', 'svm', 'random_forest'];
    const colors = {
        'naive_bayes': 'rgba(102, 126, 234, 0.8)',
        'svm': 'rgba(118, 75, 162, 0.8)',
        'random_forest': 'rgba(52, 152, 219, 0.8)'
    };
    
    // Speed Chart
    const speedCtx = document.getElementById('speed-chart').getContext('2d');
    const speedData = algorithmKeys.map(key => 
        (results[key]?.prediction_time_ms || 0).toFixed(2)
    );
    
    charts.speed = new Chart(speedCtx, {
        type: 'bar',
        data: {
            labels: algorithmNames,
            datasets: [{
                label: 'Timp Predicție (ms)',
                data: speedData,
                backgroundColor: [
                    colors.naive_bayes,
                    colors.svm,
                    colors.random_forest
                ],
                borderColor: [
                    'rgba(102, 126, 234, 1)',
                    'rgba(118, 75, 162, 1)',
                    'rgba(52, 152, 219, 1)'
                ],
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return `${context.parsed.y} ms`;
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Timp (ms)'
                    }
                }
            }
        }
    });
    
    // Confidence Chart
    const confidenceCtx = document.getElementById('confidence-chart').getContext('2d');
    const confidenceData = algorithmKeys.map(key => 
        ((results[key]?.confidence || 0) * 100).toFixed(2)
    );
    
    charts.confidence = new Chart(confidenceCtx, {
        type: 'bar',
        data: {
            labels: algorithmNames,
            datasets: [{
                label: 'Încredere (%)',
                data: confidenceData,
                backgroundColor: [
                    colors.naive_bayes,
                    colors.svm,
                    colors.random_forest
                ],
                borderColor: [
                    'rgba(102, 126, 234, 1)',
                    'rgba(118, 75, 162, 1)',
                    'rgba(52, 152, 219, 1)'
                ],
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return `${context.parsed.y}%`;
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    title: {
                        display: true,
                        text: 'Încredere (%)'
                    }
                }
            }
        }
    });
    
    // Probabilities Chart (Top predictions from all algorithms)
    const probCtx = document.getElementById('probabilities-chart').getContext('2d');
    const allProbs = {};
    
    algorithmKeys.forEach(key => {
        const probs = results[key]?.probabilities || {};
        Object.entries(probs).forEach(([cat, prob]) => {
            if (!allProbs[cat]) allProbs[cat] = 0;
            allProbs[cat] = Math.max(allProbs[cat], prob);
        });
    });
    
    const sortedProbs = Object.entries(allProbs)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 6);
    
    charts.probabilities = new Chart(probCtx, {
        type: 'doughnut',
        data: {
            labels: sortedProbs.map(([cat]) => cat),
            datasets: [{
                data: sortedProbs.map(([, prob]) => (prob * 100).toFixed(2)),
                backgroundColor: [
                    'rgba(102, 126, 234, 0.8)',
                    'rgba(118, 75, 162, 0.8)',
                    'rgba(52, 152, 219, 0.8)',
                    'rgba(46, 204, 113, 0.8)',
                    'rgba(241, 196, 15, 0.8)',
                    'rgba(231, 76, 60, 0.8)'
                ]
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return `${context.label}: ${context.parsed}%`;
                        }
                    }
                }
            }
        }
    });
    
    // Comparison Chart (Multiple metrics)
    const comparisonCtx = document.getElementById('comparison-chart').getContext('2d');
    
    charts.comparison = new Chart(comparisonCtx, {
        type: 'radar',
        data: {
            labels: ['Viteza', 'Încredere', 'Precizie'],
            datasets: algorithmKeys.map((key, idx) => {
                const speed = parseFloat(speedData[idx]);
                const maxSpeed = Math.max(...speedData.map(s => parseFloat(s)));
                const normalizedSpeed = maxSpeed > 0 ? (1 - (speed / maxSpeed)) * 100 : 0;
                
                return {
                    label: algorithmNames[idx],
                    data: [
                        normalizedSpeed, // Normalized speed (higher is better)
                        parseFloat(confidenceData[idx]),
                        parseFloat(confidenceData[idx]) // Using confidence as precision proxy
                    ],
                    backgroundColor: Object.values(colors)[idx].replace('0.8', '0.2'),
                    borderColor: Object.values(colors)[idx].replace('0.8', '1'),
                    borderWidth: 2
                };
            })
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            scales: {
                r: {
                    beginAtZero: true,
                    max: 100
                }
            }
        }
    });
}

// Display detailed stats
function displayDetailedStats(results, performance) {
    const statsDiv = document.getElementById('detailed-stats');
    statsDiv.innerHTML = '';
    
    const algorithmNames = {
        'naive_bayes': 'Naive Bayes',
        'svm': 'SVM',
        'random_forest': 'Random Forest'
    };
    
    Object.entries(results).forEach(([key, result]) => {
        const statCard = document.createElement('div');
        statCard.className = 'stat-card';
        
        const topProbs = Object.entries(result.probabilities)
            .sort((a, b) => b[1] - a[1])
            .slice(0, 3);
        
        statCard.innerHTML = `
            <h4>${algorithmNames[key]}</h4>
            <div class="stat-item">
                <span class="stat-item-label">Categorie Prezisă:</span>
                <span class="stat-item-value">${result.prediction}</span>
            </div>
            <div class="stat-item">
                <span class="stat-item-label">Încredere:</span>
                <span class="stat-item-value">${(result.confidence * 100).toFixed(2)}%</span>
            </div>
            <div class="stat-item">
                <span class="stat-item-label">Timp Predicție:</span>
                <span class="stat-item-value">${result.prediction_time_ms.toFixed(3)} ms</span>
            </div>
            <div class="stat-item">
                <span class="stat-item-label">Top 3 Probabilități:</span>
                <span class="stat-item-value"></span>
            </div>
            ${topProbs.map(([cat, prob], idx) => `
                <div class="stat-item" style="padding-left: 15px;">
                    <span class="stat-item-label">${idx + 1}. ${cat}:</span>
                    <span class="stat-item-value">${(prob * 100).toFixed(2)}%</span>
                </div>
            `).join('')}
        `;
        
        statsDiv.appendChild(statCard);
    });
}

// Clear results
function clearResults() {
    document.getElementById('results-section').classList.add('hidden');
    document.getElementById('error').classList.add('hidden');
    
    // Destroy charts
    Object.values(charts).forEach(chart => {
        if (chart) chart.destroy();
    });
    charts = {};
}

// Show loading
function showLoading() {
    document.getElementById('loading').classList.remove('hidden');
}

// Hide loading
function hideLoading() {
    document.getElementById('loading').classList.add('hidden');
}

// Show error
function showError(message) {
    const errorDiv = document.getElementById('error');
    errorDiv.innerHTML = `<i class="fas fa-exclamation-circle"></i> ${message}`;
    errorDiv.classList.remove('hidden');
    hideLoading();
}

// Drag and drop
const fileUploadArea = document.getElementById('file-upload-area');
const fileInput = document.getElementById('file-input');

fileUploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    fileUploadArea.style.background = '#e8f4f8';
});

fileUploadArea.addEventListener('dragleave', () => {
    fileUploadArea.style.background = '#f8f9ff';
});

fileUploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    fileUploadArea.style.background = '#f8f9ff';
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        fileInput.files = files;
        handleFileSelect({ target: fileInput });
    }
});

