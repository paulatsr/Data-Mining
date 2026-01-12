let currentFile = null;
let fileText = '';
let charts = {}; // Store chart instances

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

