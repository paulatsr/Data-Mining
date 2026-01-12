// Load training info on page load
document.addEventListener('DOMContentLoaded', function() {
    loadTrainingInfo();
});

async function loadTrainingInfo() {
    const loading = document.getElementById('loading');
    const content = document.getElementById('content');
    const error = document.getElementById('error');
    
    try {
        const response = await fetch('/api/training-info');
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.error || 'Eroare la încărcarea informațiilor');
        }
        
        displayTrainingInfo(data);
        loading.classList.add('hidden');
        content.classList.remove('hidden');
    } catch (err) {
        loading.classList.add('hidden');
        error.classList.remove('hidden');
        error.textContent = `❌ ${err.message}`;
    }
}

function displayTrainingInfo(info) {
    // Dataset info
    document.getElementById('dataset-name').textContent = info.dataset.name;
    document.getElementById('total-docs').textContent = info.dataset.total_documents.toLocaleString();
    document.getElementById('train-docs').textContent = info.dataset.train_documents.toLocaleString();
    document.getElementById('test-docs').textContent = info.dataset.test_documents.toLocaleString();
    document.getElementById('num-categories').textContent = info.dataset.num_categories;
    document.getElementById('num-features').textContent = info.dataset.features.toLocaleString();
    
    // Categories
    const categoriesList = document.getElementById('categories-list');
    categoriesList.innerHTML = '';
    info.dataset.categories_formatted.forEach(cat => {
        const catBadge = document.createElement('div');
        catBadge.className = 'category-badge';
        catBadge.textContent = cat;
        categoriesList.appendChild(catBadge);
    });
    
    // Preprocessing
    document.getElementById('vectorizer-type').textContent = info.preprocessing.vectorizer_type;
    document.getElementById('max-features').textContent = info.preprocessing.max_features.toLocaleString();
    document.getElementById('ngram-range').textContent = info.preprocessing.ngram_range;
    document.getElementById('stemming').textContent = info.preprocessing.use_stemming ? 'Da' : 'Nu';
    document.getElementById('stopwords').textContent = info.preprocessing.use_stopwords ? 'Da' : 'Nu';
    
    // Algorithms
    const nb = info.algorithms.naive_bayes;
    document.getElementById('nb-training-time').textContent = nb.training_time_formatted;
    document.getElementById('nb-accuracy').textContent = `${(nb.accuracy * 100).toFixed(2)}%`;
    document.getElementById('nb-params').textContent = `alpha: ${nb.parameters.alpha}`;
    
    const svm = info.algorithms.svm;
    document.getElementById('svm-training-time').textContent = svm.training_time_formatted;
    document.getElementById('svm-accuracy').textContent = `${(svm.accuracy * 100).toFixed(2)}%`;
    document.getElementById('svm-params').textContent = `kernel: ${svm.parameters.kernel}, C: ${svm.parameters.C}`;
    
    const rf = info.algorithms.random_forest;
    document.getElementById('rf-training-time').textContent = rf.training_time_formatted;
    document.getElementById('rf-accuracy').textContent = `${(rf.accuracy * 100).toFixed(2)}%`;
    document.getElementById('rf-params').textContent = `n_estimators: ${rf.parameters.n_estimators}, max_depth: ${rf.parameters.max_depth || 'None'}`;
    
    // Summary
    const bestAlgo = info.algorithms[info.summary.best_accuracy_algorithm];
    document.getElementById('best-accuracy').textContent = `${bestAlgo.name}: ${(bestAlgo.accuracy * 100).toFixed(2)}%`;
    
    const fastestAlgo = info.algorithms[info.summary.fastest_training_algorithm];
    document.getElementById('fastest-training').textContent = `${fastestAlgo.name}: ${fastestAlgo.training_time_formatted}`;
    
    // Training date
    const date = new Date(info.training_date);
    document.getElementById('training-date').textContent = date.toLocaleString('ro-RO');
    
    // Create training time chart
    createTrainingTimeChart(info.algorithms);
}

function createTrainingTimeChart(algorithms) {
    const ctx = document.getElementById('training-time-chart').getContext('2d');
    
    const labels = ['Naive Bayes', 'SVM', 'Random Forest'];
    const times = [
        algorithms.naive_bayes.training_time,
        algorithms.svm.training_time,
        algorithms.random_forest.training_time
    ];
    
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Timp Antrenare (secunde)',
                data: times,
                backgroundColor: [
                    'rgba(102, 126, 234, 0.8)',
                    'rgba(118, 75, 162, 0.8)',
                    'rgba(52, 152, 219, 0.8)'
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
                            return `${context.parsed.y.toFixed(6)}s`;
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Timp (secunde)'
                    }
                }
            }
        }
    });
}

