document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const previewArea = document.getElementById('preview-area');
    const imagePreview = document.getElementById('image-preview');
    const uploadCard = document.getElementById('upload-card');

    const loader = document.getElementById('loader');
    const resultsCard = document.getElementById('results-card');

    // Buttons
    const btnReselect = document.getElementById('btn-reselect');
    const btnAnalyze = document.getElementById('btn-analyze');
    const btnNewScan = document.getElementById('btn-new-scan');

    // Result Elements
    const diseaseName = document.getElementById('disease-name');
    const confidenceBadge = document.getElementById('confidence-badge');
    const diseaseDesc = document.getElementById('disease-desc');
    const diseaseTreatment = document.getElementById('disease-treatment');
    const resultsImage = document.getElementById('results-image');
    const probabilityGraph = document.getElementById('probability-graph');

    let selectedFile = null;

    // --- Drag and Drop Logic --- //
    dropZone.addEventListener('click', () => fileInput.click());

    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => dropZone.classList.add('dragover'), false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => dropZone.classList.remove('dragover'), false);
    });

    dropZone.addEventListener('drop', (e) => {
        let dt = e.dataTransfer;
        let files = dt.files;
        handleFiles(files);
    });

    fileInput.addEventListener('change', function () {
        handleFiles(this.files);
    });

    // --- File Handling --- //
    function handleFiles(files) {
        if (files.length === 0) return;

        const file = files[0];

        // Validate MIME type
        if (!file.type.match('image.*')) {
            alert('Please select a valid image file (JPG/PNG).');
            return;
        }

        selectedFile = file;

        // Create object URL for preview
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            // Switch UI state
            dropZone.classList.add('hidden');
            previewArea.classList.remove('hidden');
        };
        reader.readAsDataURL(file);
    }

    // --- Button Actions --- //
    btnReselect.addEventListener('click', () => {
        selectedFile = null;
        fileInput.value = '';
        imagePreview.src = '';
        previewArea.classList.add('hidden');
        dropZone.classList.remove('hidden');
    });

    btnNewScan.addEventListener('click', () => {
        // Reset Everything
        selectedFile = null;
        fileInput.value = '';
        imagePreview.src = '';

        resultsCard.classList.add('hidden');
        uploadCard.classList.remove('hidden');

        previewArea.classList.add('hidden');
        dropZone.classList.remove('hidden');
    });

    // --- API Request --- //
    btnAnalyze.addEventListener('click', async () => {
        if (!selectedFile) return;

        // Update UI States
        uploadCard.classList.add('hidden');
        loader.classList.remove('hidden');

        const formData = new FormData();
        formData.append('file', selectedFile);

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || 'Server responded with an error');
            }

            const data = await response.json();

            // Populate Results
            diseaseName.textContent = data.prediction;
            confidenceBadge.textContent = data.confidence;
            diseaseDesc.textContent = data.description;
            diseaseTreatment.textContent = data.treatment;

            // Set results image
            resultsImage.src = imagePreview.src;

            // Render Probability Graph
            renderProbabilityGraph(data.all_predictions);

            // Optional Dynamic Styling based on class
            if (data.prediction.toLowerCase().includes('healthy')) {
                diseaseName.style.background = 'linear-gradient(90deg, #10b981, #34d399)';
            } else {
                diseaseName.style.background = 'linear-gradient(90deg, #ef4444, #f59e0b)';
            }
            diseaseName.style.webkitBackgroundClip = 'text';
            diseaseName.style.webkitTextFillColor = 'transparent';

            // Show Results
            loader.classList.add('hidden');
            resultsCard.classList.remove('hidden');

        } catch (error) {
            console.error('Error during API call:', error);
            alert(`Analysis failed: ${error.message}`);

            // Revert UI
            loader.classList.add('hidden');
            uploadCard.classList.remove('hidden');
        }
    });

    function renderProbabilityGraph(predictions) {
        probabilityGraph.innerHTML = '';
        
        // Convert to array and sort by value descending
        const sortedPredictions = Object.entries(predictions)
            .sort((a, b) => b[1] - a[1]);

        sortedPredictions.forEach(([name, prob]) => {
            const percentage = (prob * 100).toFixed(1);
            
            const row = document.createElement('div');
            row.className = 'prob-row';
            
            row.innerHTML = `
                <div class="prob-label-container">
                    <span class="prob-label">${name}</span>
                    <span class="prob-value">${percentage}%</span>
                </div>
                <div class="prob-bar-bg">
                    <div class="prob-bar-fill" style="width: 0%"></div>
                </div>
            `;
            
            probabilityGraph.appendChild(row);
            
            // Animate after brief delay
            setTimeout(() => {
                const fill = row.querySelector('.prob-bar-fill');
                if (fill) fill.style.width = `${percentage}%`;
            }, 100);
        });
    }
});
