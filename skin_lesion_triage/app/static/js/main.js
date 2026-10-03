const dropZone = document.getElementById("drop-zone");
const fileInput = document.getElementById("file-input");
const previewContainer = document.getElementById("preview-container");
const previewImage = document.getElementById("preview-image");
const analyzeBtn = document.getElementById("analyze-btn");
const loading = document.getElementById("loading");
const errorBox = document.getElementById("error-box");
const resultsCard = document.getElementById("results");

let selectedFile = null;

dropZone.addEventListener("click", () => fileInput.click());

dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("drag-over");
});

dropZone.addEventListener("dragleave", () => {
    dropZone.classList.remove("drag-over");
});

dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("drag-over");
    if (e.dataTransfer.files.length) {
        handleFile(e.dataTransfer.files[0]);
    }
});

fileInput.addEventListener("change", () => {
    if (fileInput.files.length) {
        handleFile(fileInput.files[0]);
    }
});

function handleFile(file) {
    const validTypes = ["image/png", "image/jpeg"];
    if (!validTypes.includes(file.type)) {
        showError("Please upload a PNG or JPG image.");
        return;
    }
    if (file.size > 8 * 1024 * 1024) {
        showError("File too large. Maximum size is 8MB.");
        return;
    }

    selectedFile = file;
    hideError();

    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        previewContainer.classList.remove("d-none");
        analyzeBtn.disabled = false;
    };
    reader.readAsDataURL(file);
}

analyzeBtn.addEventListener("click", async () => {
    if (!selectedFile) return;

    hideError();
    resultsCard.classList.add("d-none");
    loading.classList.remove("d-none");
    analyzeBtn.disabled = true;

    const formData = new FormData();
    formData.append("image", selectedFile);

    try {
        const response = await fetch("/predict", {
            method: "POST",
            body: formData,
        });
        const data = await response.json();

        if (!response.ok) {
            showError(data.error || "Something went wrong during analysis.");
            return;
        }

        renderResults(data);
    } catch (err) {
        showError("Network error: could not reach the server.");
    } finally {
        loading.classList.add("d-none");
        analyzeBtn.disabled = false;
    }
});

function renderResults(data) {
    document.getElementById("result-original").src = "data:image/png;base64," + data.original_image_b64;
    document.getElementById("result-heatmap").src = "data:image/png;base64," + data.heatmap_overlay_b64;

    const isMalignant = data.risk_flag === "urgent_referral";
    const badgeClass = isMalignant ? "risk-badge-malignant" : "risk-badge-benign";
    const badgeText = isMalignant ? "Urgent Referral Suggested" : "Routine / Low Risk";

    document.getElementById("result-summary").innerHTML = `
        <span class="badge ${badgeClass} fs-6 mb-2">${badgeText}</span>
        <table class="table table-sm mt-2">
            <tr><th>Predicted Class</th><td>${data.label}</td></tr>
            <tr><th>Malignant-suspect model score (uncalibrated)</th><td>${Number(data.malignant_probability).toFixed(4)} <span class="text-muted">(0–1 score)</span></td></tr>
            <tr><th>Benign model score (uncalibrated)</th><td>${Number(data.benign_probability).toFixed(4)} <span class="text-muted">(complementary 0–1 score)</span></td></tr>
            <tr><th>Image Sharpness Score</th><td>${data.sharpness_score}</td></tr>
            <tr><th>Model Used</th><td>${data.model_used}</td></tr>
        </table>
        <p class="text-muted small">Model scores are uncalibrated outputs, not probabilities or a diagnosis. This is not a substitute for assessment by a qualified clinician.</p>
    `;

    resultsCard.classList.remove("d-none");
    resultsCard.scrollIntoView({ behavior: "smooth" });
}

function showError(msg) {
    errorBox.textContent = msg;
    errorBox.classList.remove("d-none");
}

function hideError() {
    errorBox.classList.add("d-none");
}
