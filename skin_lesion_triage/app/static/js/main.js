document.addEventListener("DOMContentLoaded", function () {
    const dropZone = document.getElementById("drop-zone");
    const fileInput = document.getElementById("file-input");
    const previewContainer = document.getElementById("preview-container");
    const previewImage = document.getElementById("preview-image");
    const analyzeBtn = document.getElementById("analyze-btn");
    const loading = document.getElementById("loading");
    const errorBox = document.getElementById("error-box");
    const resultsCard = document.getElementById("results");
    const resultOriginal = document.getElementById("result-original");
    const resultHeatmap = document.getElementById("result-heatmap");
    const riskMeterBar = document.getElementById("risk-meter-bar");
    const resultSummary = document.getElementById("result-summary");

    let selectedFile = null;

    function showError(message) {
        if (!errorBox) return;
        errorBox.textContent = message;
        errorBox.classList.remove("d-none");
    }

    function hideError() {
        if (!errorBox) return;
        errorBox.classList.add("d-none");
        errorBox.textContent = "";
    }

    if (dropZone && fileInput) {
        dropZone.addEventListener("click", function () {
            fileInput.click();
        });

        dropZone.addEventListener("dragover", function (e) {
            e.preventDefault();
            dropZone.classList.add("drag-over");
        });

        dropZone.addEventListener("dragleave", function () {
            dropZone.classList.remove("drag-over");
        });

        dropZone.addEventListener("drop", function (e) {
            e.preventDefault();
            dropZone.classList.remove("drag-over");
            if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
                handleFile(e.dataTransfer.files[0]);
            }
        });

        fileInput.addEventListener("change", function () {
            if (fileInput.files && fileInput.files.length > 0) {
                handleFile(fileInput.files[0]);
            }
        });
    }

    function handleFile(file) {
        hideError();
        const allowed = ["image/png", "image/jpeg", "image/jpg"];
        if (allowed.indexOf(file.type) === -1) {
            showError("Please upload a PNG or JPG image.");
            return;
        }
        selectedFile = file;
        const reader = new FileReader();
        reader.onload = function (e) {
            if (previewImage) previewImage.src = e.target.result;
            if (previewContainer) previewContainer.classList.remove("d-none");
            if (analyzeBtn) analyzeBtn.disabled = false;
        };
        reader.readAsDataURL(file);
    }

    if (analyzeBtn) {
        analyzeBtn.addEventListener("click", function () {
            if (!selectedFile) return;
            runAnalysis(selectedFile);
        });
    }

    function runAnalysis(file) {
        hideError();
        if (loading) loading.classList.remove("d-none");
        if (resultsCard) resultsCard.classList.add("d-none");
        if (analyzeBtn) analyzeBtn.disabled = true;

        const formData = new FormData();
        formData.append("image", file);

        fetch("/predict", {
            method: "POST",
            body: formData
        })
            .then(function (response) {
                if (!response.ok) {
                    return response.json().then(function (errData) {
                        throw new Error(errData.error || "Request failed.");
                    });
                }
                return response.json();
            })
            .then(function (data) {
                renderResults(data);
            })
            .catch(function (err) {
                showError(err.message || "Something went wrong while analyzing the image.");
            })
            .finally(function () {
                if (loading) loading.classList.add("d-none");
                if (analyzeBtn) analyzeBtn.disabled = false;
            });
    }

    function pick(data, keys) {
        for (let i = 0; i < keys.length; i++) {
            if (data[keys[i]]) return data[keys[i]];
        }
        return null;
    }

    function formatRiskFlag(flag) {
        if (flag === "urgent_referral") return "Urgent Referral Suggested";
        if (flag === "routine") return "Routine";
        return flag || "Routine";
    }

    function renderResults(data) {
        if (!resultsCard) return;

        const label = data.label || data.predicted_label || "Unknown";
        const score = typeof data.malignant_probability === "number" ? data.malignant_probability : 0;
        const scorePct = (score * 100).toFixed(1);
        const riskFlag = formatRiskFlag(data.risk_flag);
        const isMalignant = label.toLowerCase().indexOf("malignant") !== -1;
        const badgeClass = isMalignant ? "risk-badge-malignant" : "risk-badge-benign";

        let meterClass = "risk-meter-low";
        if (score >= 0.66) {
            meterClass = "risk-meter-high";
        } else if (score >= 0.33) {
            meterClass = "risk-meter-medium";
        }

        const originalImg = pick(data, ["original_image_b64", "original_image", "processed_image"]);
        const heatmapImg = pick(data, ["heatmap_overlay_b64", "heatmap_image", "gradcam_image"]);

        if (resultOriginal && originalImg) {
            resultOriginal.src = "data:image/png;base64," + originalImg;
        }
        if (resultHeatmap && heatmapImg) {
            resultHeatmap.src = "data:image/png;base64," + heatmapImg;
        }

        if (riskMeterBar) {
            riskMeterBar.className = "progress-bar " + meterClass;
            riskMeterBar.style.width = scorePct + "%";
        }

        let refLine = "";
        if (data.reference_code) {
            const resultUrl = window.location.origin + "/result/" + data.reference_code;
            refLine = "<div class=\"alert alert-info small mb-3\"><i class=\"bi bi-ticket-perforated\"></i> " +
                "Save this code to view this result again later: <strong>" + data.reference_code + "</strong><br>" +
                "<a href=\"" + resultUrl + "\">" + resultUrl + "</a></div>";
        }

        if (resultSummary) {
            resultSummary.innerHTML =
                "<div class=\"d-flex align-items-center justify-content-between mb-3\">" +
                "<span class=\"badge " + badgeClass + " fs-6\">" + label + "</span>" +
                "<span class=\"text-muted small\">" + riskFlag + "</span>" +
                "</div>" +
                "<div class=\"text-center small text-muted mb-3\">Malignant probability: " + scorePct + "%</div>" +
                refLine;
        }

        resultsCard.classList.remove("d-none");
        resultsCard.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }

    const tiltCards = document.querySelectorAll(".tilt-3d");
    tiltCards.forEach(function (card) {
        card.addEventListener("mousemove", function (e) {
            const rect = card.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            const centerX = rect.width / 2;
            const centerY = rect.height / 2;
            const rotateX = ((y - centerY) / centerY) * -6;
            const rotateY = ((x - centerX) / centerX) * 6;
            card.style.transform = "perspective(800px) rotateX(" + rotateX + "deg) rotateY(" + rotateY + "deg) scale(1.02)";
        });
        card.addEventListener("mouseleave", function () {
            card.style.transform = "perspective(800px) rotateX(0deg) rotateY(0deg) scale(1)";
        });
    });

    const revealEls = document.querySelectorAll(".reveal");
    if ("IntersectionObserver" in window && revealEls.length > 0) {
        const observer = new IntersectionObserver(
            function (entries) {
                entries.forEach(function (entry) {
                    if (entry.isIntersecting) {
                        entry.target.classList.add("revealed");
                        observer.unobserve(entry.target);
                    }
                });
            },
            { threshold: 0.15 }
        );
        revealEls.forEach(function (el) {
            observer.observe(el);
        });
    } else {
        revealEls.forEach(function (el) {
            el.classList.add("revealed");
        });
    }
});
