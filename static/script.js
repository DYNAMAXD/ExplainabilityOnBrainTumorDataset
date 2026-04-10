document.addEventListener("DOMContentLoaded", () => {
    const uploadArea = document.getElementById("uploadArea");
    const imageInput = document.getElementById("imageInput");
    const imagePreview = document.getElementById("imagePreview");
    const uploadContent = document.querySelector(".upload-content");
    const analyzeBtn = document.getElementById("analyzeBtn");
    const loadingSpinner = document.getElementById("loadingSpinner");
    const resultsContainer = document.getElementById("resultsContainer");

    const predClassEl = document.getElementById("predClass");
    const predConfEl = document.getElementById("predConf");

    const imgIds = [
        "origImg", "gradcamImg", "gradcamppImg", 
        "scorecamImg", "occlusionImg", "eigencamImg", "layercamImg"
    ];

    let currentFile = null;

    // Handle Drag & Drop
    uploadArea.addEventListener("dragover", (e) => {
        e.preventDefault();
        uploadArea.classList.add("dragover");
    });

    uploadArea.addEventListener("dragleave", () => {
        uploadArea.classList.remove("dragover");
    });

    uploadArea.addEventListener("drop", (e) => {
        e.preventDefault();
        uploadArea.classList.remove("dragover");
        if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
            handleFile(e.dataTransfer.files[0]);
        }
    });

    // Handle Click
    uploadArea.addEventListener("click", () => {
        imageInput.click();
    });

    imageInput.addEventListener("change", (e) => {
        if (e.target.files && e.target.files.length > 0) {
            handleFile(e.target.files[0]);
        }
    });

    function handleFile(file) {
        if (!file.type.startsWith('image/')) {
            alert('Please select a valid image file');
            return;
        }
        currentFile = file;

        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            imagePreview.classList.remove("hidden");
            uploadContent.classList.add("hidden");
            analyzeBtn.classList.remove("disabled");
            analyzeBtn.disabled = false;
        };
        reader.readAsDataURL(file);
    }

    analyzeBtn.addEventListener("click", async () => {
        if (!currentFile) return;

        // UI updates for loading state
        analyzeBtn.disabled = true;
        analyzeBtn.querySelector('span').innerText = 'Analyzing...';
        loadingSpinner.classList.remove("hidden");
        resultsContainer.classList.add("hidden");

        const formData = new FormData();
        formData.append("file", currentFile);

        try {
            const response = await fetch("/predict", {
                method: "POST",
                body: formData
            });

            if (!response.ok) {
                throw new Error(`Server returned ${response.status}: ${response.statusText}`);
            }

            const data = await response.json();
            
            // Set Text results
            predClassEl.innerText = data.pred_class;
            predConfEl.innerText = (data.confidence * 100).toFixed(2) + "%";

            // Determine if pred_class is normal or tumor for potential dynamic color coding
            if(data.pred_class === "notumor") {
                predClassEl.style.color = "#48bb78"; // Green accent
            } else {
                predClassEl.style.color = "#f56565"; // Red accent (medical alert)
            }

            // Set Image results
            document.getElementById("origImg").src = "data:image/jpeg;base64," + data.original_image;
            document.getElementById("gradcamImg").src = "data:image/jpeg;base64," + data.cams.gradcam;
            document.getElementById("gradcamppImg").src = "data:image/jpeg;base64," + data.cams.gradcam_pp;
            document.getElementById("scorecamImg").src = "data:image/jpeg;base64," + data.cams.scorecam;
            document.getElementById("occlusionImg").src = "data:image/jpeg;base64," + data.cams.occlusion;
            document.getElementById("eigencamImg").src = "data:image/jpeg;base64," + data.cams.eigencam;
            document.getElementById("layercamImg").src = "data:image/jpeg;base64," + data.cams.layercam;

            // Show results
            resultsContainer.classList.remove("hidden");

            // Scroll into view
            resultsContainer.scrollIntoView({ behavior: 'smooth', block: 'start' });

        } catch (error) {
            console.error("Error during prediction:", error);
            alert("Error: " + error.message);
        } finally {
            analyzeBtn.disabled = false;
            analyzeBtn.querySelector('span').innerText = 'Generate Analysis';
            loadingSpinner.classList.add("hidden");
        }
    });

});
