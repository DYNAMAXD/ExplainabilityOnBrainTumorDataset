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

    // UPDATE: Point this to your actual Hugging Face backend URL
    const API_URL = "https://dynamaxd-braintumorxai.hf.space/predict";

    let currentFile = null;

    // --- SAMPLE CATALOG LOGIC ---
    const SAMPLE_REPO = "https://raw.githubusercontent.com/DYNAMAXD/ExplainabilityOnBrainTumorDataset/main/testImages";
    const SAMPLE_IMAGES = {
        glioma: ["Tr-gl_1016.jpg", "Tr-gl_1111.jpg", "Tr-gl_1124.jpg", "Tr-gl_1149.jpg", "Tr-gl_1186.jpg"],
        meningioma: ["Tr-me_1041.jpg", "Tr-me_1048.jpg", "Tr-me_1088.jpg", "Tr-me_1134.jpg", "Tr-me_1137.jpg"],
        notumor: ["Tr-no_1.jpg", "Tr-no_101.jpg", "Tr-no_1043.jpg", "Tr-no_1130.jpg", "Tr-no_1161.jpg"],
        pituitary: ["Tr-pi_1122.jpg", "Tr-pi_1160.jpg", "Tr-pi_1165.jpg", "Tr-pi_1191.jpg", "Tr-pi_1239.jpg"]
    };

    const sampleModal = document.getElementById("sampleModal");
    const openCatalogBtn = document.getElementById("openCatalogBtn");
    const closeModal = document.getElementById("closeModal");
    const sampleGrid = document.getElementById("sampleGrid");
    const tabButtons = document.querySelectorAll(".tab-btn");

    openCatalogBtn.addEventListener("click", () => {
        sampleModal.classList.remove("hidden");
        loadCategory("glioma"); // Default category
    });

    closeModal.addEventListener("click", () => {
        sampleModal.classList.add("hidden");
    });

    // Close when clicking outside
    window.addEventListener("click", (e) => {
        if (e.target === sampleModal) sampleModal.classList.add("hidden");
    });

    tabButtons.forEach(btn => {
        btn.addEventListener("click", () => {
            tabButtons.forEach(b => b.classList.remove("active"));
            btn.classList.add("active");
            loadCategory(btn.dataset.category);
        });
    });

    function loadCategory(category) {
        sampleGrid.innerHTML = "";
        const images = SAMPLE_IMAGES[category];
        images.forEach(filename => {
            const item = document.createElement("div");
            item.className = "sample-item";
            const imgUrl = `${SAMPLE_REPO}/${category}/${filename}`;
            
            item.innerHTML = `
                <img src="${imgUrl}" alt="${filename}" loading="lazy">
                <div class="sample-name">${filename}</div>
            `;
            
            item.addEventListener("click", () => {
                selectSample(imgUrl, filename);
            });
            sampleGrid.appendChild(item);
        });
    }

    async function selectSample(url, filename) {
        sampleModal.classList.add("hidden");
        analyzeBtn.disabled = true;
        analyzeBtn.querySelector('span').innerText = 'Downloading Sample...';
        
        try {
            const response = await fetch(url);
            const blob = await response.blob();
            const file = new File([blob], filename, { type: "image/jpeg" });
            handleFile(file);
        } catch (err) {
            console.error("Failed to load sample:", err);
            alert("Failed to load sample image from GitHub.");
        } finally {
            analyzeBtn.disabled = false;
            analyzeBtn.querySelector('span').innerText = 'Generate Analysis';
        }
    }
    // ----------------------------

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
            const response = await fetch(API_URL, {
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
            alert("Error: " + error.message + "\nMake sure your Hugging Face Space is up and RUNNING.");
        } finally {
            analyzeBtn.disabled = false;
            analyzeBtn.querySelector('span').innerText = 'Generate Analysis';
            loadingSpinner.classList.add("hidden");
        }
    });

    // Global download function
    window.downloadImage = function(imgId, fileName) {
        const img = document.getElementById(imgId);
        if (!img || !img.src) return;

        const link = document.createElement("a");
        link.href = img.src;
        link.download = `${fileName}_Result.jpg`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    };

});
