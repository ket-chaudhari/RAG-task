const API_URL = "http://127.0.0.1:8000";

const fileInput = document.getElementById("fileInput");
const browseButton = document.getElementById("browseButton");
const dropZone = document.getElementById("dropZone");

const selectedFile = document.getElementById("selectedFile");
const fileName = document.getElementById("fileName");
const fileSize = document.getElementById("fileSize");
const removeFile = document.getElementById("removeFile");

const uploadButton = document.getElementById("uploadButton");

const processing = document.getElementById("processing");
const successMessage = document.getElementById("successMessage");

const questionInput = document.getElementById("questionInput");
const askButton = document.getElementById("askButton");
const characterCount = document.getElementById("characterCount");

const answerCard = document.getElementById("answerCard");
const answerPlaceholder = document.getElementById("answerPlaceholder");
const answerResult = document.getElementById("answerResult");

const contextSection = document.getElementById("contextSection");
const contextToggle = document.getElementById("contextToggle");
const contextContent = document.getElementById("contextContent");


let currentFile = null;
let documentReady = false;


/* ---------------- FILE SELECT ---------------- */

browseButton.addEventListener("click", () => {
    fileInput.click();
});


fileInput.addEventListener("change", () => {

    if (fileInput.files.length > 0) {
        setFile(fileInput.files[0]);
    }

});


/* ---------------- DRAG DROP ---------------- */

dropZone.addEventListener("dragover", (event) => {

    event.preventDefault();

    dropZone.classList.add("dragover");

});


dropZone.addEventListener("dragleave", () => {

    dropZone.classList.remove("dragover");

});


dropZone.addEventListener("drop", (event) => {

    event.preventDefault();

    dropZone.classList.remove("dragover");

    const files = event.dataTransfer.files;

    if (files.length > 0) {
        setFile(files[0]);
    }

});


/* ---------------- SET FILE ---------------- */

function setFile(file) {

    const extension = file.name
        .split(".")
        .pop()
        .toLowerCase();

    if (extension !== "pdf" && extension !== "txt") {

        alert("Please select a PDF or TXT file.");

        return;
    }


    currentFile = file;

    fileName.textContent = file.name;

    fileSize.textContent = formatFileSize(file.size);

    selectedFile.style.display = "flex";

    successMessage.style.display = "none";

    documentReady = false;

}


/* ---------------- REMOVE FILE ---------------- */

removeFile.addEventListener("click", () => {

    currentFile = null;

    fileInput.value = "";

    selectedFile.style.display = "none";

    successMessage.style.display = "none";

    documentReady = false;

});


/* ---------------- FILE SIZE ---------------- */

function formatFileSize(bytes) {

    if (bytes < 1024) {
        return bytes + " B";
    }

    if (bytes < 1024 * 1024) {
        return (bytes / 1024).toFixed(1) + " KB";
    }

    return (bytes / (1024 * 1024)).toFixed(1) + " MB";
}


/* ---------------- UPLOAD ---------------- */

uploadButton.addEventListener("click", async () => {

    if (!currentFile) {

        alert("Please select a document first.");

        return;
    }


    const formData = new FormData();

    formData.append("file", currentFile);


    uploadButton.disabled = true;

    uploadButton.querySelector("span:first-child").textContent =
        "Processing...";

    processing.style.display = "flex";

    successMessage.style.display = "none";


    try {

        const response = await fetch(
            `${API_URL}/upload`,
            {
                method: "POST",
                body: formData
            }
        );


        const data = await response.json();


        if (!response.ok || data.error) {

            throw new Error(
                data.error || "Upload failed."
            );
        }


        /*
         * Backend uses BackgroundTasks.
         * The upload response means processing has started.
         */

        documentReady = true;


        processing.style.display = "none";

        successMessage.style.display = "flex";


        uploadButton.querySelector("span:first-child").textContent =
            "Document Ready";


        /*
         * Enable question section visually
         */

        questionInput.focus();


    } catch (error) {

        processing.style.display = "none";

        alert(
            "Could not process document.\n\n" +
            error.message
        );

        uploadButton.disabled = false;

        uploadButton.querySelector("span:first-child").textContent =
            "Process Document";
    }

});


/* ---------------- CHARACTER COUNT ---------------- */

questionInput.addEventListener("input", () => {

    const count = questionInput.value.length;

    characterCount.textContent =
        `${count} characters`;

});


/* ---------------- ASK QUESTION ---------------- */

askButton.addEventListener("click", askQuestion);


questionInput.addEventListener("keydown", (event) => {

    if (
        event.key === "Enter" &&
        !event.shiftKey
    ) {

        event.preventDefault();

        askQuestion();
    }

});


async function askQuestion() {

    const question =
        questionInput.value.trim();


    if (!question) {

        alert("Please enter a question.");

        questionInput.focus();

        return;
    }


    if (!documentReady) {

        alert(
            "Please upload and process a document first."
        );

        return;
    }


    askButton.disabled = true;

    askButton.innerHTML =
        "Thinking <span>◌</span>";


    answerPlaceholder.style.display = "none";

    answerResult.style.display = "block";

    answerResult.textContent =
        "Searching your document...";


    try {

        const response = await fetch(
            `${API_URL}/ask`,
            {
                method: "POST",

                headers: {
                    "Content-Type": "application/json"
                },

                body: JSON.stringify({
                    question: question
                })
            }
        );


        const data = await response.json();


        if (!response.ok || data.error) {

            throw new Error(
                data.error || "Failed to get answer."
            );
        }


        answerResult.textContent =
            data.answer || "No answer generated.";


        /*
         * Show retrieved context
         */

        if (
            data.context_used &&
            data.context_used.length > 0
        ) {

            contextSection.style.display = "block";

            contextContent.textContent =
                data.context_used.join("\n\n---\n\n");

        }


        /*
         * Scroll answer into view
         */

        answerCard.scrollIntoView({
            behavior: "smooth",
            block: "center"
        });


    } catch (error) {

        answerResult.textContent =
            "Error: " + error.message;

    } finally {

        askButton.disabled = false;

        askButton.innerHTML =
            'Ask AI <span>✦</span>';

    }

}


/* ---------------- CONTEXT TOGGLE ---------------- */

contextToggle.addEventListener("click", () => {

    const visible =
        contextContent.style.display === "block";


    contextContent.style.display =
        visible ? "none" : "block";

});
