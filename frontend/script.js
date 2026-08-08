const API_URL = "http://127.0.0.1:8000";

const fileInput = document.getElementById("fileInput");
const uploadBtn = document.getElementById("uploadBtn");
const askBtn = document.getElementById("askBtn");

const fileName = document.getElementById("fileName");
const uploadStatus = document.getElementById("uploadStatus");

const questionInput = document.getElementById("questionInput");
const chatMessages = document.getElementById("chatMessages");

const documentStatus =
    document.getElementById("documentStatus");


let selectedFile = null;


// ================================
// FILE SELECTION
// ================================

fileInput.addEventListener("change", function () {

    selectedFile = fileInput.files[0];

    if (!selectedFile) {
        fileName.textContent = "No file selected";
        return;
    }

    fileName.textContent = selectedFile.name;
});


// ================================
// UPLOAD
// ================================

uploadBtn.addEventListener("click", async function () {

    if (!selectedFile) {

        uploadStatus.textContent =
            "Please select a PDF or TXT file.";

        return;
    }

    const formData = new FormData();

    formData.append("file", selectedFile);

    uploadBtn.disabled = true;

    uploadStatus.textContent =
        "Uploading and processing document...";

    try {

        const response = await fetch(
            `${API_URL}/upload`,
            {
                method: "POST",
                body: formData
            }
        );

        const data = await response.json();

        if (!response.ok) {
            throw new Error(
                data.detail || data.error || "Upload failed"
            );
        }

        uploadStatus.textContent =
            "✓ Document uploaded successfully.";

        documentStatus.textContent = "Document ready";

        documentStatus.style.background = "#dcfce7";
        documentStatus.style.color = "#166534";

    } catch (error) {

        uploadStatus.textContent =
            "❌ " + error.message;

    } finally {

        uploadBtn.disabled = false;
    }
});


// ================================
// ASK QUESTION
// ================================

askBtn.addEventListener("click", askQuestion);


questionInput.addEventListener("keydown", function (event) {

    if (event.key === "Enter" && !event.shiftKey) {

        event.preventDefault();

        askQuestion();
    }
});


async function askQuestion() {

    const question =
        questionInput.value.trim();

    if (!question) {
        return;
    }

    addMessage(
        question,
        "user"
    );

    questionInput.value = "";

    askBtn.disabled = true;

    const loading = addMessage(
        "Thinking...",
        "bot"
    );

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

        loading.remove();

        if (!response.ok) {

            throw new Error(
                data.detail ||
                data.error ||
                "Something went wrong"
            );
        }

        if (data.error) {

            addMessage(
                "❌ " + data.error,
                "bot"
            );

        } else {

            addMessage(
                data.answer,
                "bot"
            );
        }

    } catch (error) {

        loading.remove();

        addMessage(
            "❌ " + error.message,
            "bot"
        );

    } finally {

        askBtn.disabled = false;
        questionInput.focus();
    }
}


// ================================
// CHAT MESSAGE
// ================================

function addMessage(text, sender) {

    const welcome =
        document.querySelector(".welcome");

    if (welcome) {
        welcome.remove();
    }

    const message =
        document.createElement("div");

    message.className =
        `message ${sender}`;

    const content =
        document.createElement("div");

    content.className =
        "message-content";

    content.textContent = text;

    message.appendChild(content);

    chatMessages.appendChild(message);

    chatMessages.scrollTop =
        chatMessages.scrollHeight;

    return message;
}
