from pypdf import PdfReader
import io


# =====================================================
# 📌 Extract PDF text
# =====================================================
def extract_text(content, file_type="pdf"):
    if file_type == "pdf":
        reader = PdfReader(io.BytesIO(content))
        text = ""

        for page in reader.pages:
            text += page.extract_text() or ""

        return text


# =====================================================
# 📌 Chunking text
# =====================================================
def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []

    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i + chunk_size])

    return chunks