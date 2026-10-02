import hashlib
import re
import tempfile
from io import BytesIO
from pathlib import Path

import faiss
import gdown
import numpy as np
import streamlit as st
from docx import Document
from groq import Groq
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


# -----------------------------
# App settings
# -----------------------------
st.set_page_config(
    page_title="AI Document Assistant",
    page_icon="📚",
    layout="wide",
)

st.title("📚 AI Document Assistant")
st.caption(
    "Upload documents or load public Google Drive files, then ask questions "
    "using semantic + keyword search."
)

SUPPORTED_TYPES = [".pdf", ".docx", ".txt", ".md"]
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
GROQ_MODEL = "openai/gpt-oss-120b"


# -----------------------------
# Text extraction
# -----------------------------
def extract_pdf(file_bytes):
    """Extract PDF text and keep the PDF page number."""
    documents = []
    reader = PdfReader(BytesIO(file_bytes))

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            documents.append(
                {
                    "text": text.strip(),
                    "page": page_number,
                }
            )

    return documents


def extract_docx(file_bytes):
    """Extract DOCX paragraphs. DOCX does not expose reliable page numbers here."""
    document = Document(BytesIO(file_bytes))
    text = "\n".join(
        paragraph.text for paragraph in document.paragraphs if paragraph.text.strip()
    )

    return [{"text": text.strip(), "page": None}] if text.strip() else []


def extract_txt(file_bytes):
    """Extract plain TXT text."""
    text = file_bytes.decode("utf-8", errors="ignore").strip()
    return [{"text": text, "page": None}] if text else []


def extract_md(file_bytes):
    """Extract Markdown text as plain text."""
    text = file_bytes.decode("utf-8", errors="ignore").strip()
    return [{"text": text, "page": None}] if text else []


def extract_document(filename, file_bytes):
    """Choose the correct extractor from the file extension."""
    extension = Path(filename).suffix.lower()

    if extension == ".pdf":
        return extract_pdf(file_bytes)

    if extension == ".docx":
        return extract_docx(file_bytes)

    if extension == ".txt":
        return extract_txt(file_bytes)

    if extension == ".md":
        return extract_md(file_bytes)

    raise ValueError(f"Unsupported file type: {extension}")


# Cache extraction results for identical file bytes.
@st.cache_data(show_spinner=False)
def cached_extract_document(filename, file_bytes):
    return extract_document(filename, file_bytes)


# -----------------------------
# Chunking
# -----------------------------
def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into overlapping word-based chunks."""
    words = text.split()

    if not words:
        return []

    step = max(1, chunk_size - overlap)
    chunks = []

    for start in range(0, len(words), step):
        chunk = " ".join(words[start:start + chunk_size])

        if chunk.strip():
            chunks.append(chunk.strip())

        if start + chunk_size >= len(words):
            break

    return chunks


def create_chunks(extracted_documents):
    """Create chunks while preserving filename and page metadata."""
    all_chunks = []

    for document in extracted_documents:
        for chunk in chunk_text(document["text"]):
            all_chunks.append(
                {
                    "text": chunk,
                    "filename": document["filename"],
                    "page": document["page"],
                }
            )

    return all_chunks


# -----------------------------
# Embeddings + FAISS
# -----------------------------
@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)


def build_faiss_index(chunks):
    """Create document embeddings once and store them in a FAISS index."""
    model = load_embedding_model()

    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    return index, embeddings


# -----------------------------
# Keyword search
# -----------------------------
STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "how", "i", "in", "is", "it", "of", "on", "or", "that", "the",
    "this", "to", "was", "what", "when", "where", "which", "who",
    "why", "with", "you", "your",
}


def important_words(text):
    """Return simple keywords from a question."""
    words = re.findall(r"\b[a-zA-Z0-9]+\b", text.lower())
    return [word for word in words if word not in STOP_WORDS and len(word) > 2]


def keyword_scores(question, chunks):
    """Score chunks by how many important question words they contain."""
    question_words = set(important_words(question))
    scores = []

    for chunk in chunks:
        chunk_words = set(important_words(chunk["text"]))
        if not question_words:
            scores.append(0.0)
        else:
            scores.append(len(question_words & chunk_words) / len(question_words))

    return np.array(scores, dtype="float32")


# -----------------------------
# Hybrid search
# -----------------------------
def hybrid_search(question, chunks, index, top_k=5):
    """Combine semantic similarity and keyword matching."""
    if not chunks:
        return []

    model = load_embedding_model()

    question_embedding = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    # Get more candidates from semantic search so keyword ranking can help.
    candidate_k = min(len(chunks), max(top_k * 4, 10))
    semantic_scores, semantic_ids = index.search(question_embedding, candidate_k)

    keyword = keyword_scores(question, chunks)

    candidates = []
    for rank, chunk_id in enumerate(semantic_ids[0]):
        if chunk_id < 0:
            continue

        semantic_score = float(semantic_scores[0][rank])
        keyword_score = float(keyword[chunk_id])

        # 70% semantic + 30% keyword
        hybrid_score = (0.70 * semantic_score) + (0.30 * keyword_score)

        candidates.append(
            {
                "text": chunks[chunk_id]["text"],
                "filename": chunks[chunk_id]["filename"],
                "page": chunks[chunk_id]["page"],
                "semantic_score": semantic_score,
                "keyword_score": keyword_score,
                "hybrid_score": hybrid_score,
            }
        )

    candidates.sort(key=lambda item: item["hybrid_score"], reverse=True)
    return candidates[:top_k]


# -----------------------------
# Groq
# -----------------------------
def get_groq_client():
    """Read the API key from Streamlit secrets."""
    try:
        api_key = st.secrets["GROQ_API_KEY"]
    except Exception:
        raise RuntimeError(
            "GROQ_API_KEY is missing. Add it to Streamlit Secrets."
        )

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is empty. Add your Groq API key to Streamlit Secrets."
        )

    return Groq(api_key=api_key)


def answer_question(question, retrieved_chunks):
    """Ask Groq to answer only from retrieved document context."""
    context_parts = []

    for number, chunk in enumerate(retrieved_chunks, start=1):
        page_text = (
            f"Page {chunk['page']}"
            if chunk["page"] is not None
            else "Page not available"
        )

        context_parts.append(
            f"[Source {number}]\n"
            f"Filename: {chunk['filename']}\n"
            f"{page_text}\n"
            f"Text:\n{chunk['text']}"
        )

    context = "\n\n".join(context_parts)

    system_prompt = """You are an AI document assistant.

Answer the user's question ONLY from the supplied document context.

Rules:
1. Do not use outside knowledge.
2. If the answer is not available in the context, say:
   "The information is not available in the provided documents."
3. Do not invent facts, sources, page numbers, or details.
4. Give a clear and concise answer.
"""

    client = get_groq_client()

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"DOCUMENT CONTEXT:\n{context}\n\nQUESTION:\n{question}",
            },
        ],
    )

    return response.choices[0].message.content


# -----------------------------
# Google Drive
# -----------------------------
def load_drive_files(url):
    """
    Download a public Google Drive file or folder.

    gdown handles public Drive links. Folder links can download multiple files.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        if "/folders/" in url:
            downloaded = gdown.download_folder(
                url=url,
                output=str(temp_path),
                quiet=True,
                use_cookies=False,
                remaining_ok=True,
            )

            if not downloaded:
                return []

            paths = [Path(item) for item in downloaded]
        else:
            output_dir = str(temp_path) + "/"
            downloaded = gdown.download(
                url=url,
                output=output_dir,
                quiet=True,
                use_cookies=False,
            )

            if not downloaded:
                return []

            paths = [Path(downloaded)]

        files = []

        for path in paths:
            if path.is_file() and path.suffix.lower() in SUPPORTED_TYPES:
                files.append(
                    {
                        "name": path.name,
                        "bytes": path.read_bytes(),
                    }
                )

        return files


# -----------------------------
# Processing
# -----------------------------
def process_files(file_items):
    """Extract and chunk all supplied files."""
    extracted = []
    document_info = []

    for item in file_items:
        filename = item["name"]
        file_bytes = item["bytes"]

        pages_or_sections = cached_extract_document(filename, file_bytes)

        if not pages_or_sections:
            continue

        for part in pages_or_sections:
            extracted.append(
                {
                    "filename": filename,
                    "page": part["page"],
                    "text": part["text"],
                }
            )

        document_info.append(
            {
                "filename": filename,
                "size_kb": round(len(file_bytes) / 1024, 2),
                "pages_or_sections": len(pages_or_sections),
            }
        )

    chunks = create_chunks(extracted)

    if not chunks:
        return document_info, chunks, None, None

    index, embeddings = build_faiss_index(chunks)

    return document_info, chunks, index, embeddings


def make_file_signature(file_items):
    """Create a stable signature so the same documents are not processed again."""
    hasher = hashlib.sha256()

    for item in sorted(file_items, key=lambda x: x["name"]):
        hasher.update(item["name"].encode("utf-8"))
        hasher.update(item["bytes"])

    return hasher.hexdigest()


# -----------------------------
# Session state
# -----------------------------
defaults = {
    "file_items": [],
    "processed_signature": None,
    "document_info": [],
    "chunks": [],
    "faiss_index": None,
    "embeddings": None,
    "drive_files": [],
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    top_k = st.slider("Retrieved chunks", min_value=1, max_value=10, value=5)
    st.caption(f"Chunk size: {CHUNK_SIZE} words")
    st.caption(f"Chunk overlap: {CHUNK_OVERLAP} words")
    st.caption(f"Embedding model: {EMBEDDING_MODEL}")
    st.caption(f"Groq model: {GROQ_MODEL}")


# -----------------------------
# Document sources
# -----------------------------
st.subheader("1. Add documents")

uploaded_files = st.file_uploader(
    "Upload PDF, DOCX, TXT or MD files",
    type=["pdf", "docx", "txt", "md"],
    accept_multiple_files=True,
)

drive_url = st.text_input(
    "Google Drive file or folder link",
    placeholder="Paste a public/shared Google Drive link",
)

col1, col2 = st.columns(2)

with col1:
    if st.button("➕ Add local uploads", use_container_width=True):
        if not uploaded_files:
            st.warning("Please select at least one local document.")
        else:
            local_files = [
                {
                    "name": file.name,
                    "bytes": file.getvalue(),
                }
                for file in uploaded_files
            ]

            existing_names = {
                item["name"] for item in st.session_state.file_items
            }

            for item in local_files:
                if item["name"] not in existing_names:
                    st.session_state.file_items.append(item)

            st.success(f"Added {len(local_files)} local file(s).")

with col2:
    if st.button("☁️ Load Google Drive", use_container_width=True):
        if not drive_url.strip():
            st.warning("Please paste a Google Drive file or folder link.")
        else:
            try:
                with st.spinner("Loading Google Drive files..."):
                    drive_files = load_drive_files(drive_url.strip())

                if not drive_files:
                    st.warning(
                        "No supported PDF, DOCX, TXT or MD files were found. "
                        "Make sure the Drive file/folder is publicly accessible."
                    )
                else:
                    existing_names = {
                        item["name"] for item in st.session_state.file_items
                    }

                    for item in drive_files:
                        if item["name"] not in existing_names:
                            st.session_state.file_items.append(item)

                    st.session_state.drive_files = drive_files
                    st.success(f"Loaded {len(drive_files)} supported file(s).")

            except Exception as error:
                st.error(f"Google Drive loading failed: {error}")


# -----------------------------
# Current documents
# -----------------------------
if st.session_state.file_items:
    st.subheader("2. Selected documents")

    for item in st.session_state.file_items:
        st.write(f"📄 {item['name']}")

    if st.button("🧠 Process documents", type="primary", use_container_width=True):
        signature = make_file_signature(st.session_state.file_items)

        if signature == st.session_state.processed_signature:
            st.info("These documents are already processed. Reusing the existing embeddings.")
        else:
            with st.spinner("Extracting text, creating chunks and embeddings..."):
                (
                    document_info,
                    chunks,
                    index,
                    embeddings,
                ) = process_files(st.session_state.file_items)

            st.session_state.document_info = document_info
            st.session_state.chunks = chunks
            st.session_state.faiss_index = index
            st.session_state.embeddings = embeddings
            st.session_state.processed_signature = signature

            st.success("Documents processed successfully.")

# -----------------------------
# Document information
# -----------------------------
if st.session_state.document_info:
    st.subheader("3. Extracted document information")

    st.dataframe(
        st.session_state.document_info,
        use_container_width=True,
        hide_index=True,
    )

    st.metric(
        "Created chunks",
        len(st.session_state.chunks),
    )

# -----------------------------
# Question answering
# -----------------------------
if st.session_state.faiss_index is not None:
    st.subheader("4. Ask a question")

    question = st.text_input(
        "Your question",
        placeholder="Example: What is the main purpose of this document?",
    )

    if st.button("🔎 Ask", type="primary", use_container_width=True):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            try:
                with st.spinner("Searching documents and generating answer..."):
                    results = hybrid_search(
                        question,
                        st.session_state.chunks,
                        st.session_state.faiss_index,
                        top_k=top_k,
                    )

                    answer = answer_question(question, results)

                st.subheader("Answer")
                st.write(answer)

                st.subheader("Retrieved sources")

                for number, result in enumerate(results, start=1):
                    page = (
                        f"Page {result['page']}"
                        if result["page"] is not None
                        else "Page not available"
                    )

                    with st.expander(
                        f"Source {number} — {result['filename']} — {page}"
                    ):
                        st.caption(
                            f"Hybrid score: {result['hybrid_score']:.3f} | "
                            f"Semantic: {result['semantic_score']:.3f} | "
                            f"Keyword: {result['keyword_score']:.3f}"
                        )
                        st.write(result["text"])

            except Exception as error:
                st.error(f"Could not generate the answer: {error}")

else:
    st.info("Add documents and click **Process documents** to start asking questions.")
