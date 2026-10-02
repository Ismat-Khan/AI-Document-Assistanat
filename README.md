# 📚 AI Document Assistant

A simple Streamlit RAG application that lets you upload documents or load public Google Drive files and ask questions about them.

## Features

- PDF extraction with page numbers
- DOCX extraction
- TXT extraction
- Markdown (MD) extraction
- Overlapping text chunks
- Sentence Transformers embeddings
- FAISS semantic search
- Simple keyword search
- Hybrid semantic + keyword ranking
- Groq-powered answers
- Retrieved sources shown after every answer
- Public Google Drive file/folder loading
- Streamlit session state so processed documents are reused
- Streamlit cache for the embedding model and extraction
- Groq API key stored in Streamlit Secrets

## Project files

```text
ai_document_assistant/
├── app.py
├── requirements.txt
└── README.md
```

## 1. Install

Use Python 3.10+.

```bash
pip install -r requirements.txt
```

## 2. Add your Groq API key

Create this file:

```text
.streamlit/secrets.toml
```

Put this inside it:

```toml
GROQ_API_KEY = "your-groq-api-key"
```

Do not commit `secrets.toml` to GitHub.

For Streamlit Community Cloud, add the same value in your app's Secrets settings.

## 3. Run

```bash
streamlit run app.py
```

## 4. How the app works

### Step 1 — Add documents

Upload:

- PDF
- DOCX
- TXT
- MD

You can also paste a public/shared Google Drive file or folder link.

Google Drive support uses `gdown`. Only supported PDF, DOCX, TXT and MD files are sent into the document pipeline.

### Step 2 — Extract text

The app uses separate functions:

```text
extract_pdf()
extract_docx()
extract_txt()
extract_md()
```

PDF extraction keeps the page number.

DOCX, TXT and MD keep the filename but use `None` for page because these formats do not provide a reliable page number through this simple extraction pipeline.

### Step 3 — Chunk text

The extracted text is split into overlapping chunks.

Current defaults:

```text
Chunk size: 900 words
Overlap: 150 words
```

Every chunk keeps:

```text
filename
page
text
```

### Step 4 — Create embeddings

The app uses:

```text
all-MiniLM-L6-v2
```

Each document chunk is converted into a vector.

The vectors are stored in the app's session state so they do not need to be recreated for every question.

### Step 5 — FAISS search

The question is converted into an embedding and compared with the document vectors using FAISS.

The most relevant semantic candidates are selected.

### Step 6 — Keyword search

Important words from the question are compared with words in the chunks.

This creates a simple keyword score.

### Step 7 — Hybrid search

The final ranking combines:

```text
70% semantic similarity
30% keyword matching
```

This helps the search use both meaning and exact important words.

### Step 8 — Groq answer

The retrieved chunks are sent to Groq together with the question.

The system instruction tells the model:

- answer only from the provided context
- do not use outside knowledge
- do not invent facts
- say when the answer is not available

The current model in `app.py` is:

```text
openai/gpt-oss-120b
```

### Step 9 — Sources

After every answer, the app shows the retrieved chunks with:

- filename
- page number when available
- retrieved text
- semantic score
- keyword score
- hybrid score

## Google Drive notes

The Google Drive source is designed for public/shared links that `gdown` can access.

For a folder link, the app downloads the folder and filters the downloaded files to:

```text
.pdf
.docx
.txt
.md
```

Private files that require an account login may not be accessible through this simple public-link approach.

## Important caching behavior

The app uses:

```text
st.cache_resource
```

for the Sentence Transformers model.

It uses:

```text
st.cache_data
```

for document extraction.

It uses:

```text
st.session_state
```

for:

- processed documents
- chunks
- FAISS index
- embeddings
- document information

The app also creates a document signature. If the same documents are processed again, it reuses the existing embeddings instead of rebuilding them.

## Security

Never hardcode your Groq API key in `app.py`.

Use:

```toml
GROQ_API_KEY = "your-groq-api-key"
```

inside Streamlit Secrets.

Also add `.streamlit/secrets.toml` to `.gitignore` if you work locally.

## Simple architecture

```text
Local Uploads / Google Drive
            ↓
      Text Extraction
            ↓
        Chunking
            ↓
   Sentence Transformers
            ↓
       FAISS Index
            ↓
   ┌───────────────────┐
   │ Hybrid Search     │
   │ Semantic + Keyword│
   └───────────────────┘
            ↓
     Relevant Chunks
            ↓
        Groq LLM
            ↓
     Answer + Sources
```

## Deploy on Streamlit Community Cloud

1. Create a GitHub repository.
2. Upload:
   - `app.py`
   - `requirements.txt`
   - `README.md`
3. Open Streamlit Community Cloud.
4. Create a new app and select the repository.
5. Set the main file to `app.py`.
6. Open the app's Secrets settings.
7. Add:

```toml
GROQ_API_KEY = "your-groq-api-key"
```

8. Deploy.

Do not upload your API key to GitHub.
