# PDF RAG Question Answering System

A Retrieval-Augmented Generation (RAG) application that allows users to ask questions about PDF documents and receive context-aware answers using Google's Gemini API.

Built using LangChain, Hugging Face embeddings, and FAISS, with a command-line interface.

## Features

* Load and process PDF documents.
* Split text into smaller chunks.
* Generate semantic embeddings using Hugging Face.
* Store and retrieve embeddings using FAISS.
* Generate context-grounded answers using Gemini.
* Display source page references.
* Support multiple questions in a single session.

## Tech Stack

* Python
* LangChain
* PyPDF
* Hugging Face Sentence Transformers
* FAISS
* Google Gemini API
* python-dotenv

## Architecture

```text
PDF Document
     |
     v
Document Loading
     |
     v
Text Chunking
     |
     v
Hugging Face Embeddings
     |
     v
FAISS Vector Store
     |
     v
User Question
     |
     v
Similarity Search
     |
     v
Relevant Chunks
     |
     v
Gemini API
     |
     v
Generated Answer
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/pdf-rag.git
cd pdf-rag
```

### 2. Create and activate a virtual environment

**Windows:**

```powershell
python -m venv .venv
.venv\Scripts\activate
```

**Linux / macOS:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure the Gemini API

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL=gemini-3.8-flash
GEMINI_TIMEOUT_SECONDS=60
```

Get your API key from [Google AI Studio](https://aistudio.google.com/apikey).

**Never upload your `.env` file or expose your API key publicly.**

## Usage

Place your PDF inside the `data/` directory. The default path is:

```text
data/sample.pdf
```

Generate the FAISS index:

```bash
python ingest.py
```

Start the application:

```bash
python app.py
```

Ask questions through the command-line interface. Type `exit` to quit.

## Configuration

| Parameter       | Value                                  |
| --------------- | -------------------------------------- |
| Embedding Model | sentence-transformers/all-MiniLM-L6-v2 |
| Chunk Size      | 1000                                   |
| Chunk Overlap   | 200                                    |
| Retrieval Top-K | 3                                      |
| LLM             | Google Gemini                          |
| Temperature     | 0                                      |

## Limitations

* Uses a predefined PDF path.
* Supports one indexed document at a time.
* Requires a Gemini API key for answer generation.
* Embedding model requires an initial download.

## Future Improvements

* Multiple PDF support.
* Conversation history.
* Improved retrieval using reranking.
* Web-based interface.
* Support for additional LLM providers.

## License

This project is intended for educational and learning purposes.
