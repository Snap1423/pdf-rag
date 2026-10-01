
from dotenv import load_dotenv
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# Load the model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load FAISS database
vector_store = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

# Create a retriever
retriever = vector_store.as_retriever(
    search_kwargs={"k": 3}
)

# Create LLM
llm = ChatGoogleGenerativeAI(
    model=os.getenv("GEMINI_MODEL", "gemini-3.8-flash"),
    temperature=0,
    request_timeout=float(os.getenv("GEMINI_TIMEOUT_SECONDS", "60")),
    max_retries=1,
)

def ask_question(query):

    # Retrieve relevant chunks with scores
    results = vector_store.similarity_search_with_score(
        query,
        k=3
    )

    # Check relevance
    if not results or results[0][1] > 1.0:
        return "I don't know based on the provided document.", []

    # Extract documents
    documents = [document for document, score in results]

    # Combine retrieved chunks
    context = "\n\n".join(
        document.page_content
        for document in documents
    )

    # Create prompt
    prompt = f"""
    Answer the question using only the context provided below.

    If the answer cannot be found in the context, say:
    "I don't know based on the provided document."

    Context:
    {context}

    Question:
    {query}

    Answer:
    """

    # Generate answer
    response = llm.invoke(prompt)

    # Extract source pages
    sources = set()

    for document in documents:
        page = document.metadata.get("page")

        if page is not None:
            sources.add(page + 1)

    return response.text, sorted(sources)