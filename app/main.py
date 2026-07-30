import os
import traceback
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_classic.chains.retrieval_qa.base import RetrievalQA

app = FastAPI()

# Configuration
CHROMA_PERSIST_DIRECTORY = "/chroma/chroma"
DATA_DIRECTORY = "/app/data"
LLAMA_CPP_URL = os.getenv("LLAMA_CPP_URL", "http://host.docker.internal:8080/v1")

# Initialize Embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Initialize LLM (connecting to llama.cpp)
llm = ChatOpenAI(
    base_url=LLAMA_CPP_URL,
    api_key="not-needed",
    temperature=0.7,
)

# Global variable for the QA chain
qa_chain = None

class QueryRequest(BaseModel):
    query: str

class IngestRequest(BaseModel):
    pass

@app.on_event("startup")
async def startup_event():
    global qa_chain
    
    # Initialize Chroma client
    vectorstore = Chroma(
        persist_directory=CHROMA_PERSIST_DIRECTORY,
        embedding_function=embeddings
    )
    
    # Set up the QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
    )
    print("RAG system initialized.")

@app.post("/ingest")
async def ingest_data():
    """Ingests documents from the /app/data directory."""
    global qa_chain
    
    if not os.path.exists(DATA_DIRECTORY):
        raise HTTPException(status_code=400, detail="Data directory does not exist.")

    try:
        # Load documents
        loader = DirectoryLoader(DATA_DIRECTORY, glob="**/*.pdf", loader_cls=PyPDFLoader)
        documents = loader.load()
        
        if not documents:
            return {"message": "No documents found to ingest."}

        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        texts = text_splitter.split_documents(documents)

        # Add to vector store
        vectorstore = Chroma(
            persist_directory=CHROMA_PERSIST_DIRECTORY,
            embedding_function=embeddings
        )
        vectorstore.add_documents(texts)
        
        # Re-initialize QA chain to ensure it uses the updated vectorstore
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(),
        )
        
        return {"message": f"Successfully ingested {len(documents)} documents."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
async def query_rag(request: QueryRequest):
    """Queries the RAG system."""
    global qa_chain
    if qa_chain is None:
        raise HTTPException(status_code=503, detail="RAG system not initialized.")
    
    try:
        response = qa_chain.invoke(request.query)
        return {"answer": response["result"]}
    except Exception as e:
        print("Error during query:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
