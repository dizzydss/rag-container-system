# RAG Container System

This project implements a Retrieval-Augmented Generation (RAG) system using a containerized architecture. It allows users to ingest PDF documents, store their embeddings in a vector database (ChromaDB), and query the information using a Large Language Model (LLM) served via `llama.cpp`.

## Architecture

The system consists of two main components running in Docker containers:

1.  **`rag-app`**: A FastAPI application that handles:
    *   **Data Ingestion**: Loading PDF documents from a local directory, splitting them into chunks, and generating embeddings.
    *   **Vector Storage**: Interacting with ChromaDB to store and retrieve document embeddings.
    *   **RAG Pipeline**: Orchestrating the retrieval and generation process using LangChain.
    *   **API Endpoints**: Providing `/ingest` and `/query` endpoints.
2.  **`chroma-db`**: A ChromaDB instance used as the persistent vector store for document embeddings.

The LLM itself is run on the **host machine** via `llama.cpp` (or similar OpenAI-compatible servers) to leverage host hardware (GPU/CPU) efficiently. The `rag-app` container communicates with the host using `host.docker.internal`.

## Prerequisites

*   [Docker](https://docs.docker.com/get-docker/) and [Docker Compose](https://docs.docker.com/compose/install/)
*   An OpenAI-compatible LLM server running on your host machine (e.g., `llama.cpp` server) listening on port `8080`.
*   **Firewall Configuration**: Ensure that your host firewall (e.g., `ufw` on Linux) allows incoming connections on the LLM server's port (default `8080`) from the Docker network.

## Setup and Installation

### 1. Start the LLM Server
Run your LLM server on the host machine. For example, using `llama.cpp`:

```bash
./llama-server -m path/to/your/model.gguf --port 8080 --host 0.0.0.0
```
*Note: Ensure the server is listening on `0.0.0.0` so it can accept connections from the Docker container.*

### 2. Configure the Firewall
If you are using `ufw` on Linux, allow traffic on the LLM port:

```bash
sudo ufw allow 8080/tcp
```

### 3. Launch the Containers
From the project root, run:

```bash
docker compose up --build
```

## Usage

### 1. Ingest Documents
Place your PDF documents in the `./data` directory on your host machine. Then, trigger the ingestion process by sending a POST request to the `/ingest` endpoint:

```bash
curl -X POST http://localhost:8000/ingest
```

### 2. Query the System
Once ingestion is complete, you can ask questions about your documents via the `/query` endpoint:

```bash
curl -X POST http://localhost:8000/query \
     -H "Content-Type: application/json" \
     -d '{"query": "What is the main topic of the documents?"}'
```

## API Reference

### `POST /ingest`
*   **Description**: Scans the `/app/data` directory for PDFs and populates the vector store.
*   **Response**: `{"message": "Successfully ingested X documents."}`

### `POST /query`
*   **Description**: Performs a RAG query.
*   **Request Body**:
    ```json
    {
      "query": "your question here"
    }
    ```
*   **Response**: `{"answer": "The response from the LLM..."}`

## Troubleshooting

*   **Connection Timeout**: If `/query` hangs, ensure the LLM server is running on the host and that `ufw` allows port 8080.
*   **Host Connectivity**: The `rag-app` uses `host.docker.internal` to reach the host. This is enabled in `docker-compose.yml` via the `extra_hosts` directive.
