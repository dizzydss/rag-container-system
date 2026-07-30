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

## Zed Integration (via `rag-proxy`)

To use this system directly within the Zed editor's Assistant Panel, we use a dedicated proxy server (`rag-proxy`). This allows Zed to interact with the RAG system as if it were a standard OpenAI-compatible API.

### How it works

The `rag-proxy` acts as an intelligent middleman between Zed and the backend components:

1.  **Intercept**: Zed sends a standard OpenAI-format chat completion request to the `rag-proxy` (running on port `8001`).
2.  **Retrieve**: The proxy intercepts the user's prompt and queries the `rag-app` (via the `/query` endpoint) to retrieve relevant context from the vector database.
3.  **Augment**: The proxy enriches the original prompt by prepending the retrieved context.
4.  **Generate**: The augmented prompt is then sent to the LLM (running on the host machine) to generate a response that incorporates your private documents.
5.  **Respond**: The proxy formats the final answer into a strictly compliant OpenAI-style JSON response and returns it to Zed.

### Configuration

To connect Zed to your local RAG system, update your Zed `settings.json` with the following configuration:

```json
{
  "language_models": {
    "openai": {
      "api_url": "http://localhost:8001/v1"
    }
  }
}
```

Once configured, you can ask questions in the Zed Assistant Panel that leverage the knowledge contained within your ingested documents.

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
