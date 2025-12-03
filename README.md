# DirtyRagger

A working **RAG (Retrieval-Augmented Generation) prototype** with local document embeddings, vector search, web search, and tool integration.

***NOTE: Working on a 2060 RTX with ~6GB of memory!***

---

## Setup

1. Install dependencies:
``` bash
pip install -r requirements.txt
```

2. (Optional) Place your documents in the `documents/` folder.

3. (Optional) Build the database:
```bash
python build_db.py
```
This will embed documents into a FAISS vector store.

---

## Usage

### Run the RAG server
```bash
python rag_server.py
```

### Query locally (terminal client)
```bash
python rag_client.py
```

### Query via Discord bot
```bash
python rag_discord_bot.py
```

---

## Features

- Embeds documents and stores them in a FAISS vector store  
- Retrieves relevant knowledge for questions  
- Supports **web searches** for additional context  
- Tool integration for advanced RAG workflows  

---

## Notes

- `rag_server.py` Runs embedding, reranking, generation, websearch, and tool passes and allows for url queries
- `rag_client.py` allow for local queries
- `rag_discord_bot.py` allow for queries and responses through discord