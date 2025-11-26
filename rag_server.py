from fastapi import FastAPI, HTTPException, Query
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager

from typing import  Any, Callable
import uvicorn

from rag_server_config import RAGConfig
from rag_server_embedder import Embedder
from rag_server_generator import Generator
from rag_server_reranker import QwenReranker
from rag_server_tool_pass import ToolParseAndExecute
from vector_store import PersistentVectorStore

global SERVER_CONFIG
SERVER_CONFIG = RAGConfig()

class RAG:
    def __init__(self):
        self.generator = Generator()
        self.reranker = QwenReranker()
        self.toolparser = ToolParseAndExecute()
        self.embedder = Embedder() 
        self.vector_store = PersistentVectorStore(dim=self.embedder.dim) 

        # Parameters for search/rerank (can be defined in RAGConfig)
        self.initial_k = 20 # Number of results to retrieve from Vector Store
        self.final_k = 3   # Number of top results to pass to the Generator
        
    def rag_retrieve(self, query: str) -> str:
        print(f"Embedding query: {query[:50]}...")
        query_vector = self.embedder.get_embedding(query)

        initial_results = self.vector_store.search(query_vector, k=self.initial_k)
        context_prompt = ""
        if not initial_results:
            print("No documents found in vector store.")
        else:
            docs_text = [res["text"] for res in initial_results]
            rerank_scores = self.reranker.score(query, docs_text)
            scored_results = sorted(
                zip(initial_results, rerank_scores), 
                key=lambda x: x[1], 
                reverse=True
            )
            
            top_score = scored_results[0][1] if scored_results else 0.0
            if top_score > SERVER_CONFIG.RERANKER_MIN_SCORE_THRESHOLD:
                final_context_docs = scored_results[:self.final_k]
                
                context = "\n\n".join([
                    f"Document {i+1} (Source: {res[0].get('source')}): {res[0]['text']}"
                    for i, res in enumerate(final_context_docs)
                ])

                context_prompt = (
                    "Use the following **Context** to help answer the user's request. "
                    "If the context is irrelevant or insufficient, rely on your general knowledge. "
                    "**Context**:\n" + context
                )

                print(f"Retrieved and reranked {len(final_context_docs)} documents.")
        

        messages = [
                {"role": "system", "content": SERVER_CONFIG.QWEN_PERSONALITY_PROMPT + "\n\n" + context_prompt}, 
                {"role": "user", "content": query}                        
            ]
        
        # check for tool use, omit this if too slow (TODO: Add to config as option)
        initial = self.generator.generate(messages=messages, usetools=True)
        parsed_result = self.toolparser.parse_tool_call(initial)

        if parsed_result is None:
            print("No tool call detected, proceeding with initial generation.")
            return initial
        
        # round 2 generation with tool results
        messages = [
                {"role": "system", "content": f"{SERVER_CONFIG.QWEN_PERSONALITY_PROMPT}\n\n {parsed_result}"}, 
                {"role": "user", "content": query}                        
            ]
        
        final_result = self.generator.generate(messages=messages, usetools=False)
        return final_result
    
# ------- Server Setup ----------------------------------------------------------------------------------------------------------------
async def handle_rag_query( query: str, rag_method: Callable[str, Any]) -> JSONResponse:
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    try:
        response_text = await run_in_threadpool(rag_method, query)
        return JSONResponse(
            content={"query": query, "response": response_text},
            media_type="application/json"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error during generation: {str(e)}")
     
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initializes the RAG model components when the server starts."""
    global global_rag
    global_rag = RAG()
    print("Generator model loaded successfully.")
    yield 

app = FastAPI(lifespan=lifespan, title="RAG API Server")

@app.get("/")
def read_root():
    return {"status": "ok", "message": "RAG API is running. Use /query to ask a question."}
    
@app.post("/query")
async def query_llm(query: str = Query(..., description="The query to send to the LLM.")):
    return await handle_rag_query(query, global_rag.rag_retrieve)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)