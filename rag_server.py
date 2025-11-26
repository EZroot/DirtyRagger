from fastapi import FastAPI, HTTPException, Query
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager

from transformers import BitsAndBytesConfig
from typing import Dict, Any, Callable

import uvicorn
import torch

from rag_server_config import RAGConfig
from rag_server_generator import Generator
from rag_server_reranker import QwenReranker
from rag_server_tool_pass import ToolParseAndExecute

global SERVER_CONFIG
SERVER_CONFIG = RAGConfig()

class RAG:
    def __init__(self):
        self.generator = Generator()
        self.reranker = QwenReranker()
        self.toolparser = ToolParseAndExecute()

    def rag_retrieve(self, query: str) -> str:
        messages = [
                {"role": "system", "content": SERVER_CONFIG.QWEN_PERSONALITY_PROMPT}, 
                {"role": "user", "content": query}                        
            ]
        
        # check for tool use, omit this if too slow (TODO: Add to config as option)
        initial = self.generator.generate(messages, True)
        parsed_result = self.toolparser.parse_tool_call(initial)

        if parsed_result is None:
            print("No tool call detected, proceeding with initial generation.")
            return initial
        
        # round 2 generation with tool results
        messages = [
                {"role": "system", "content": f"{SERVER_CONFIG.QWEN_PERSONALITY_PROMPT}\n\n {parsed_result}"}, 
                {"role": "user", "content": query}                        
            ]
        
        final_result = self.generator.generate(messages, False)
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