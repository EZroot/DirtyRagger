from fastapi import FastAPI, HTTPException, Query
from contextlib import asynccontextmanager
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from typing import List, Dict, Any, Callable

import uvicorn

from rag_server_config import RAGConfig
from rag_server_generator import Generator
from rag_server_reranker import QwenReranker
from rag_server_tool_pass import ToolParseAndExecute

global SERVER_CONFIG
SERVER_CONFIG = RAGConfig()

def model_load_kwargs(device: str) -> Dict[str, Any]:
    common = {"trust_remote_code": True}
    if device == "cuda":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=SERVER_CONFIG.BNB_COMPUTE_DTYPE,
        )
        return {
            **common,
            "quantization_config": bnb_config,
            "device_map": "auto", 
            "dtype": torch.float16,
            "low_cpu_mem_usage": True,
        }
    else:
        # Force CPU load path (smaller memory spikes)
        return {
            **common,
            "device_map": "cpu",
            "low_cpu_mem_usage": True,
        }

class RAG:
    def __init__(self):
        self.generator = Generator(model_load_kwargs=model_load_kwargs(SERVER_CONFIG.DEVICE_GENERATOR))
        self.reranker = QwenReranker(model_load_kwargs=model_load_kwargs(SERVER_CONFIG.DEVICE_RERANKER))
        self.toolparser = ToolParseAndExecute()

    def generate_with_tool_parsing(self, query: str) -> str:
        messages = [
                {"role": "system", "content": SERVER_CONFIG.QWEN_PERSONALITY_PROMPT}, 
                {"role": "user", "content": query}                        
            ]
        
        initial = self.generator.generate(messages, True)

        parsed_result = self.toolparser.parse_tool_call(initial)
        if parsed_result is None:
            return "Tool call failed to parse."

        # round 2 generation with tool results
        new_messages = [
                {"role": "system", "content": f"{SERVER_CONFIG.QWEN_PERSONALITY_PROMPT}\n\n {parsed_result}"}, 
                {"role": "user", "content": query}                        
            ]
        
        final_result = self.generator.generate(new_messages, False)
        print("Final Result:\n"+final_result)
        return final_result
    

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
    return await handle_rag_query(query, global_rag.generate_with_tool_parsing)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)