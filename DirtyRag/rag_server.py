from fastapi import FastAPI, HTTPException, Query
from contextlib import asynccontextmanager
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from typing import List, Dict, Any, Callable

import uvicorn

from web_scraper import WebScraper 


EMBED_MODEL = "Qwen/Qwen3-Embedding-0.6B"
RERANK_MODEL = "Qwen/Qwen3-Reranker-0.6B"
GEN_MODEL = "Qwen/Qwen3-4B-Instruct-2507"#"Qwen/Qwen3-1.7B" #"Qwen/Qwen3-4B-Instruct-2507"

DATA_DIR = "./data"
DEVICE_GENERATOR = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE_EMBEDDER = "cpu"
DEVICE_RERANKER = "cpu"

MAX_GEN_TOKENS = 4096
ENABLED_THINKING = False
BNB_COMPUTE_DTYPE = torch.float16

USE_WEB_SCRAPER = True
MAX_SCRAPE_CHARS = 2000

QWEN_PERSONALITY_RESPONSE_TOKENS = 256
QWEN_PERSONALITY_PROMPT = "\n".join([
    "Be concise, sarcastic and blunt.",
    f"**TRY TO LIMIT TOKENS TO {QWEN_PERSONALITY_RESPONSE_TOKENS} OR LESS.**",
])

def model_load_kwargs(device: str) -> Dict[str, Any]:
    common = {"trust_remote_code": True}
    if device == "cuda":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=BNB_COMPUTE_DTYPE,
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
    
class Generator:
    def __init__(self, model_name=GEN_MODEL, device=DEVICE_GENERATOR):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
                    model_name, 
                    **model_load_kwargs(device)
                ).eval()
        if device == "cuda" and torch.__version__ >= "2.0":
            self.model = torch.compile(self.model, mode="reduce-overhead")
            
    @torch.no_grad()
    def generate(self, messages: List[dict], max_new_tokens=MAX_GEN_TOKENS) -> str:
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=ENABLED_THINKING
        )
        
        enc = self.tokenizer([text], return_tensors="pt")
        # move each tensor directly to target device to avoid temporary full-GPU copy
        inputs = {k: v.to(self.model.device, non_blocking=True) for k, v in enc.items()}
        
        out = self.model.generate(
            **inputs, 
            max_new_tokens=max_new_tokens,
            do_sample=True, 
            use_cache=True 
        )
        
        out_ids = out[0][inputs["input_ids"].shape[1]:]
        response = self.tokenizer.decode(out_ids, skip_special_tokens=False)
        
        return response

class RAG:
    def __init__(self):
        self.generator = Generator()
        self.webscraper = WebScraper()
    
    def webscrape_and_generate(self, query: str) -> str:
        results = self.webscraper.search(query, num_results=3)
        
        if results:
            print("\n--- Extracted Search Results (Title and URL) ---")
            for i, result in enumerate(results):
                print(f"Result {i+1}:")
                print(f"  Title: {result['title']}")
                print(f"  URL:   {result['url']}")
                print("-" * 20)

        plain_text = self.webscraper.scrape_url(results[0]['url'])
        combined_query = f"{query}\n\nWeb Information:\n{plain_text[:MAX_SCRAPE_CHARS]}"
        messages = [
            {"role": "system", "content": QWEN_PERSONALITY_PROMPT}, 
            {"role": "user", "content": combined_query}                        
        ]
        return self.generator.generate(messages)
    
    def generate(self, query: str) -> str:
        messages = [
                {"role": "system", "content": QWEN_PERSONALITY_PROMPT}, 
                {"role": "user", "content": query}                        
            ]
        return self.generator.generate(messages)
    

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

@app.post("/websearch_query")
async def query_llm_websearch(query: str = Query(..., description="The query to send to the LLM to search the web.")):
    return await handle_rag_query(query, global_rag.webscrape_and_generate)

    
@app.post("/query")
async def query_llm(query: str = Query(..., description="The query to send to the LLM.")):
    return await handle_rag_query(query, global_rag.generate)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)