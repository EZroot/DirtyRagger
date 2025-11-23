from typing import List, AsyncGenerator
import torch
import json
import uvicorn
import asyncio
from contextlib import asynccontextmanager # NEW: For modern lifespan handling

# IMPORTANT: Ensure all required components are imported from fastapi
from fastapi import FastAPI, Query, HTTPException 
from starlette.responses import StreamingResponse 

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig 
from sentence_transformers import SentenceTransformer

from vector_store import PersistentVectorStore
from web_scraper import WebScraper

# -----------------------------
# Models & Settings
# -----------------------------
EMBED_MODEL = "Qwen/Qwen3-Embedding-0.6B"
RERANK_MODEL = "Qwen/Qwen3-Reranker-0.6B"
GEN_MODEL = "Qwen/Qwen3-1.7B"

DATA_DIR = "./data"
DEVICE_GENERATOR = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE_EMBEDDER = "cpu"
DEVICE_RERANKER = "cpu"

TOP_K = 5 
RERANK_TOP_K = 3 
MAX_GEN_TOKENS = 4096 
RERANKER_MAX_INPUT = 1024 
ENABLED_THINKING = False 
STREAM_GENERATION_TOKENS = True # Always True for this streaming API

CONFIDENCE_SCORE_THRESHOLD_FOR_FREE_ANSWER = 0.5 
SEARCH_WEB_IF_LOW_CONFIDENCE = True
WEB_SEARCH_NUM_RESULT = 1 
WEB_TOKEN_LIMIT = 512 
BNB_COMPUTE_DTYPE = torch.float16 

def model_load_kwargs(device: str):
    if device == "cuda":
        # 1. Define the 4-bit Quantization Configuration
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, 
            bnb_4bit_use_double_quant=True, 
            bnb_4bit_quant_type="nf4", 
            bnb_4bit_compute_dtype=BNB_COMPUTE_DTYPE 
        )
        
        # 2. Return the config along with device_map
        return {
            "quantization_config": bnb_config, 
            "dtype": torch.float16, 
            "device_map": "auto", 
            "trust_remote_code": True
        }
    else:
        return {"trust_remote_code": True} 

# ============================================================
# Qwen-native reranker 
# ============================================================
class QwenReranker:
    def __init__(self, model_name=RERANK_MODEL, device=DEVICE_RERANKER):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_load_kwargs(device)
        ).eval()

        self.token_false = self.tokenizer.convert_tokens_to_ids("no")
        self.token_true = self.tokenizer.convert_tokens_to_ids("yes")

        self.prefix = (
            "<|im_start|>system\n"
            "Judge whether the Document meets the requirements based on the Query and the Instruct provided. "
            "Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n"
            "<|im_start|>user\n"
        )

        self.suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

        self.prefix_ids = self.tokenizer(self.prefix, add_special_tokens=False)["input_ids"]
        self.suffix_ids = self.tokenizer(self.suffix, add_special_tokens=False)["input_ids"]
        self.max_length = RERANKER_MAX_INPUT

    def _format_pair(self, query: str, doc: str) -> str:
        return f"<Instruct>: Given a web search query, retrieve relevant passages.\n<Query>: {query}\n<Document>: {doc}"

    def _prepare_inputs(self, texts: List[str]):
        batch_texts = [self.prefix + t + self.suffix for t in texts]

        enc = self.tokenizer(
            batch_texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {k: v.to(self.model.device) for k, v in enc.items()}


    @torch.no_grad()
    def score(self, query: str, docs: List[str]) -> List[float]:
        pairs = [self._format_pair(query, d) for d in docs]
        inputs = self._prepare_inputs(pairs)

        logits = self.model(**inputs).logits[:, -1, :]
        true_logit = logits[:, self.token_true]
        false_logit = logits[:, self.token_false]

        stacked = torch.stack([false_logit, true_logit], dim=1)
        probs = torch.nn.functional.log_softmax(stacked, dim=1).exp()
        return probs[:, 1].tolist()

# ============================================================
# Qwen-native generator
# ============================================================
class Generator:
    def __init__(self, model_name=GEN_MODEL, device=DEVICE_GENERATOR):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
                    model_name, 
                    **model_load_kwargs(device)
                ).eval()
        
    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens=MAX_GEN_TOKENS) -> str:
        # Non-streaming implementation (kept for completeness)
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=ENABLED_THINKING
        )
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        out = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        out_ids = out[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(out_ids, skip_special_tokens=True)

    @torch.no_grad()
    async def generate_stream(self, prompt: str, max_new_tokens=MAX_GEN_TOKENS) -> AsyncGenerator[str, None]:
        """
        Stream the output token by token, yielding text chunks for the HTTP server.
        """
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=ENABLED_THINKING
        )

        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        generated_ids = inputs.input_ids.clone()
        
        print("[SERVER] Starting stream generation...") 

        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Note: This is a simplified generation loop. A true high-performance
                # streaming implementation would leverage key/value cache for speed.
                outputs = self.model(input_ids=generated_ids, attention_mask=torch.ones_like(generated_ids))
                next_token_logits = outputs.logits[:, -1, :]
                next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)

                generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)
                next_token_text = self.tokenizer.decode(next_token_id[0], skip_special_tokens=True)

                if next_token_text:
                    yield next_token_text
                    # Use asyncio.sleep(0) to yield control back to the event loop
                    await asyncio.sleep(0) 

                if next_token_id.item() == self.tokenizer.eos_token_id:
                    break
        
        pass
    
# ============================================================
# RAG Pipeline
# ============================================================
class RAG:
    def __init__(self, data_dir=DATA_DIR):
        print(f"[{self.__class__.__name__}] Initializing RAG components...")
        self.embedder = SentenceTransformer(EMBED_MODEL, device=DEVICE_EMBEDDER)
        dim = self.embedder.get_sentence_embedding_dimension()

        self.store = PersistentVectorStore(dim, data_dir=data_dir)
        self.reranker = QwenReranker(RERANK_MODEL, DEVICE_RERANKER)
        self.generator = Generator(GEN_MODEL, DEVICE_GENERATOR)
        self.webscraper = WebScraper()
        print(f"[{self.__class__.__name__}] Initialization complete.")


    def retrieve(self, query: str, k=TOP_K):
        qvec = self.embedder.encode([query], convert_to_numpy=True)[0]
        return self.store.search(qvec, k)

    def generate_answer(self, query: str):
        """
        Generates the answer, returning an async generator and the confidence score.
        """
        hits = self.retrieve(query, TOP_K)
        docs = [h["text"] for h in hits]

        if docs:
            scores = self.reranker.score(query, docs)
            ranked = sorted(zip(docs, scores), key=lambda x: -x[1])
            top_ranked = ranked[:RERANK_TOP_K]
            top_docs = [d for d, _ in top_ranked]
            top_scores = [s for _, s in top_ranked]
            confidence_score = top_scores[0] if top_scores else 0.0

        else:
            top_docs = []
            confidence_score = 0.0

        print(f"[DEBUG] Comparing {confidence_score} to {CONFIDENCE_SCORE_THRESHOLD_FOR_FREE_ANSWER}")
        
        # --- Prompt Construction Logic ---
        if confidence_score > CONFIDENCE_SCORE_THRESHOLD_FOR_FREE_ANSWER:
            context = "\n\n---\n\n".join(
                f"Passage {i+1}:\n{d}" for i, d in enumerate(top_docs)
            )
            prompt = (
                "Use the following context passages to answer the question.\n\n"
                f"{context}\n\n"
                f"Question: {query}\n\n"
                "If the answer is not in the passages, say:\n"
                "'I don't know based on the provided passages.'"
            )
        else:
            if SEARCH_WEB_IF_LOW_CONFIDENCE:
                web_search_result = self.webscraper.search(query, num_results=WEB_SEARCH_NUM_RESULT)
                if web_search_result:
                    first_url = web_search_result[0]['url']
                    plain_text = self.webscraper.scrape_url(first_url)
                    filtered_result = plain_text[:WEB_TOKEN_LIMIT]
                    prompt = f"Answer the following question based on your internal knowledge. Additional information context:{filtered_result}:\n\nQuestion: {query}"
                else:
                    prompt = f"Answer the following question based on your internal knowledge:\n\nQuestion: {query}"
        
        # --- Streaming Server Response Logic ---
        
        # This nested async generator will handle the two-part response structure
        async def streaming_generator() -> AsyncGenerator[str, None]:
            # 1. Yield JSON metadata first
            metadata = {"confidence": confidence_score}
            # NOTE: We do not yield a newline separator here.
            yield json.dumps(metadata)
            
            # 2. Stream the text chunks
            async for chunk in self.generator.generate_stream(prompt):
                yield chunk
        
        return streaming_generator(), confidence_score


# ============================================================
# FastAPI Server Setup
# ============================================================
global global_rag

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initializes the RAG model components when the server starts (replaces @app.on_event)."""
    global global_rag
    print(f"[{app.title}] Server starting up. Initializing RAG...")
    # NOTE: The models load here, this will take time!
    global_rag = RAG()
    print(f"[{app.title}] RAG components loaded.")
    yield # Server runs here
    # Optional cleanup code here

# Create the FastAPI application instance using the modern lifespan
app = FastAPI(lifespan=lifespan)


@app.get("/")
def read_root():
    return {"status": "ok", "message": "RAG API is running. Use /query to ask a question."}


# NOTE: The dummy streaming_generator function has been removed.
@app.post("/query")
async def query_llm_stream(query: str = Query(..., description="The query to send to the LLM.")):
    """
    The main endpoint for streaming responses, using the RAG object's internal generator.
    """
    
    if not global_rag:
        raise HTTPException(status_code=503, detail="RAG system is still loading. Please wait.")
        
    # Check for empty query
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    # Get the generator from the RAG object (tuple is generator, confidence_score)
    generator, _ = global_rag.generate_answer(query)

    # Use StreamingResponse with the actual generator
    return StreamingResponse(
        content=generator,
        media_type="application/json" # Changed media type to something simpler
    )

if __name__ == "__main__":
    print(f"Starting RAG Server on {DEVICE_GENERATOR}...")
    # The RAG object will be initialized in the startup_event (now lifespan)
    uvicorn.run(app, host="0.0.0.0", port=8000)