# rag_tooling_server.py
import re
import json
import asyncio
import logging
from typing import List, AsyncGenerator, Dict, Any
from contextlib import asynccontextmanager

import torch
import uvicorn
import httpx # For making asynchronous tool calls

# IMPORTANT: Ensure all required components are imported from fastapi
from fastapi import FastAPI, Query, HTTPException 
from starlette.responses import StreamingResponse 

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig 
from sentence_transformers import SentenceTransformer

from vector_store import PersistentVectorStore
from web_scraper import WebScraper

# ============================================================
# Logging Configuration
# ============================================================
logging.basicConfig(
    level=logging.INFO, # Set to DEBUG for maximum output, INFO for general
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("RAG_SERVER")
logger.setLevel(logging.DEBUG) # Set RAG_SERVER logs to DEBUG for verbosity
# ============================================================


# -----------------------------
# Models & Settings
# -----------------------------
EMBED_MODEL = "Qwen/Qwen3-Embedding-0.6B"
RERANK_MODEL = "Qwen/Qwen3-Reranker-0.6B"
GEN_MODEL = "Qwen/Qwen3-1.7B"#"Qwen/Qwen3-1.7B" #"Qwen/Qwen3-4B-Instruct-2507"

DATA_DIR = "./data"
DEVICE_GENERATOR = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE_EMBEDDER = "cpu"
DEVICE_RERANKER = "cpu"

TOP_K = 1 
RERANK_TOP_K = 1
MAX_GEN_TOKENS = 1024 
RERANKER_MAX_INPUT = 256 
ENABLED_THINKING = False 
STREAM_GENERATION_TOKENS = True 

CONFIDENCE_SCORE_THRESHOLD_FOR_FREE_ANSWER = 0.5 
SEARCH_WEB_IF_LOW_CONFIDENCE = True 
WEB_SEARCH_NUM_RESULT = 1 
WEB_TOKEN_LIMIT = 1024 
BNB_COMPUTE_DTYPE = torch.float16 

TOOL_SERVER_URL = "http://0.0.0.0:8001"
ENABLE_TOOLING = False # Dont have enough memory for the double pass it requires, pretty annoying and not that nessesary

def model_load_kwargs(device: str) -> Dict[str, Any]:
    logger.debug(f"Configuring load kwargs for device: {device}")
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
            "device_map": "auto",              # let transformers map layers to GPU/CPU
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

# ============================================================
# Qwen-native reranker
# ============================================================
class QwenReranker:
    def __init__(self, model_name=RERANK_MODEL, device=DEVICE_RERANKER):
        logger.info(f"Loading Reranker Model: {model_name} on device: {device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_load_kwargs(device)
        ).eval()
        logger.info("Reranker Model loaded successfully.")

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
        logger.debug(f"Starting reranking for {len(docs)} documents.")
        if not docs:
            return []
            
        pairs = [self._format_pair(query, d) for d in docs]
        inputs = self._prepare_inputs(pairs)

        logits = self.model(**inputs).logits[:, -1, :]
        true_logit = logits[:, self.token_true]
        false_logit = logits[:, self.token_false]

        stacked = torch.stack([false_logit, true_logit], dim=1)
        probs = torch.nn.functional.log_softmax(stacked, dim=1).exp()
        scores = probs[:, 1].tolist()
        logger.debug(f"Reranking completed. Top score: {max(scores) if scores else 'N/A'}")
        return scores
    
# ============================================================
# Qwen-native generator
# ============================================================
class Generator:
    def __init__(self, model_name=GEN_MODEL, device=DEVICE_GENERATOR):
        logger.info(f"Loading Generator Model: {model_name} on device: {device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
                    model_name, 
                    **model_load_kwargs(device)
                ).eval()
        logger.info("Generator Model loaded successfully.")
    
    @torch.no_grad()
    def generate(self, messages: List[dict], available_tools: List[dict], max_new_tokens=MAX_GEN_TOKENS) -> str:
        """Non-streaming generation used for tool-call intent detection."""
        logger.debug("Starting non-streaming generation for tool detection.")
        
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=ENABLED_THINKING, tools=available_tools
        )
        
        enc = self.tokenizer([text], return_tensors="pt")
        # move each tensor directly to target device to avoid temporary full-GPU copy
        inputs = {k: v.to(self.model.device, non_blocking=True) for k, v in enc.items()}
        del enc
        torch.cuda.empty_cache()  # safe to call on CUDA

        
        out = self.model.generate(
            **inputs, 
            max_new_tokens=max_new_tokens,
            do_sample=False, 
            use_cache=True 
        )
        
        out_ids = out[0][inputs["input_ids"].shape[1]:]
        response = self.tokenizer.decode(out_ids, skip_special_tokens=False)
        
        logger.debug("Non-streaming generation complete.")
        logger.debug(f"Initial response (truncated): {response[:200]}...")
        
        return response

    @torch.no_grad()
    async def generate_stream(self, messages: List[dict], available_tools: List[dict], max_new_tokens=MAX_GEN_TOKENS) -> AsyncGenerator[str, None]:
        """Stream the output token by token, yielding text chunks using manual KV Caching."""
        logger.info("Starting stream generation (KV Cache enabled).")
        
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=ENABLED_THINKING, tools=available_tools
        )
                
        enc = self.tokenizer([text], return_tensors="pt")

        # move each tensor directly to target device to avoid temporary full-GPU copy
        inputs = {k: v.to(self.model.device, non_blocking=True) for k, v in enc.items()}
        del enc

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        # 1. First Pass: Process the full prompt to generate the first logit and cache
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True 
        )
        
        # Explicitly delete the full input tensors, they are no longer needed
        del inputs, input_ids
        
        if self.model.device.type == "cuda": 
            torch.cuda.empty_cache()
            logger.debug("CUDA cache cleared after processing large prompt.")

        # Initialize state from the first pass
        past_key_values = outputs.past_key_values
        
        # Get the logits for the *first* generated token from the prompt processing output
        next_token_logits = outputs.logits[:, -1, :] 
        
        # Select the next token (this will be the *first* token yielded in the loop)
        next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        
        # Clean up the output from the prompt processing (only cache is needed)
        del outputs
        
        # Prepare the attention mask for the first generated token
        attention_mask = torch.cat(
            [attention_mask, torch.ones((attention_mask.shape[0], 1), dtype=torch.long, device=self.model.device)], dim=-1
        )
        last_token_id = next_token_id # This is the first token to be processed/yielded

        # 2. Generation Loop: Use KV cache to stream one token at a time
        for _ in range(max_new_tokens):
            
            # Core Generation Step: Use the single last token and the accumulated cache
            outputs = self.model(
                input_ids=last_token_id,
                attention_mask=attention_mask, 
                past_key_values=past_key_values,
                use_cache=True
            )
            
            # --- Yield Logic ---
            
            # Decode and yield the *current* token (last_token_id)
            current_token_text = self.tokenizer.decode(last_token_id[0], skip_special_tokens=True)

            if current_token_text:
                yield current_token_text
                await asyncio.sleep(0) # Yield control to the event loop

            # Check for EOS after yielding the token
            if last_token_id.item() == self.tokenizer.eos_token_id:
                logger.info("Stream generation hit EOS token.")
                break
            
            # --- Prepare for Next Iteration ---
            
            # Update cache for the next iteration
            past_key_values = outputs.past_key_values
            
            # Select the *next* token
            next_token_logits = outputs.logits[:, -1, :]
            new_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # Update the token ID and attention mask for the next iteration
            last_token_id = new_token_id
            attention_mask = torch.cat(
                [attention_mask, torch.ones((attention_mask.shape[0], 1), dtype=torch.long, device=self.model.device)], dim=-1
            )
            
        logger.warning("Stream generation finished.")
# ============================================================
# TOOL SCHEMA DEFINITIONS 
# ============================================================
def get_current_time_json_schema() -> Dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Returns the current system time (date and time) in ISO format to answer time-related questions.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    }

# ============================================================
# RAG Pipeline (UPDATED FOR REMOTE TOOLING)
# ============================================================
class RAG:
    def __init__(self, data_dir=DATA_DIR):
        logger.info("Initializing RAG components...")
        self.embedder = SentenceTransformer(EMBED_MODEL, device=DEVICE_EMBEDDER)
        dim = self.embedder.get_sentence_embedding_dimension()
        logger.info(f"Embedder loaded. Embedding dimension: {dim}")

        self.store = PersistentVectorStore(dim, data_dir=data_dir)
        self.reranker = QwenReranker(RERANK_MODEL, DEVICE_RERANKER)
        self.generator = Generator(GEN_MODEL, DEVICE_GENERATOR)
        self.webscraper = WebScraper()
        
        # Initialize and store the available tool schemas
        self.tools = self.get_available_tools() 

        # after loading all models in RAG.__init__
        if DEVICE_GENERATOR == "cuda":
            torch.cuda.empty_cache()

        logger.info(f"RAG Initialization complete. Tools available: {len(self.tools)}")

    # ------------------
    # Tool Execution (REMOTE)
    # ------------------
    async def _execute_remote_tool_call(self, function_name: str, args: dict) -> str:
        """Executes a function on the remote tooling server via HTTP."""
        payload = {
            "function_name": function_name,
            "arguments": args
        }
        
        logger.info(f"Tool Call: {function_name}({args}). Connecting to {TOOL_SERVER_URL}/execute")
        
        # Use httpx for asynchronous request
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(f"{TOOL_SERVER_URL}/execute", json=payload, timeout=30)
                response.raise_for_status() 
                
                result_data = response.json()
                
                if result_data.get("success"):
                    output = result_data.get("output", "")
                    logger.info(f"Tool '{function_name}' executed successfully. Output (truncated): {output[:50]}...")
                    return output
                else:
                    detail = result_data.get('detail', 'No details provided')
                    logger.error(f"Tool execution failed (Server reported failure): {detail}")
                    return f"Tool execution failed: {detail}"

            except httpx.HTTPStatusError as e:
                logger.error(f"HTTP Error calling tool server ({e.response.status_code}): {e.response.text}")
                return f"Remote tool server HTTP error ({e.response.status_code}): {e.response.text}"
            except httpx.RequestError as e:
                logger.error(f"Request Error calling tool server: {e.__class__.__name__} - {e}")
                return f"Could not connect to remote tool server at {TOOL_SERVER_URL}: {e.__class__.__name__}"


    def get_available_tools(self) -> List[dict]:
        return [
            get_current_time_json_schema(),
            # Add other tool schemas here 
        ]
        
    def retrieve(self, query: str, k=TOP_K):
        logger.info(f"Retrieving documents for query: '{query}' (k={k})")
        qvec = self.embedder.encode([query], convert_to_numpy=True)[0]
        hits = self.store.search(qvec, k)
        logger.debug(f"Retrieved {len(hits)} initial documents.")
        return hits


    async def generate_answer(self, query: str):
        """Generates the answer, integrating RAG, Web Search, and Remote Tool Calling."""
        
        # 1. Retrieval and Reranking (No Change)
        hits = self.retrieve(query, TOP_K)
        docs = [h["text"] for h in hits]
        
        top_docs = []
        confidence_score = 0.0

        if docs:
            scores = self.reranker.score(query, docs)
            ranked = sorted(zip(docs, scores), key=lambda x: -x[1])
            top_ranked = ranked[:RERANK_TOP_K]
            top_docs = [d for d, _ in top_ranked]
            top_scores = [s for _, s in top_ranked]
            confidence_score = top_scores[0] if top_scores else 0.0
            logger.info(f"Reranking complete. Top confidence score: {confidence_score:.4f}")
        
        # 2. Context & Message History Setup (No Change)
        messages = [{"role": "user", "content": query}]
        system_context = None 
        
        if confidence_score > CONFIDENCE_SCORE_THRESHOLD_FOR_FREE_ANSWER:
            context = "\n\n---\n\n".join(f"Passage {i+1}:\n{d}" for i, d in enumerate(top_docs))
            system_context = f"Use the following context passages to answer the question, or use an available tool if required:\n\n{context}"
        
        if system_context:
            messages.insert(0, {"role": "system", "content": system_context})
        
        logger.debug(f"Initial messages prepared: {messages}")

        # --- 3. Tool Orchestration Loop (STREAMING DETECTION) ---
        
        # Start the *streaming* generation, passing the tool schema
        initial_stream = self.generator.generate_stream(messages, self.tools)
        
        full_response = ""
        tool_call_detected = False
        tool_call_started = False
        final_generator = None
        tool_call_used = False
        
        # Use a list to store chunks that will be streamed out *after* metadata
        streamed_chunks = []
        
        # The first pass: Stream chunks, accumulate response, and detect tool call completion
        async for chunk in initial_stream:
            full_response += chunk
            streamed_chunks.append(chunk)

            # 1. Detect start and completion of the tool call tags
            if not tool_call_started and "<tool_call>" in full_response:
                tool_call_started = True
            
            if tool_call_started and "</tool_call>" in full_response:
                tool_call_detected = True
                logger.info("Complete tool call detected in stream. Stopping generation.")
                break
        
        # CUDA cleanup after the initial stream (which processed the prompt)
        if DEVICE_GENERATOR == "cuda":
            torch.cuda.empty_cache()
            logger.info("CUDA cache cleared after initial stream detection.")
        
        # --- Handle Tool Execution or Final Answer ---

        if tool_call_detected and ENABLE_TOOLING:
            tool_call_used = True
            
            try:
                # Parse the accumulated tool call text (full_response)
                tool_call_json_match = re.search(r"<tool_call>(.*?)<\/tool_call>", full_response, re.DOTALL)
                
                # ... (Your existing, correct tool parsing logic) ...
                if not tool_call_json_match:
                    raise ValueError("Tool call tags found, but JSON content was not successfully matched.")

                tool_calls_str = tool_call_json_match.group(1).strip()
                raw_tool_calls = json.loads(tool_calls_str)
                tool_calls = [raw_tool_calls] if isinstance(raw_tool_calls, dict) else raw_tool_calls
                tool_call = tool_calls[0]
                function_name = tool_call['name']
                function_args = tool_call.get('arguments', {})
                
                # Execute the REMOTE function
                tool_output = await self._execute_remote_tool_call(function_name, function_args)
                
                # Append history for the final streaming generation
                messages.append({"role": "assistant", "content": full_response}) # Tool call message
                messages.append({"role": "tool", "content": tool_output})        # Tool's result
                logger.debug("Tool output added to messages. Starting final streaming generation.")
                
                # Second Generation: Model answers using the tool output
                final_generator = self.generator.generate_stream(messages, self.tools)
                
            except Exception as e:
                logger.error(f"Tool parsing/execution failed: {e}. Falling back to initial streamed text.")
                # If tool fails, use the initial accumulated text as the final output.
                tool_call_used = False
        
        # If no tool call was detected, or if tool execution failed, the final_generator
        # must be constructed from the initial streamed_chunks.
        if final_generator is None:
            logger.info("Streaming initial response as final answer (no tool call or tool failure).")
            
            # This replaces _mock_stream_from_text(response) for the successful RAG path
            # and the tool-call-failed path.
            async def re_stream_chunks():
                for chunk in streamed_chunks:
                    yield chunk
                    await asyncio.sleep(0.001)
            
            final_generator = re_stream_chunks()
        
        # 4. Streaming Server Response Logic (Wrapper)
        async def streaming_generator() -> AsyncGenerator[str, None]:
            # First yield: Metadata
            metadata = {"confidence": confidence_score, "tool_used": tool_call_used}
            yield json.dumps(metadata)
            
            # Second yield: Content chunks
            async for chunk in final_generator:
                yield chunk
        
        return streaming_generator(), confidence_score

# ============================================================
# FastAPI Server Setup
# ============================================================
global global_rag

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initializes the RAG model components when the server starts."""
    global global_rag
    logger.info("Server starting up. Initializing RAG components in lifespan event.")
    try:
        global_rag = RAG()
        logger.info("RAG components loaded successfully.")
    except Exception as e:
        logger.critical(f"FATAL ERROR during RAG initialization: {e}")
        global_rag = None # Ensure it's None if loading failed
    
    yield # Server runs here

# Create the FastAPI application instance using the modern lifespan
app = FastAPI(lifespan=lifespan, title="RAG API Server")


@app.get("/")
def read_root():
    logger.debug("Root endpoint / called.")
    return {"status": "ok", "message": "RAG API is running. Use /query to ask a question."}


@app.post("/query")
async def query_llm_stream(query: str = Query(..., description="The query to send to the LLM.")):
    """The main endpoint for streaming responses, using the RAG object's internal generator."""
    logger.info(f"Received query: '{query}'")
    
    if not global_rag:
        logger.error("RAG system not initialized. Returning 503.")
        raise HTTPException(status_code=503, detail="RAG system is still loading or failed to initialize. Please wait.")
        
    if not query:
        logger.warning("Empty query received.")
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    try:
        # AWAIT the new async RAG method
        generator, confidence = await global_rag.generate_answer(query)
        logger.info(f"Returning StreamingResponse for query with confidence: {confidence:.4f}")

        # Use StreamingResponse with the actual generator
        return StreamingResponse(
            content=generator,
            media_type="application/json"
        )
    except Exception as e:
        logger.critical(f"Unhandled exception during query processing: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

if __name__ == "__main__":
    logger.info(f"Starting RAG Server. Generator Device: {DEVICE_GENERATOR}...")
    uvicorn.run(app, host="0.0.0.0", port=8000)