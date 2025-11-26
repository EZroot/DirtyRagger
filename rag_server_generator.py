from typing import Any, Callable, Dict, List
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from rag_server_config import RAGConfig


global SERVER_CONFIG
SERVER_CONFIG = RAGConfig()

class Generator:
    def __init__(self, model_load_kwargs: Callable[[str], Dict[str, Any]], model_name=SERVER_CONFIG.GEN_MODEL, device=SERVER_CONFIG.DEVICE_GENERATOR):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
                    model_name, 
                    **model_load_kwargs(device)
                ).eval()
        if device == "cuda" and torch.__version__ >= "2.0":
            self.model = torch.compile(self.model, mode="reduce-overhead")
            
    @torch.no_grad()
    def generate(self, messages: List[dict], usetools: bool, max_new_tokens=SERVER_CONFIG.MAX_GEN_TOKENS) -> str:
        if usetools:
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=SERVER_CONFIG.ENABLED_THINKING, tools=SERVER_CONFIG.TOOL_SCHEMAS
            )
        else:
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=SERVER_CONFIG.ENABLED_THINKING
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