from typing import Any, Dict

import torch
from transformers import BitsAndBytesConfig

from rag_server_config import RAGConfig

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
        # default to cpu load
        return {
            **common,
            "device_map": "cpu",
            "low_cpu_mem_usage": True,
        }