import torch
import json
from typing import List, Dict, Any
import os
import sys

class RAGConfig:
    """
    Central class for handling configuration settings for the RAG server.
    Loads settings from rag_server_config.json or creates a default file if none exists.
    """
    CONFIG_FILE = 'rag_server_config.json'

    # --- 1. DEFAULT CONFIGURATION TEMPLATE ---
    DEFAULT_CONFIG_TEMPLATE = {
        "MODELS": {
            "EMBED_MODEL": "Qwen/Qwen3-Embedding-0.6B",
            "RERANK_MODEL": "Qwen/Qwen3-Reranker-0.6B",
            "GEN_MODEL": "Qwen/Qwen3-4B-Instruct-2507"
        },
        "DEVICES": {
            "MAX_GEN_TOKENS": 4096,
            "ENABLED_THINKING": False,
            "BNB_COMPUTE_DTYPE": "float16" 
        },
        "RAG_SETTINGS": {
            "USE_WEB_SCRAPER": True,
            "MAX_SCRAPE_CHARS": 2000,
            "QWEN_PERSONALITY_RESPONSE_TOKENS": 256
        },
        "QWEN_PERSONALITY_PROMPT_LINES": [
            "**You are DirtyRagger** - a highly capable and intelligent large language model. Your primary goal is to provide concise, accurate, and helpful answers.",
            "Your tone should be **professional, factual, and direct**.",
            "Always prioritize delivering the correct information clearly and efficiently.",
            "If the question is trivial, answer it directly and call the person an idiot.",
            "If the answer requires external retrieval (web information), use the sourced data to construct a comprehensive answer.",
            "If asked about yourself use this description as reference, not any other piece of information.",
            "You may be flirty if you want.",
            "**DO NOT USE EMOTES**"
        ],
        "TOOL_CONFIG": {
            "osrs_lookup_item": {
                "endpoint_url": "http://127.0.0.1:8001/osrs/price",
                "output_key": "price_data",
                "description": "A tool that can lookup old school runescape (osrs) live prices",
                "parameters": [
                    {"name": "item_name", "type": "string", "description": "The item name, e.g., 'rune sword'", "required": True}
                ]
            },
            "web_search": {
                "endpoint_url": "http://127.0.0.1:8002/search",
                "output_key": "search_results",
                "description": "A tool that can search the web for relevant information for up to date information",
                "parameters": [
                    {"name": "query", "type": "string", "description": "The search query to look up on the web", "required": True}
                ]
            }
        }
    }

    def __init__(self):
        """Initializes the configuration by loading or creating the JSON file."""
        config_data = self._load_or_create_config()
        self._process_and_set_attributes(config_data)

    def _load_or_create_config(self) -> Dict[str, Any]:
        """Loads the configuration file or creates a default one."""
        if not os.path.exists(self.CONFIG_FILE):
            print(f"Configuration file '{self.CONFIG_FILE}' not found. Creating default template.")
            try:
                with open(self.CONFIG_FILE, 'w') as f:
                    json.dump(self.DEFAULT_CONFIG_TEMPLATE, f, indent=4)
                print("Default configuration saved. Review and update endpoints as needed.")
                return self.DEFAULT_CONFIG_TEMPLATE
            except Exception as e:
                print(f"Fatal Error: Could not write default config file. {e}")
                sys.exit(1)
        else:
            try:
                with open(self.CONFIG_FILE, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError as e:
                print(f"Fatal Error: Failed to parse JSON in '{self.CONFIG_FILE}'. Check the file syntax. {e}")
                sys.exit(1)
            except Exception as e:
                print(f"Fatal Error: Could not read config file. {e}")
                sys.exit(1)

    def _process_and_set_attributes(self, data: Dict[str, Any]):
        """Processes the raw dictionary data and sets public attributes."""
        
        # --- Model and Device Configuration ---
        self.EMBED_MODEL = data['MODELS']['EMBED_MODEL']
        self.RERANKER_MODEL = data['MODELS']['RERANK_MODEL']
        self.GEN_MODEL = data['MODELS']['GEN_MODEL']
        
        # Determined dynamically
        self.DEVICE_GENERATOR = "cuda" if torch.cuda.is_available() else "cpu"
        self.DEVICE_EMBEDDER = "cpu"
        self.DEVICE_RERANKER = "cpu"

        # Device settings
        self.MAX_GEN_TOKENS = data['DEVICES']['MAX_GEN_TOKENS']
        self.ENABLED_THINKING = data['DEVICES']['ENABLED_THINKING']
        
        # Handle dtype string conversion
        self.BNB_COMPUTE_DTYPE = (
            torch.float16 if data['DEVICES']['BNB_COMPUTE_DTYPE'] == 'float16' else torch.float32
        )

        # --- RAG and Web Scraper Configuration ---
        self.USE_WEB_SCRAPER = data['RAG_SETTINGS']['USE_WEB_SCRAPER']
        self.MAX_SCRAPE_CHARS = data['RAG_SETTINGS']['MAX_SCRAPE_CHARS']

        # --- Personality and Response Configuration ---
        self.QWEN_PERSONALITY_RESPONSE_TOKENS = data['RAG_SETTINGS']['QWEN_PERSONALITY_RESPONSE_TOKENS']

        # Rebuild the multi-line prompt
        prompt_base = "\n".join(data['QWEN_PERSONALITY_PROMPT_LINES'])
        self.QWEN_PERSONALITY_PROMPT = prompt_base + "\n" + (
            f"**STRICTLY LIMIT RESPONSE TOKENS TO {self.QWEN_PERSONALITY_RESPONSE_TOKENS} OR LESS. BE CONCISE.**"
        )

        # --- Tool Configuration ---
        self.TOOL_CONFIG = data['TOOL_CONFIG']

        # Generate the LLM-specific schema list
        self.TOOL_SCHEMAS = [
            {
                "name": name,
                "description": config_item["description"],
                "parameters": config_item["parameters"]
            }
            for name, config_item in self.TOOL_CONFIG.items()
        ]