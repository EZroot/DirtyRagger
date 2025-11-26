from typing import Any, Callable, Dict, List
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rag_server_config import RAGConfig

global SERVER_CONFIG
SERVER_CONFIG = RAGConfig()

class QwenReranker:
    def __init__(self, model_load_kwargs: Callable[[str], Dict[str, Any]], model_name: str = SERVER_CONFIG.RERANKER_MODEL, device: str = SERVER_CONFIG.DEVICE_RERANKER, reranker_max_input: int = SERVER_CONFIG.RERANKER_MAX_INPUT):
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
        self.max_length = reranker_max_input

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
        print(f"Starting reranking for {len(docs)} documents.")
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
        print(f"Reranking completed. Top score: {max(scores) if scores else 'N/A'}")
        return scores