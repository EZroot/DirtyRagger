import os
from typing import List
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from bs4 import BeautifulSoup
import fitz  # PyMuPDF

from vector_store import PersistentVectorStore
from web_scraper import WebScraper

# -----------------------------
# Models & Settings
# -----------------------------
EMBED_MODEL = "Qwen/Qwen3-Embedding-0.6B"
RERANK_MODEL = "Qwen/Qwen3-Reranker-0.6B"
GEN_MODEL = "Qwen/Qwen3-4B-Instruct-2507"

DATA_DIR = "./data"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TOP_K = 5
RERANK_TOP_K = 5
MAX_GEN_TOKENS = 1024

# ============================================================
# Helpers: PDF / HTML / Markdown cleaning
# ============================================================
def clean_html(text: str) -> str:
    return BeautifulSoup(text, "html.parser").get_text(separator="\n")

def clean_pdf(path: str) -> str:
    doc = fitz.open(path)
    return "\n".join([page.get_text() for page in doc])

def clean_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# ============================================================
# Qwen-native reranker (yes/no)
# ============================================================
class QwenReranker:
    def __init__(self, model_name=RERANK_MODEL, device=DEVICE):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype="auto",
            device_map="auto" if device == "cuda" else None
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
        self.max_length = 8192

    def _format_pair(self, query: str, doc: str) -> str:
        return f"<Instruct>: Given a web search query, retrieve relevant passages.\n<Query>: {query}\n<Document>: {doc}"

    def _prepare_inputs(self, texts: List[str]):
        # Use __call__ directly for fast tokenizer batching
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length - len(self.prefix_ids) - len(self.suffix_ids),
            return_tensors="pt",
        )
        # Insert prefix/suffix manually
        for i in range(len(enc["input_ids"])):
            enc["input_ids"][i] = torch.cat([
                torch.tensor(self.prefix_ids),
                enc["input_ids"][i],
                torch.tensor(self.suffix_ids)
            ])
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
    def __init__(self, model_name=GEN_MODEL, device=DEVICE):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        kwargs = {"dtype": "auto", "device_map": "auto"} if device == "cuda" else {}
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs).eval()

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens=MAX_GEN_TOKENS) -> str:
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        out = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        out_ids = out[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(out_ids, skip_special_tokens=True)


# ============================================================
# RAG Pipeline
# ============================================================
class RAG:
    def __init__(self, data_dir=DATA_DIR):
        self.embedder = SentenceTransformer(EMBED_MODEL, device=DEVICE)
        dim = self.embedder.get_sentence_embedding_dimension()

        self.store = PersistentVectorStore(dim, data_dir=data_dir)
        self.reranker = QwenReranker(RERANK_MODEL, DEVICE)
        self.generator = Generator(GEN_MODEL, DEVICE)
        self.webscraper = WebScraper()
        
    def retrieve(self, query: str, k=TOP_K):
        qvec = self.embedder.encode([query], convert_to_numpy=True)[0]
        return self.store.search(qvec, k)

    def generate_answer(self, query: str):
        hits = self.retrieve(query, TOP_K)
        docs = [h["text"] for h in hits]

        if docs:
            scores = self.reranker.score(query, docs)
            ranked = sorted(zip(docs, scores), key=lambda x: -x[1])
            top_docs = [d for d, _ in ranked[:RERANK_TOP_K]]
        else:
            top_docs = []

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

        return self.generator.generate(prompt)


if __name__ == "__main__":
    rag = RAG()
    try:
        while True:
            q = input("Query> ").strip()
            if not q:
                continue
            print("\n" + rag.generate_answer(q) + "\n")
    except KeyboardInterrupt:
        print("\nExiting.")
