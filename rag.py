import os
from typing import List
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from bs4 import BeautifulSoup

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

TOP_K = 5 # how many files to look through
RERANK_TOP_K = 3 # how many to pass to ranker (top 3)
MAX_GEN_TOKENS = 256 # how many tokens to generate
MAX_RERANKER_LENGTH = 1024 # how many reranker tokens to generate
ENABLED_THINKING = False # main model thinking

STREAM_GENERATION_TOKENS = True # Streams the text as its generated from our ai for more responsive feel
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
        self.max_length = MAX_RERANKER_LENGTH

    def _format_pair(self, query: str, doc: str) -> str:
        return f"<Instruct>: Given a web search query, retrieve relevant passages.\n<Query>: {query}\n<Document>: {doc}"

    def _prepare_inputs(self, texts: List[str]):
        # Combine prefix/suffix directly into the text
        batch_texts = [self.prefix + t + self.suffix for t in texts]

        # Use __call__ directly (fast path) with padding/truncation
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
    def __init__(self, model_name=GEN_MODEL, device=DEVICE):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        kwargs = {"dtype": "auto", "device_map": "auto"} if device == "cuda" else {}
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs).eval()

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens=MAX_GEN_TOKENS) -> str:
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=ENABLED_THINKING
        )
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        out = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        out_ids = out[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(out_ids, skip_special_tokens=True)

    @torch.no_grad()
    def generate_stream(self, prompt: str, max_new_tokens=MAX_GEN_TOKENS):
        """
        Stream the output token by token
        """
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=ENABLED_THINKING
        )

        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        generated_ids = inputs.input_ids.clone()

        print("\nAnswer (streaming): ", end="", flush=True)

        with torch.no_grad():
            for _ in range(max_new_tokens):
                outputs = self.model(generated_ids)
                next_token_logits = outputs.logits[:, -1, :]
                # sample token instead of argmax for diversity
                next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)

                generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)
                next_token_text = self.tokenizer.decode(next_token_id[0], skip_special_tokens=True)

                print(next_token_text, end="", flush=True)

                if next_token_id.item() == self.tokenizer.eos_token_id:
                    break

        print()  # newline after streaming
        output_ids = generated_ids[0][inputs.input_ids.shape[1]:]
        return self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()
    
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
            # Extract the top docs and their scores
            top_ranked = ranked[:RERANK_TOP_K]
            top_docs = [d for d, _ in top_ranked]
            top_scores = [s for _, s in top_ranked]

            # Calculate a simple confidence score based on the highest ranked score
            # You might use the mean, the max, or a threshold check depending on your reranker output
            # Here we use the score of the top-ranked document as the confidence score.
            confidence_score = top_scores[0] if top_scores else 0.0

        else:
            top_docs = []
            confidence_score = 0.0

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

        # Generate the answer
        if STREAM_GENERATION_TOKENS:
            answer = self.generator.generate_stream(prompt)
        else:
            answer = self.generator.generate(prompt)

        # Return the answer and the confidence score
        return answer, confidence_score


if __name__ == "__main__":
    print(DEVICE)
    rag = RAG()
    try:
        while True:
            q = input("Query> ").strip()
            if not q:
                continue
            answer, confidence = rag.generate_answer(q)
            # Print both the answer and the confidence score
            print(f"\nAnswer:\n{answer} [CONF:{confidence:.4f}]")
    except KeyboardInterrupt:
        print("\nExiting.")