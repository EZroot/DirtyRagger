from typing import List

def chunk_text(text: str, tokenizer, chunk_size: int = 512, overlap: int = 64) -> List[str]:
    """
    Token-level chunking using a provided tokenizer.
    - tokenizer: must provide encode/decode methods compatible with HuggingFace tokenizers
                 (e.g. sentence_transformer.tokenizer or AutoTokenizer).
    """
    # encode -> list[int]
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(token_ids) == 0:
        return []

    chunks = []
    start = 0
    n = len(token_ids)

    while start < n:
        end = start + chunk_size
        token_chunk = token_ids[start:end]
        # decode token ids to text
        # use tokenizer.decode which exists on HF tokenizers
        chunk_text = tokenizer.decode(token_chunk, clean_up_tokenization_spaces=True, skip_special_tokens=True)
        chunks.append(chunk_text)

        if end >= n:
            break
        start += chunk_size - overlap

    return chunks
