from typing import List, Dict, Any, Callable

def create_sentence_based_chunks(
    sentences: List[str],
    max_chunk_words: int,
    max_tokens: int,
    count_tokens: Callable[[str], int],
    is_valid_chunk: Callable[[str], bool]
) -> List[Dict[str, Any]]:
    """
    Create chunks from sentences, respecting max_chunk_words and max_tokens.

    Args:
        sentences (List[str]): List of sentences.
        max_chunk_words (int): Maximum words per chunk.
        max_tokens (int): Maximum tokens per chunk.
        count_tokens (Callable[[str], int]): Function to count tokens in text.
        is_valid_chunk (Callable[[str], bool]): Function to validate chunk.

    Returns:
        List[Dict[str, Any]]: List of chunks with metadata.
    """
    chunks = []
    current_chunk = []
    current_word_count = 0
    current_token_count = 0

    for sentence in sentences:
        word_count = len(sentence.split())
        token_count = count_tokens(sentence)

        if (current_word_count + word_count <= max_chunk_words and
            current_token_count + token_count <= max_tokens):
            current_chunk.append(sentence)
            current_word_count += word_count
            current_token_count += token_count
        else:
            if current_chunk:
                chunk_text = " ".join(current_chunk)
                chunks.append({
                    "text": chunk_text,
                    "word_count": current_word_count,
                    "char_length": len(chunk_text),
                    "token_count": current_token_count,
                    "is_valid": is_valid_chunk(chunk_text),
                    "parent_id": None  # No parent for sentence-based
                })
            current_chunk = [sentence]
            current_word_count = word_count
            current_token_count = token_count

    # Add the last chunk
    if current_chunk:
        chunk_text = " ".join(current_chunk)
        chunks.append({
            "text": chunk_text,
            "word_count": current_word_count,
            "char_length": len(chunk_text),
            "token_count": current_token_count,
            "is_valid": is_valid_chunk(chunk_text),
            "parent_id": None
        })

    return chunks