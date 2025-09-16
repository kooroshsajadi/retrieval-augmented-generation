from typing import List, Dict, Any, Callable

def create_parent_chunks(
    text: str,
    sentences: List[str],
    max_tokens: int,
    count_tokens: Callable[[str], int],
    is_valid_chunk: Callable[[str], bool]
) -> List[Dict[str, Any]]:
    """
    Create parent and child chunks, where parent is the full text and children are sentences.

    Args:
        text (str): Full input text (parent chunk).
        sentences (List[str]): List of sentences (child chunks).
        max_tokens (int): Maximum tokens per chunk.
        count_tokens (Callable[[str], int]): Function to count tokens in text.
        is_valid_chunk (Callable[[str], bool]): Function to validate chunk.

    Returns:
        List[Dict[str, Any]]: List of chunks with parent and child metadata.
    """
    chunks = []
    parent_id = f"parent_{hash(text) % 1000000}"  # Unique parent ID
    parent_token_count = count_tokens(text)

    # Add parent chunk (full text, if within max_tokens)
    if parent_token_count <= max_tokens:
        chunks.append({
            "text": text,
            "word_count": len(text.split()),
            "char_length": len(text),
            "token_count": parent_token_count,
            "is_valid": is_valid_chunk(text),
            "parent_id": None,
            "chunk_type": "parent"
        })

    # Create child chunks (sentences)
    for i, sentence in enumerate(sentences):
        token_count = count_tokens(sentence)
        if token_count <= max_tokens:
            chunks.append({
                "text": sentence,
                "word_count": len(sentence.split()),
                "char_length": len(sentence),
                "token_count": token_count,
                "is_valid": is_valid_chunk(sentence),
                "parent_id": parent_id,
                "chunk_type": "child"
            })

    return chunks