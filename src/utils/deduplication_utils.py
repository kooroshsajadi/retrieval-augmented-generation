import hashlib

# Fingerprinting for deduplication
def fingerprint(doc_content) -> str:
    return hashlib.md5(doc_content.encode()).hexdigest()