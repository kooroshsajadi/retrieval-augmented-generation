import hashlib
from pathlib import Path
import json

def fingerprint(doc_content) -> str:
    return hashlib.md5(doc_content.encode()).hexdigest()

def get_unique_text_files(input_dir):
    """
    Returns a list of unique text file paths from input_dir based on fingerprints stored in summary.json.
    Ignores files with duplicate fingerprints and validates that the file exists in the input directory.

    Parameters:
        input_dir (Path): The input directory containing .txt files and the summary.json metadata file.

    Returns:
        List[Path]: List of unique text file paths (no duplicates by fingerprint).
    """
    summary_path = input_dir / 'cleaning_summary.json'
    if not summary_path.exists():
        raise FileNotFoundError(f"cleaning_summary.json does not exist in {input_dir}")

    with open(summary_path, 'r', encoding='utf-8') as f:
        summary = json.load(f)
    
    seen_fingerprints = set()
    unique_files = []
    for entry in summary:
        file_path = Path(entry['file_path'])
        fingerprint = entry['fingerprint']
        if (
            file_path.suffix == '.txt' and
            input_dir in file_path.parents and
            fingerprint not in seen_fingerprints and
            file_path.exists()
        ):
            unique_files.append(file_path)
            seen_fingerprints.add(fingerprint)
    return unique_files
