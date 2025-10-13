import hashlib
from pathlib import Path
import json

def fingerprint(doc_content) -> str:
    return hashlib.md5(doc_content.encode()).hexdigest()

def get_unique_text_files(input_dir, cleaning_summary_path: Path) -> list[Path]:
    """
    Returns a list of unique text file paths from input_dir based on fingerprints stored in summary.json.
    Ignores files with duplicate fingerprints and validates that the file exists in the input directory.

    Parameters:
        input_dir (Path): The input directory containing .txt files and the summary.json metadata file.
        cleaning_summary_path (Path): Path to the cleaning summary JSON file.

    Returns:
        List[Path]: List of unique text file paths (no duplicates by fingerprint).
    """

    if not cleaning_summary_path.exists():
        raise FileNotFoundError(f"Cleaning summary file not found: {cleaning_summary_path}")

    with open(cleaning_summary_path, 'r', encoding='utf-8') as f:
        summary = json.load(f)
    
    seen_fingerprints = set()
    unique_files = []

    # Total number of the cleaned files.
    print(f"Total number of the cleaned files: {len(summary)}")

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
    
    print(f"Number of unique text files after deduplication: {len(unique_files)}")
    return unique_files
