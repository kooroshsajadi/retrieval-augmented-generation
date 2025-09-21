import os
import shutil
import zipfile
import tempfile
import logging
from typing import List

class FileMover:
    """Moves all files, including those extracted from zip archives, from a source directory and its subdirectories to a destination directory."""
    
    def __init__(self, source_dir: str, dest_dir: str):
        """
        Initialize the FileMover with source and destination directories.
        
        Args:
            source_dir (str): Path to the source directory (path A).
            dest_dir (str): Path to the destination directory (path B).
        """
        self.source_dir = os.path.abspath(source_dir)
        self.dest_dir = os.path.abspath(dest_dir)
        self._setup_logging()
        self._ensure_directories()

    def _setup_logging(self):
        """Set up logging for the file mover."""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[logging.StreamHandler()]
        )
        self.logger = logging.getLogger(__name__)

    def _ensure_directories(self):
        """Ensure source and destination directories exist."""
        if not os.path.exists(self.source_dir):
            self.logger.error(f"Source directory does not exist: {self.source_dir}")
            raise FileNotFoundError(f"Source directory does not exist: {self.source_dir}")
        if not os.path.exists(self.dest_dir):
            self.logger.info(f"Creating destination directory: {self.dest_dir}")
            os.makedirs(self.dest_dir)

    def _extract_zip(self, zip_path: str, temp_dir: str) -> List[str]:
        """
        Extract files from a zip archive to a temporary directory and return their paths.
        
        Args:
            zip_path (str): Path to the zip file.
            temp_dir (str): Temporary directory for extraction.
        
        Returns:
            List[str]: List of extracted file paths.
        """
        extracted_files = []
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
                self.logger.info(f"Extracted zip: {zip_path} to {temp_dir}")
                
                # Collect all files from the extracted content (including nested folders)
                for root, _, files in os.walk(temp_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        extracted_files.append(file_path)
        except zipfile.BadZipFile:
            self.logger.error(f"Corrupt zip file: {zip_path}")
        except Exception as e:
            self.logger.error(f"Failed to extract {zip_path}: {e}")
        return extracted_files

    def get_all_files(self) -> List[str]:
        """
        Recursively collect all file paths in the source directory, including files extracted from zips.
        
        Returns:
            List[str]: List of file paths (non-zipped and extracted).
        """
        file_paths = []
        # Create a temporary directory for zip extraction
        with tempfile.TemporaryDirectory() as temp_dir:
            for root, _, files in os.walk(self.source_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    if file.lower().endswith('.zip'):
                        # Extract zip and collect its files
                        extracted_files = self._extract_zip(file_path, temp_dir)
                        file_paths.extend(extracted_files)
                    else:
                        # Add non-zip files directly
                        file_paths.append(file_path)
        return file_paths

    def move_files(self):
        """Move all files (non-zipped and extracted from zips) to the destination directory."""
        file_paths = self.get_all_files()
        
        if not file_paths:
            self.logger.info("No files found in source directory or zip archives.")
            return

        for file_path in file_paths:
            file_name = os.path.basename(file_path)
            dest_path = os.path.join(self.dest_dir, file_name)
            
            try:
                # Check for file name conflicts
                if os.path.exists(dest_path):
                    base, ext = os.path.splitext(file_name)
                    counter = 1
                    while os.path.exists(dest_path):
                        new_file_name = f"{base}_{counter}{ext}"
                        dest_path = os.path.join(self.dest_dir, new_file_name)
                        counter += 1
                    self.logger.warning(f"File name conflict for {file_name}. Renamed to {new_file_name}")

                # Ensure destination directory exists
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                shutil.move(file_path, dest_path)
                self.logger.info(f"Moved: {file_path} -> {dest_path}")
            except Exception as e:
                self.logger.error(f"Failed to move {file_path}: {e}")

def main():
    SOURCE_DIR = "data/prefettura_dump_16_07"
    DEST_DIR = "data/prefettura_v1"

    try:
        file_mover = FileMover(SOURCE_DIR, DEST_DIR)
        file_mover.move_files()
    except Exception as e:
        logging.error(f"Error in file moving process: {e}")

if __name__ == "__main__":
    main()