from enum import Enum
from typing import Optional

TEXT_EXTENSIONS = {'.txt', '.text'}
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.jfif', '.bmp', '.tiff', '.gif'}

class FileType(Enum):
    PDF = "pdf"
    IMAGE = "image"
    TXT = "txt"
    UNKNOWN = "unknown"

class IngestionMetadata:
    def __init__(
            self,
            file_path: Optional[str] = None,
            file_name: Optional[str] = None,
            file_type: Optional[FileType] = None,
            page_metadata: Optional[list] = None,
            is_valid: bool = False,
            error: Optional[str] = None):
        
        self.file_path = file_path
        self.file_name = file_name
        self.file_type = file_type
        self.page_metadata = page_metadata if page_metadata is not None else []
        self.is_valid = is_valid
        self.error = error
