from enum import Enum
from typing import Optional

TEXT_EXTENSIONS = {'.txt', '.text'}
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.jfif', '.bmp', '.tiff', '.gif'}
PDF_TYPE = {'text-based', 'image-based', 'unknown'}

class FileType(Enum):
    PDF = "pdf"
    IMAGE = "image"
    TXT = "txt"
    P7M = "p7m"
    UNKNOWN = "unknown"

class PDFType(Enum):
    TEXT_BASED = "text-based"
    IMAGE_BASED = "image-based"
    UNKNOWN = "unknown"

class IngestionMetadata:
    def __init__(
            self,
            file_path: Optional[str] = None,
            file_name: Optional[str] = None,
            file_type: Optional[FileType] = None,
            pdf_type: Optional[PDFType] = None,
            page_metadata: Optional[list] = None,
            is_valid: bool = False,
            is_supported: bool = True,
            error: Optional[str] = None,):
        
        self.file_path = file_path
        self.file_name = file_name
        self.file_type = file_type
        self.pdf_type = pdf_type
        self.page_metadata = page_metadata if page_metadata is not None else []
        self.is_valid = is_valid
        self.is_supported = is_supported
        self.error = error
