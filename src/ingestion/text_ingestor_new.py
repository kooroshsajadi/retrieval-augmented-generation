import json
from pathlib import Path
from src.utils.ingestion_metadata import IngestionMetadata, FileType
from src.utils.ingestion.source_classifier import SourceClassifier
import src.utils.ingestion.text_ingestors as ext
from typing import Optional
from src.utils.logging_utils import setup_logger
import logging


class TextIngestor:
    def __init__(self, input_dir, output_dir, output_metadata_file,
                 language="ita", logger:Optional[logging.Logger]=None):
        
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_metadata_file = Path(output_metadata_file)
        self.classifier = SourceClassifier()
        self.language = language
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger or setup_logger("src.ingestion.text_ingestor")

    def process_directory(self):
        aggregated_metadata = []
        for file_path in self.input_dir.iterdir():
            if not file_path.is_file():
                continue
            cls = self.classifier.classify_file(file_path)
            meta = IngestionMetadata(
                file_path=str(file_path),
                file_name=file_path.name,
                file_type=FileType(cls["file_type"]) if cls["file_type"] in FileType._value2member_map_ else FileType.UNKNOWN,
                page_metadata=[],
                is_valid=True,
                error=None
            )
            if not cls["is_supported"]:
                meta.error = cls["error"]
                meta.is_valid = False
                meta.page_metadata = []
                self.logger.warning("Unsupported file type for %s", file_path.name)
            else:
                if meta.file_type == FileType.PDF:
                    text, page_md, valid = ext.extract_text_with_pdfplumber(file_path)
                    if not valid:
                        text, page_md, valid = ext.extract_text_with_ocr(file_path, language=self.language)
                elif meta.file_type == FileType.TXT:
                    text, page_md, valid = ext.extract_text_from_txt(file_path)
                elif meta.file_type == FileType.IMAGE:
                    text, page_md, valid = ext.extract_text_image_with_pymupdf(file_path)
                else:
                    text, page_md, valid = "", [], False
                meta.page_metadata = page_md
                if not valid:
                    meta.is_valid = False
                    self.logger.warning("Text extraction failed for %s", file_path.name)
            # Save extracted text only if not empty (independent of meta.is_valid)
            if len(text.strip()) > 0:
                out_path = self.output_dir / (file_path.stem + ".txt")
                out_path.write_text(text, encoding="utf-8")
                self.logger.info(f"Text extraction was successful for {file_path.name}")
            else:
                self.logger.warning(f"No text extracted from {file_path.name} but not marked as invalid.")
            # Convert enum fields to their value before appending
            meta_dict = meta.__dict__.copy()
            if isinstance(meta_dict.get("file_type"), FileType):
                meta_dict["file_type"] = meta_dict["file_type"].value
            aggregated_metadata.append(meta_dict)
    
        with open(self.output_metadata_file, "w", encoding="utf-8") as f:
            json.dump(aggregated_metadata, f, ensure_ascii=False, indent=2)



if __name__ == "__main__":
    ingestor = TextIngestor(
        input_dir=Path("data/prefettura_v1_files"),
        output_dir=Path("data/prefettura_v1.3_texts"),
        output_metadata_file=Path("data/metadata/extraction_prefettura_v1.3.json"),
        language='ita'
    )
    ingestor.process_directory()
    print("Text ingestion and classification completed. Metadata saved.")
