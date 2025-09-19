from typing import List, Dict, Any, Optional
from pathlib import Path
import hashlib
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

class ParentChildChunking:
    """Implements parent-child chunking approach using LangChain."""

    def __init__(
        self,
        text_dir: str,
        max_tokens: int = 768,
        max_chunk_length: int = 2000
    ):
        """
        Initialize ParentChildChunking.

        Args:
            text_dir (str): Path to text directory.
            max_tokens (int): Max tokens for child chunks (default 768).
            max_chunk_length (int): Max chunk length for parents in characters (default 2000).
        """
        self.text_dir = Path(text_dir)
        self.max_tokens = max_tokens
        self.max_chunk_length = max_chunk_length

        # Parent splitter: Larger chunks
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_chunk_length,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )

        # Child splitter: Smaller chunks, adjusted for max_tokens
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_tokens - 100,  # Buffer for tokens
            chunk_overlap=50,
            separators=["\n\n", "\n", " ", ""]
        )

    def load_documents(self, file_path: Optional[str] = None) -> List[Document]:
        """
        Load documents from text directory or a single file as LangChain Documents.

        Args:
            file_path (str, optional): Path to a single text file. If None, load all files in text_dir.

        Returns:
            List[Document]: List of loaded documents.
        """
        documents = []
        if file_path:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
            doc = Document(page_content=text, metadata={"source": str(file_path)})
            documents.append(doc)
        else:
            for file_path in self.text_dir.glob("*.txt"):
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()
                doc = Document(page_content=text, metadata={"source": str(file_path)})
                documents.append(doc)
        return documents

    def chunk(self, file_path: str, count_tokens: callable, is_valid_chunk: callable, file_stem: str) -> List[Dict[str, Any]]:
        """
        Perform parent-child chunking for a single file.

        Args:
            file_path (str): Path to the input text file.
            count_tokens (callable): Function to count tokens in text.
            is_valid_chunk (callable): Function to validate chunk.
            file_stem (str): Stem of the file name for chunk_id generation.

        Returns:
            List[Dict[str, Any]]: List of chunks with metadata.
        """
        chunks = []
        doc = self.load_documents(file_path=file_path)[0]
        parents = self.parent_splitter.split_documents([doc])
        
        for parent_idx, parent in enumerate(parents, 1):
            parent_chunk_name = f"{file_stem}_parent_{parent_idx:03d}"
            parent_chunk_id = hashlib.md5(parent_chunk_name.encode()).hexdigest()  # Hash parent_chunk_id
            
            # Add child chunks
            children = self.child_splitter.split_text(parent.page_content)
            for child_idx, child_text in enumerate(children, 1):
                child_chunk_name = f"{file_stem}_chunk_{parent_idx:03d}_{child_idx:03d}"
                child_chunk_id = hashlib.md5(child_chunk_name.encode()).hexdigest()
                chunks.append({
                    "text": child_text,
                    "word_count": len(child_text.split()),
                    "char_length": len(child_text),
                    "token_count": count_tokens(child_text),
                    "is_valid": is_valid_chunk(child_text),
                    "parent_id": parent_chunk_id,
                    "chunk_id": child_chunk_id,
                    "chunk_type": "child",
                    "parent_file_name": parent_chunk_name,
                    "file_name": child_chunk_name,
                })
            
            # Add parent chunk
            file_id = hashlib.md5(file_stem.encode()).hexdigest()
            chunks.append({
                "text": parent.page_content,
                "word_count": len(parent.page_content.split()),
                "char_length": len(parent.page_content),
                "token_count": count_tokens(parent.page_content),
                "is_valid": is_valid_chunk(parent.page_content),
                "chunk_type": "parent",
                "chunk_id": parent_chunk_id,
                "parent_id": file_id,
                "parent_file_name": file_stem,
                "file_name": parent_chunk_name
            })
        
        return chunks