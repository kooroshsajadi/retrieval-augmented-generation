import logging
from typing import List, Dict, Any, Optional, Tuple
from pymilvus import connections, Collection, MilvusException
from src.utils.logging_utils import setup_logger
import time

class MilvusConnector:
    """Handles connection and search operations for Milvus vector database."""

    def __init__(
        self,
        collection_name: str = "legal_texts",
        milvus_host: str = "localhost",
        milvus_port: str = "19530",
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize MilvusConnector.

        Args:
            collection_name (str): Milvus collection name.
            milvus_host (str): Milvus server host.
            milvus_port (str): Milvus server port.
            logger (logging.Logger, optional): Logger instance.
        """
        self.logger = logger or setup_logger("src.retrieval.milvus_connector")
        self.collection_name = collection_name

        # Connect to Milvus
        try:
            connections.connect(alias="default", host=milvus_host, port=milvus_port)
            self.collection = Collection(collection_name)
            self.collection.load()
            self.logger.info(f"Connected to Milvus collection: {collection_name}")
        except Exception as e:
            self.logger.error(f"Failed to connect to Milvus: {str(e)}")
            raise

    def search(self, query_vector: List[float], top_k: int = 5, output_fields: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Perform vector search in Milvus collection.

        Args:
            query_vector (List[float]): Query embedding vector.
            top_k (int): Number of chunks to retrieve.
            output_fields (List[str], optional): Fields to retrieve from Milvus. Defaults to ["chunk_id", "text", "parent_id", "parent_file_path"].

        Returns:
            List[Dict[str, Any]]: Retrieved chunks with requested fields and distance.
        """
        try:
            search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
            output_fields = output_fields or ["chunk_id", "text", "parent_id", "parent_file_path"]
            results = self.collection.search(
                data=[query_vector],
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                output_fields=output_fields
            )
            retrieved = [
                {
                    "chunk_id": hit.entity.get("chunk_id", ""),
                    "text": hit.entity.get("text", ""),
                    "parent_id": hit.entity.get("parent_id", ""),
                    "parent_file_path": hit.entity.get("parent_file_path", ""),
                    "distance": hit.distance
                }
                for hit in results[0]
            ]
            self.logger.info(f"Retrieved {len(retrieved)} chunks")
            return retrieved
        except Exception as e:
            self.logger.error(f"Search failed: {str(e)}")
            raise

    def get_all_texts(self) -> Tuple[List[str], List[str], List[str], List[str]]:
        """
        Retrieve all texts, chunk IDs, parent IDs, and parent file paths from the Milvus collection.
    
        Returns
        -------
        Tuple[List[str], List[str], List[str], List[str]]
            Lists of texts, chunk_ids, parent_ids, and parent_file_paths.
    
        Raises
        ------
        MilvusException
            If the query fails after all retries or the Milvus server returns an error.
    
        Notes
        -----
        This function paginates queries with strict adherence to Milvus's
        max query result window: (offset + limit) <= 16384 per query.
        Empty texts are skipped by default. Retries with exponential backoff are included.
        """
        
        texts, chunk_ids, parent_ids, parent_file_paths = [], [], [], []
        offset = 0
        base_limit = 1000  # Adjustable batch size, must be << 16384
        max_window = 16384    # Milvus max query window
        retries = 3
        while True:
             # Compute the largest possible limit for this batch that does not exceed max_window
            limit = min(base_limit, max_window - offset)
            if limit <= 0:
                break  # Prevents overrun of max_window
            for attempt in range(retries):
                try:
                    # Query all entities with pagination
                    result = self.collection.query(
                        expr="",  # Empty expression to fetch all
                        offset=offset,
                        limit=limit,
                        output_fields=["chunk_id", "text", "parent_id", "parent_file_path"]
                    )
                    # Extract and append the batch results.
                    for hit in result:
                        text = hit.get("text", "")
                        if not text:
                            self.logger.warning(f"Skipping entry with empty text for chunk_id: {hit.get('chunk_id')}")
                            continue
                        texts.append(text)
                        chunk_ids.append(hit.get("chunk_id", ""))
                        parent_ids.append(hit.get("parent_id", ""))
                        parent_file_paths.append(hit.get("parent_file_path", ""))
                    break  # Success: exit retry loop
                except MilvusException as e:
                    self.logger.error(f"Attempt {attempt + 1} to fetch texts failed: {str(e)}")
                    if attempt < retries - 1:
                        time.sleep(5 * (attempt + 1))  # Exponential backoff.
                        continue # Retry next attempt
                    else:
                        raise
                except Exception as e:
                    self.logger.error(f"Unexpected error fetching texts: {str(e)}")
                    raise
            offset += limit
            if len(result) < limit:
                break # No more data to fetch
        self.logger.info(f"Retrieved {len(texts)} texts from collection {self.collection.name}")
        return texts, chunk_ids, parent_ids, parent_file_paths