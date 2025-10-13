# Retrieval-Augmented Generation

This repository provides a fully modular implementation of a **Retrieval-Augmented Generation (RAG) pipeline** tailored for Italian legal-domain documents. The system handles the complete workflow: extracting and preprocessing raw text data, transforming it into dense vector representations, and storing embeddings efficiently in **Milvus** for retrieval. Beyond storage, it integrates a **hybrid retrieval approach** that combines BM25 with dense vector similarity, followed by reranking, to achieve high-quality, contextually relevant document retrieval.  

The pipeline is optimized for modern machine learning hardware accelerators (e.g., GPUs or specialized inference hardware), and parameters are configurable to adapt to different workloads.

Designed with modularity in mind, each component can be run independently as a Python module, while the entire pipeline can be orchestrated through a central entry point. This makes experimentation, debugging, and production deployment more flexible.  

The project is licensed under the **Apache License 2.0**, which permits both academic and commercial usage with proper attribution.

## 🧱 Components

### ♻️ Deduplication

This component handles duplicate text detection to ensure storage efficiency and avoid redundant data in the vector database. The current implementation uses a signature-based deduplication strategy.

### 🧬 Embedding
This component generates dense vector embeddings for chunked texts using the `SentenceTransformer` library with a sekected model such as `dlicari/Italian-Legal-BERT-SC`. It supports parent-child chunking strategies for hierarchical retrieval, processes directories or individual files with configurable max/min chunk lengths, and saves normalized, truncated embeddings while handling metadata like word counts and parent IDs. Optimized for hardware accelerators (prioritizing XPU or GPU with CPU fallback), the module includes comprehensive logging for device usage, embedding generation, and error recovery to ensure robust, modular operation in RAG pipelines.

'''on OOM errors via torch cache clearing and temporary device migration'''

### 🔀 Hybrid Retrieval

This component implements hybrid retrieval by combining BM25 sparse retrieval for keyword matching with dense vector similarity search using Milvus embeddings to capture semantic relevance in texts. Retrieved candidates from both methods are fused using a weighted score.

### 🎯 Reranker

This component refines fused candidates from hybrid retrieval using a cross-encoder model, such as `dlicari/Italian-Legal-BERT`, which jointly embeds query and chunk pairs to compute nuanced relevance scores beyond initial similarity metrics.

### 📝 Logging

Logging is embedded in each module using Python's built-in logging library, with loggers named by module to enable granular control and avoid root logger conflicts. Messages are categorized by levels (DEBUG for diagnostics, INFO for workflow tracking, WARNING/ERROR for issues) and output to console for real-time monitoring while persisting to timestamped files in `logs/` for auditing and debugging in the RAG pipeline.
 
## 🚀 How to Run the Project

This project follows a modular structure, and each script can be executed as a module using Python's `-m` flag from the repository's root directory. This approach ensures that relative imports are resolved correctly.  

To run the complete RAG pipeline (after installing all required dependencies), execute:

```
python -m main
```

## 🚧 Status

> **Note**: This project is **under active development**. Expect changes in structure and functionality in the near future. Currently focused on document deduplication management.

## 📄 License

This project is licensed under the [Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0) (the "License"). You may not use, copy, modify, or distribute this project except in compliance with the License. A copy of the License is included in the [LICENSE](./LICENSE) file in this repository.

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations.
