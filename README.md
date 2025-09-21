# Retrieval-Augmented Generation

This repository provides a fully modular implementation of a **Retrieval-Augmented Generation (RAG) pipeline** tailored for Italian legal-domain documents. The system handles the complete workflow: extracting and preprocessing raw text data, transforming it into dense vector representations, and storing embeddings efficiently in **Milvus** for retrieval. Beyond storage, it integrates a **hybrid retrieval approach** that combines BM25 with dense vector similarity, followed by reranking, to achieve high-quality, contextually relevant document retrieval.  

The pipeline is optimized for modern machine learning hardware accelerators (e.g., GPUs or specialized inference hardware), and parameters are configurable to adapt to different workloads.

Designed with modularity in mind, each component can be run independently as a Python module, while the entire pipeline can be orchestrated through a central entry point. This makes experimentation, debugging, and production deployment more flexible.  

The project is licensed under the **Apache License 2.0**, which permits both academic and commercial usage with proper attribution.

## 🧱 Components

### 🔀 Hybrid Retrieval

### 🛠️ Data Transformation
- [**sentence_transformer.py**](./src.embeddings.sentence_transformer.py): Generates embeddings from sentence chunks.

### 🗄️ Data Management
- [**vector_store.py**](./src.data.vector_store.py): Stores generated embeddings into a vector database, e.g., Milvus.

### 📝 Logging

Proper code is embedded in each script and execution logs for each is saved to `logs/` for debugging.
 
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
