# Retrieval-Augmented Generation

This project implements a Retrieval-Augmented Generation (RAG) pipeline for processing Italian PDFs, extracting text, and preparing data for GraphDB/Neo4j integration. Licensed under the Apache License 2.0.

## 🗄️ Data Ingestion (Data Layer)

- [**file_classifier.py**](./src/ingestion/file_classifier.py): Classifies PDFs as text-based or image-based using `pdfplumber` and saves the result as metadata for later usage.
- [**text_ingestor.py**](./src/ingestion/text_ingestor.py): Extracts text from files using `pdfplumber` (text-based), `Tesseract OCR` (image-based, `lang="ita"`), and `PyMuPDF` (images).
- [**text_cleaner.py**](./src/ingestion/text_cleaner.py): Cleans the extracted text.
- [**text_chunker.py**](./src/ingestion/text_chunker.py): Splits cleaned text into sentence-based chunks.

## 📝 Logging

Proper code is embedded in each script and execution logs for each is saved to `logs/` for debugging.

## 📚 Data Sources

- **Knowledge Graph of the Italian Legislation**
  - Colombo, A. (2024). Knowledge Graph of the Italian Legislation [Data set]. Zenodo. DOI: 10.5281/zenodo.13798158

## 📄 License

This project is licensed under the [Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0) (the "License"). You may not use, copy, modify, or distribute this project except in compliance with the License. A copy of the License is included in the [LICENSE](./LICENSE) file in this repository.

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations.
