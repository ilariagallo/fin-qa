# FinQA: A Question-Answering Solution for Financial Reports

This repository implements a solution for question answering over financial reports using the ConvFinQA dataset. For more details about the dataset, refer to the original repository: [ConvFinQA GitHub Repository](https://github.com/czyssrs/ConvFinQA/).

## System Architecture

The solution employs a Retrieval-Augmented Generation (RAG) system, which integrates Large Language Models (LLMs) with traditional retrieval mechanisms to answer questions about specific sources of information. The architecture consists of three primary components:

1. **Indexing**: Processes and encodes documents into numerical representations (embeddings) and stores them in an index for efficient retrieval.
2. **Retrieval**: Handles user queries at runtime by retrieving the most relevant information from the indexed documents.
3. **Generation**: Utilizes the LLM to generate answers based on the retrieved documents and the user’s question.

### Indexing

The indexing phase includes several key steps:

#### Data Loading

The system loads data from the `train.json` file into memory and preprocesses it to standardize key structures. Specifically, the dataset contains question-answer fields with inconsistent naming conventions: some samples use `qa_0` and `qa_1`, while others use `qa`. To ensure uniformity, all single-field instances of `qa` are renamed to `qa_0`, enabling seamless iteration over dataset elements.

For implementation details, see the `data_loader.py` module.

#### Splitting

Reports are divided into chunks, with each chunk corresponding to a single element from the "pre_text" or "post_text" arrays. Additionally, each table within the report is treated as a unique chunk.

#### Embeddings

The solution uses the `OpenAIEmbeddings` model (specifically, `text-embedding-3-large`) to convert each document chunk into a numerical representation. For tables, the content is first converted to a string format where each line represents a table row before embeddings are generated.

#### Storing

Documents are stored in a vector store to facilitate efficient retrieval. For this prototype, an in-memory vector store (`InMemoryVectorStore`) is employed, as the focus is on conducting local analyses of the solution’s performance. However, for scalability and production-readiness, a more robust and efficient vector store, such as OpenSearch, should be adopted.

For details on splitting, embedding, and storing processes, refer to the `indexing.py` module.

