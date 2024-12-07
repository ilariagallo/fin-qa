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

### Retrieval & Generation

The retrieval and generation steps are orchestrated using a graph-based approach, which sequences the two steps: retrieval followed by generation.

*(Placeholder: Add an illustrative graph of the process here.)*

#### Retrieval

The retrieval process leverages the similarity search functionality of the `InMemoryVectorStore`. This functionality computes cosine similarity between the query vector and the document embeddings stored in the vector store to identify the top-k most relevant documents. Several approaches were explored for the retrieval step:

- **Approach 1**: Perform similarity search with a default value of `k=4` on documents derived from the `pre_text` and `post_text` paragraphs, while always including the table (retrieved by ID) as an additional document. This results in a total of five documents. However, this approach has limitations in generalizing to financial reports containing multiple tables.
- **Approach 2**: Perform similarity search across all documents, including the table document, and allow the similarity search to determine the top-k documents. In this approach, `k` is set to `8` to improve recall. This method was selected for the final solution due to its ability to generalize effectively to reports with multiple tables.

Detailed results comparing these two approaches can be found in the *Experiment Setup & Results* section.

#### Generation

In the generation step, an LLM is utilized to formulate the final answer to the user’s query. The chosen prompt, `rlm/rag-prompt`, is sourced from the LangChain hub, which offers a variety of task-specific prompts. The prompt provides a concise description of the task and takes the user’s query and retrieved documents as input context.

```
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

Question: {question} 

Context: {context} 

Answer:
```

For the model, experiments were conducted using both `gpt-4o-mini` and `gpt-4o`. While `gpt-4o-mini` demonstrated strong mathematical reasoning capabilities, `gpt-4o`, with its larger parameter size (1.8 trillion vs. ~10 billion for `gpt-4o-mini`), yielded superior results. Consequently, `gpt-4o` was selected for the final solution.

Further details and performance comparisons can be found in the *Experiment Setup & Results* section.

### Inference

Functionality has been implemented to support inference on both individual reports and batches of reports. For each report, the indexing and storing steps are performed dynamically, as the objective is to process each report independently. 

If the solution were extended to retrieve information across the entire dataset, it would be more efficient to compute the indexing and storing steps offline. This approach would save both time and resources by eliminating the need to recompute embeddings repeatedly. 

The results of the inference process are saved in the `/results` folder.

### Evaluation

To evaluate the accuracy of the answers generated by the solution, an LLM-based evaluation approach was employed. The evaluation prompt, sourced from the LangChain hub (`langchain-ai/rag-answer-vs-reference`), takes as input the question, ground truth answer, and generated answer. It assigns a score of `0` for incorrect answers, `1` for correct answers, and provides a brief explanation for the score.

To enhance the granularity of the evaluation, the prompt was modified to include a score of `0.5` for relevant answers containing minor approximation errors (within ±0.5). This custom prompt can be found in `prompt.py`.

Finally, a detailed evaluation report was generated. The report summarizes the evaluation results, including the count and percentage of samples for each score category.


### Experiment Setup & Results

Three different approaches were evaluated using the previously described evaluation framework. Each approach was tested on the first 1,000 financial reports from `train.json` to control resource costs.

#### Experiment 1

This experiment involved performing similarity search with a default value of `k=4` on documents derived from the `pre_text` and `post_text` paragraphs, while always including the table (retrieved by ID) as an additional document. This resulted in a total of five documents. As anticipated, this approach showed limitations in generalizing to financial reports containing multiple tables. The LLM used for the generation step in this experiment was `gpt-4o-mini`.

| Score | Count | Percentage (%) |
|-------|-------|----------------|
| 0.0   | 405   | 40.5           |
| 0.5   | 156   | 15.6           |
| 1.0   | 439   | 43.9           |

*Table 1: Distribution of scores for the first proposed approach.*


Representative **examples** of answers evaluated by the LLM and assigned a **score of 0.5**:


| Question | Actual Answer | Expected Answer | Score | Explanation |
|----------|---------------|-----------------|-------|-------------|
| what was the percentage change in the net cash used in investing activities from 2006 to 2007 | "The net cash used in investing activities decreased from -$1460 million in 2006 to -$1367 million in 2007. This represents a percentage change of approximately 6.36% decrease." | -6.4% | 0.5 | "The student's answer of -6.4% is close to the correct percentage change of approximately -6.36%. While it is not exactly accurate, it falls within an acceptable margin of ±0.5%. The student's answer does not contain conflicting statements and is factually aligned with the context of the ground truth answer." |
| what was the percentage change in rent expenses included in selling , general and administrative expense from 2014 to 2015? | "The rent expense included in selling, general, and administrative expense was $59.0 million in 2014 and $83.0 million in 2015. This represents a percentage change of approximately 40.68% from 2014 to 2015." | 41% | 0.5 | "The student's answer of 41% is close to the ground truth answer of approximately 40.68%, falling within the acceptable margin of ±0.5%. However, it lacks the detail provided in the ground truth regarding the specific rent expense amounts for both years. Therefore, while the percentage is accurate within the margin, the answer does not fully meet all criteria due to the omission of key information." |
| as of december 31 , 2017 what was the percent of the system energy credit facility utilization | "As of December 31, 2017, the utilization of the system energy credit facility was approximately 56.8%. This is calculated based on $17.8 million in letters of credit and $50 million in loans outstanding against a total credit facility of $120 million." | 56.7% | 0.5 | "The student's answer of 56.7% is very close to the ground truth answer of 56.8%, falling within the acceptable margin of ±0.5%. However, the student did not provide the full context or calculation related to the total credit facility, letters of credit, and loans outstanding, which would have strengthened their answer. Therefore, while the percentage itself is accurate within the margin, the lack of additional details affects the overall completeness of the response." |

Results from this experiment indicate that **59.6% of the answers were relevant**, with 15.6% affected by minor approximation errors.

#### Experiment 2

In this experiment, similarity search was performed across all documents, including the table document, allowing the similarity search to determine the top-k documents. Here, `k` was set to `8` to enhance recall. Given the concise nature of the documents, this configuration remained computationally reasonable. The LLM used in this experiment was again `gpt-4o-mini`.

| Score | Count | Percentage (%) |
|-------|-------|----------------|
| 0.0   | 414   | 41.4           |
| 0.5   | 148   | 14.8           |
| 1.0   | 438   | 43.8           |

*Table 2: Distribution of scores for the second proposed approach.*

Results from this experiment closely resemble those of the first, with **58.6% of the answers being relevant** and 14.8% exhibiting minor approximation errors. The increased `k` value (8) improved the likelihood of including all relevant information, such as the table, in the context.

#### Experiment 3

The final experiment retained the same retrieval configuration as Experiment 2 but used `gpt-4o` as the LLM for generation. The larger, more capable model was expected to yield more accurate results when provided with sufficient context.

| Score | Count | Percentage (%) |
|-------|-------|----------------|
| 0.0   | 355   | 35.5           |
| 0.5   | 109   | 10.9           |
| 1.0   | 536   | 53.6           |

*Table 3: Distribution of scores for the third proposed approach.*

This configuration delivered the best results, with **64.5% of the answers being relevant** and only 10.9% affected by minor approximation errors. These results demonstrate that the retrieval step is effectively including relevant context in the prompt. Additionally, the improved mathematical and reasoning capabilities of the larger `gpt-4o` model contributed significantly to better performance. This highlights the potential for further advancements by leveraging more powerful models in the generation step.









