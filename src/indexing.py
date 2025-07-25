import json
import uuid

import pandas as pd
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings


class InMemoryIndex:
    """Implements an in-memory index to store and retrieve document embeddings efficiently"""

    EMBEDDINGS = OpenAIEmbeddings(model="text-embedding-3-large")

    def __init__(self, embeddings=EMBEDDINGS):
        """
        Constructor for the in-memory index.
        Vector store is set to InMemoryVectorStore implemented by LangChain.

        :param embeddings: embedding function
        """
        self.embeddings = embeddings
        self.vector_store = InMemoryVectorStore(self.embeddings)

    def add_documents_from_report(self, report: json):
        """
        Adds chunks from a given financial report to the vector store.
        Non-textual information is converted to string before generating the embeddings.

        :param report: json file for a single financial report
        """
        texts = report['pre_text'] + report['post_text']
        table = report['table']

        # Add texts
        doc_ids = [str(uuid.uuid4()) for _ in texts]
        docs = [Document(page_content=text, metadata={"doc_id": doc_ids[i]}) for i, text in enumerate(texts)]

        # Add table
        table_docs = [Document(page_content=self.convert_table_to_str(table), metadata={"doc_id": str(uuid.uuid4())})]

        _ = self.vector_store.add_documents(documents=docs + table_docs)

    @staticmethod
    def convert_table_to_str(table):
        """
        Converts table to string.
        First, generates a dataframe for better formatting and then convert it to string.

        :param table: table from a financial record
        :return: string with table headers and rows
        """
        # Extract headers and rows
        headers = ["", *table[0][1:]]  # Add empty header for row labels
        rows = [row for row in table[1:]]

        # Create a DataFrame for better formatting
        df = pd.DataFrame(rows, columns=headers)

        # Format for aligned display
        table_str = df.to_string(index=False)
        return table_str
