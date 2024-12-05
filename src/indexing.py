import json
import uuid

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings


class InMemoryIndex:

    EMBEDDINGS = OpenAIEmbeddings(model="text-embedding-3-large")

    def __init__(self, embeddings=EMBEDDINGS):
        self.embeddings = embeddings
        self.vector_store = InMemoryVectorStore(self.embeddings)

    def add_documents_from_report(self, report: json):
        texts = report['pre_text'] + report['post_text']
        table = report['table']

        # Add texts
        doc_ids = [str(uuid.uuid4()) for _ in texts]
        docs = [Document(page_content=text, metadata={"doc_id": doc_ids[i]}) for i, text in enumerate(texts)]

        # Add table
        table_id = str(uuid.uuid4())
        table_doc = [Document(id='table', page_content=str(table), metadata={"doc_id": table_id})]

        _ = self.vector_store.add_documents(documents=docs + table_doc)
