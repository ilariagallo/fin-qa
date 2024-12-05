from langchain import hub
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from typing_extensions import List, TypedDict
from langgraph.graph import START, StateGraph


class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


class QAGraph:

    PROMPT = hub.pull("rlm/rag-prompt")
    LLM = ChatOpenAI(model="gpt-4o-mini")

    def __init__(self, vector_store):
        graph = StateGraph(State).add_sequence([self.retrieve, self.generate])
        graph.add_edge(START, "retrieve")
        self.graph = graph.compile()
        self.vector_store = vector_store

    def retrieve(self, state: State):
        retrieved_docs = self.vector_store.similarity_search(state["question"])
        retrieved_table = self.vector_store.get_by_ids(["table"])
        return {"context": retrieved_docs + retrieved_table}

    def generate(self, state: State):
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        messages = self.PROMPT.invoke({"question": state["question"], "context": docs_content})
        response = self.LLM.invoke(messages)
        return {"answer": response.content}
