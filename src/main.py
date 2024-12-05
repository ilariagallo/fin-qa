import pandas as pd

from src.data_loader import DataLoader
from src.indexing import InMemoryIndex
from src.qa_graph import QAGraph

if __name__ == "__main__":
    # Read dataset from JSON
    input_file = '../data/train.json'
    data_loader = DataLoader(input_file)

    outputs = [['question', 'actual_answer', 'expected_answer']]
    for i in range(5):
        # Extract single report
        report = data_loader.data[i]

        # Initialise vector store and add report to vector store
        index = InMemoryIndex()
        index.add_documents_from_report(report)

        # Predict
        qa_graph = QAGraph(index.vector_store)
        response = qa_graph.graph.invoke({"question": report['qa_0']['question']})
        print('Actual answer: ', response["answer"])
        print('Expected answer: ', report['qa_0']["answer"])
        out = [report['qa_0']['question'], response["answer"], report['qa_0']["answer"]]
        outputs.append(out)

    results = pd.DataFrame(outputs[1:], columns=outputs[0])
    results.to_csv('../data/results-solution-1.csv', index=False)
