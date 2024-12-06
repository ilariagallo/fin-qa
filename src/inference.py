import json

import pandas as pd

from src.data_loader import DataLoader
from src.indexing import InMemoryIndex
from src.qa_graph import QAGraph


class Inference:
    """
    A class to return model predictions
    """

    output_columns = ['question', 'actual_answer', 'expected_answer']

    def run_inference(self, sample_report):
        """
        Run inference on a single financial report

        :param sample_report: single financial report including report content, question and answer
        :return: question, actual answer, expected answer
        """
        # Create index for report and add documents
        in_memory_index = InMemoryIndex()
        in_memory_index.add_documents_from_report(sample_report)

        # Predict
        qa_graph = QAGraph(in_memory_index.vector_store)
        response = qa_graph.graph.invoke({"question": sample_report['qa_0']['question']})
        print('Actual answer: ', response["answer"])
        print('Expected answer: ', sample_report['qa_0']["answer"])

        return [sample_report['qa_0']['question'], response["answer"], sample_report['qa_0']["answer"]]

    def run_batch_inference(self, dataset: json):
        """
        Run batch inference on the entire dataset provided as input

        :param dataset: dataset, including multiple reports
        """
        predictions = []
        for report in dataset:
            output = self.run_inference(report)
            predictions.append(output)

        predictions_df = pd.DataFrame(predictions, columns=self.output_columns)
        return predictions_df


if __name__ == "__main__":
    # Read dataset from JSON
    input_file = '../data/train.json'
    data_loader = DataLoader(input_file)

    inference = Inference()
    predictions = inference.run_batch_inference(data_loader.data)

    results_folder = '../data/results/'
    predictions.to_csv(results_folder + 'results-solution-test.csv', index=False)
