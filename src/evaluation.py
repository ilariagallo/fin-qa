import pandas as pd
from langchain import hub
from langchain_openai import ChatOpenAI

from src.prompts import fin_qa_evaluation_prompt


class AnswerEvaluator:
    """
    A simple evaluator for RAG answer accuracy
    """

    def __init__(self, custom_prompt=None):
        self.llm = ChatOpenAI(model="gpt-4o-mini")
        self.grader_prompt = hub.pull("langchain-ai/rag-answer-vs-reference")
        self.grader_prompt.messages[0].prompt.template = custom_prompt
        self.answer_grader = self.grader_prompt | self.llm

    def evaluate_answer(self, qa: pd.Series) -> list:
        """
        Runs evaluation on a single question

        :param qa: pd.Series including question, actual answer and expected answer for a single question
        :returns score and explanation
        """

        # Get question, ground truth answer, RAG chain answer
        input_question = qa['question']
        reference = qa['expected_answer']
        prediction = qa['actual_answer']

        # Run evaluator
        evaluation = self.answer_grader.invoke({"question": input_question,
                                                "correct_answer": reference,
                                                "student_answer": prediction})
        score = evaluation["Score"]
        explanation = evaluation['Explanation']
        print(score)

        return [score, explanation]

    def run(self, input_filepath: str, output_filepath: str):
        """
        Runs evaluation on all predictions in the input file

        :param input_filepath: filepath to file with the predictions
        :param output_filepath: filepath to file with scores and explanations
        """
        predictions = pd.read_csv(input_filepath)
        scores = predictions.apply(lambda row: self.evaluate_answer(row), axis=1)
        scores_df = pd.DataFrame(data=list(scores), columns=['score', 'explanation'])

        final_df = pd.concat([predictions, scores_df], axis=1)
        final_df.to_csv(output_filepath, index=False)

    @staticmethod
    def create_report(evaluation_filepath: str, report_filepath: str):
        """
        Creates report with evaluation metrics

        :param evaluation_filepath: filepath to file with scores and explanations
        :param report_filepath: filepath to report file
        """
        evaluation = pd.read_csv(evaluation_filepath)
        scores = evaluation['score']

        # Count occurrences of each value
        counts = scores.value_counts().sort_index()

        # Calculate percentages
        percentages = (counts / len(scores)) * 100

        # Combine counts and percentages into a DataFrame
        result = pd.DataFrame({'Count': counts, 'Percentage (%)': percentages})

        result.to_csv(report_filepath)


if __name__ == "__main__":
    data_folder = '../data/results/'
    predictions_filepath = data_folder + 'results-solution-3.csv'
    evaluation_filepath = data_folder + 'results-solution-3-evaluated.csv'
    report_filepath = data_folder + 'results-solution-3-report.csv'

    evaluator = AnswerEvaluator(fin_qa_evaluation_prompt)
    evaluator.run(predictions_filepath, evaluation_filepath)
    evaluator.create_report(evaluation_filepath, report_filepath)
