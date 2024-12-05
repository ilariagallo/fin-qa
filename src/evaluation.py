import pandas as pd
from langchain import hub
from langchain_openai import ChatOpenAI

from src.prompts import fin_qa_evaluation_prompt

# Grade prompt
grade_prompt_answer_accuracy = prompt = hub.pull("langchain-ai/rag-answer-vs-reference")
grade_prompt_answer_accuracy.messages[0].prompt.template = fin_qa_evaluation_prompt
llm = ChatOpenAI(model="gpt-4o-mini")


class AnswerEvaluator:

    def __init__(self, custom_prompt=None):
        self.llm = ChatOpenAI(model="gpt-4o-mini")
        self.grader_prompt = hub.pull("langchain-ai/rag-answer-vs-reference")
        self.grader_prompt.messages[0].prompt.template = custom_prompt
        self.answer_grader = grade_prompt_answer_accuracy | llm

    def evaluate_answer(self, qa: pd.Series) -> list:
        """
        A simple evaluator for RAG answer accuracy

        :param qa: pd.Series including question, actual answer and expected answer for a single question
        :returns score and explanation

        """

        # Get question, ground truth answer, RAG chain answer
        input_question = qa['question']
        reference = qa['actual_answer']
        prediction = qa['expected_answer']

        # Run evaluator
        evaluation = self.answer_grader.invoke({"question": input_question,
                                                "correct_answer": reference,
                                                "student_answer": prediction})
        score = evaluation["Score"]
        explanation = evaluation['Explanation']

        return [score, explanation]

    def run(self, input_file, output_file):
        predictions = pd.read_csv(input_file)
        scores = predictions.apply(lambda row: self.evaluate_answer(row), axis=1)
        scores_df = pd.DataFrame(data=list(scores), columns=['score', 'explanation'])

        final_df = pd.concat([predictions, scores_df], axis=1)
        final_df.to_csv(output_file, index=False)


if __name__ == "__main__":
    evaluator = AnswerEvaluator(fin_qa_evaluation_prompt)
    evaluator.run("../data/results-solution-1.csv", '../data/results-solution-1-evaluated.csv')
