import pandas as pd
from langchain import hub
from langchain_openai import ChatOpenAI

from src.prompts import fin_qa_evaluation_prompt

# Grade prompt
grade_prompt_answer_accuracy = prompt = hub.pull("langchain-ai/rag-answer-vs-reference")
grade_prompt_answer_accuracy.messages[0].prompt.template = fin_qa_evaluation_prompt
llm = ChatOpenAI(model="gpt-4o-mini")


def answer_evaluator(result: pd.Series) -> list:
    """
    A simple evaluator for RAG answer accuracy
    """

    # Get question, ground truth answer, RAG chain answer
    input_question = result['question']
    reference = result['actual_answer']
    prediction = result['expected_answer']

    # Structured prompt
    answer_grader = grade_prompt_answer_accuracy | llm

    # Run evaluator
    evaluation = answer_grader.invoke({"question": input_question,
                                       "correct_answer": reference,
                                       "student_answer": prediction})
    score = evaluation["Score"]
    explanation = evaluation['Explanation']

    return [score, explanation]


if __name__ == "__main__":
    results = pd.read_csv("../data/results-solution-1.csv")
    scores = results.apply(lambda row: answer_evaluator(row), axis=1)
    scores_df = pd.DataFrame(data=list(scores), columns=['score', 'explanation'])

    final_df = pd.concat([results, scores_df], axis=1)
    final_df.to_csv('../data/results-solution-1-evaluated.csv', index=False)
