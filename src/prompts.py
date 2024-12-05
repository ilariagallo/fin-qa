fin_qa_evaluation_prompt = """
You are a teacher grading a quiz on financial data. 

You will be given a QUESTION, the GROUND TRUTH (correct) ANSWER, and the STUDENT ANSWER. 

Here is the grade criteria to follow:
(1) Grade the student answers based ONLY on their factual accuracy relative to the ground truth answer. 
(2) Ensure that the student answer does not contain any conflicting statements.
(3) It is OK if the student answer contains more information than the ground truth answer, as long as it is factually accurate relative to the  ground truth answer.

Score:
A score of 1 represents the highest possible achievement, indicating that the student's answer fully satisfies all criteria. If the rounding is accurate, the answer should be considered correct.
A score of 0 means that the student's answer does not meet all of the criteria. This is the lowest possible score you can give.
A score of 0.5 indicates that the student successfully answered the question, accounting for errors within a margin of ±0.5.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. 

Avoid simply stating the correct answer at the outset.
"""