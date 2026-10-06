import json
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openai import OpenAI
from query import search, generate_answer

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def load_questions():
    with open("evals/questions.json", "r") as f:
        return json.load(f)


def evaluate_answer(question, expected, answer):

    response = client.responses.create(
        model="gpt-5-mini",
        instructions="""
You are evaluating the quality of an AI-generated answer.

Determine whether the generated answer is factually consistent
with the expected answer.

The generated answer does NOT need to use the same words as
the expected answer.

Treat these as equivalent when appropriate:
- numbers written as words vs digits
- singular/plural variations
- minor wording differences
- additional relevant information

Mark the answer as PASS if it contains the expected fact
or an equivalent fact.

Mark it as FAIL if the answer contradicts the expected answer
or fails to provide the expected information.

Return ONLY valid JSON in this format:

{
  "result": "PASS" or "FAIL",
  "reason": "brief explanation"
}
""",
        input=f"""
Question:
{question}

Expected answer:
{expected}

Generated answer:
{answer}
"""
    )

    try:
        evaluation = json.loads(response.output_text)
        return evaluation
    except json.JSONDecodeError:
        print("Could not parse evaluator response:")
        print(response.output_text)

        return {
            "result": "FAIL",
            "reason": "Evaluator returned invalid JSON"
        }


def run_eval(test):

    question = test["question"]
    expected = test["expected"]

    print("\n" + "=" * 70)
    print(f"QUESTION: {question}")
    print(f"EXPECTED: {expected}")

    results = search(question)

    print("\nRETRIEVED CHUNKS:")
    for source, section, content, similarity in results:
        print(f"  Source:     {source}")
        print(f"  Section:    {section}")
        print(f"  Similarity: {similarity:.4f}")

    answer = generate_answer(question, results)

    evaluation = evaluate_answer(
        question,
        expected,
        answer
    )

    print(f"ANSWER:   {answer}")
    print(f"RESULT:   {evaluation['result']}")
    print(f"REASON:   {evaluation['reason']}")

    return evaluation["result"] == "PASS"


def main():

    tests = load_questions()

    passed = 0

    for test in tests:

        if run_eval(test):
            passed += 1

    total = len(tests)
    accuracy = (passed / total) * 100

    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)

    print(f"Tests:     {total}")
    print(f"Passed:    {passed}")
    print(f"Failed:    {total - passed}")
    print(f"Accuracy:  {accuracy:.1f}%")

    print("=" * 70)


if __name__ == "__main__":
    main()