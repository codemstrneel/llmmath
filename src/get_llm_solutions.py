import os
from pathlib import Path
import re
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def extract_question_solution_tests(data):
    # Extract the question section
    question_match = re.search(r'QUESTION\n(.*?)\nSOL & TESTS', data, re.DOTALL)
    question = question_match.group(1).strip() if question_match else None

    # Extract the solution code between <|Solution Begin|> and <|Solution End|>
    solution_match = re.search(r'<\|Solution Begin\|>\s*```python\n(.*?)\n```', data, re.DOTALL)
    solution_code = solution_match.group(1).strip() if solution_match else None

    # Extract the test code between <|Test Begin|> and <|Test End|>
    test_match = re.search(r'<\|Test Begin\|>\s*```python\n(.*?)\n```', data, re.DOTALL)
    test_code = test_match.group(1).strip() if test_match else None

    return {"question": question, "solution": solution_code, "tests": test_code}


def get_openai_response(prompt, model="gpt-4o-mini"):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )
        msg = response.choices[0].message.content.strip()

        return msg
    except Exception as e:
        return f"Error: {e}"


def get_problem_solution(question):
    prompt = solve_problem_prompt(question)
    return get_openai_response(prompt)


def solve_problem_prompt(question):
    return f"""You are a coding assistant. Solve the following programming problem.
Respond with only a Python function called 'solution'. Do not include explanations or comments.

Question:
{question}
"""


if __name__ == '__main__':
    directory = Path("./problems")
    benchmark_data = []
    # Loop through all .txt or .py or any files in the directory
    for file_path in directory.glob("*"):  # change the pattern as needed
        with file_path.open("r", encoding="utf-8") as file:
            content = file.read()
            entry = extract_question_solution_tests(content)
            entry["file_name"] = file_path.name
            benchmark_data.append(entry)

    benchmark_data = [d for d in benchmark_data if None not in d.values()]

    for i, data in enumerate(benchmark_data):
        solution = get_problem_solution(data['question'])
        file_name = data["file_name"]
        with open(f"./solutions/{file_name}_sol", "w") as f:
            f.write(solution)

        print(f"Done solving {i + 1}th problem")

