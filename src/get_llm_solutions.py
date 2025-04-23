import os
from pathlib import Path
import re
from dotenv import load_dotenv
from openai import OpenAI

from parse import extract_question_solution_tests

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_openai_response(prompt, model="o1"):
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
    return f"""You are a coding assistant. Solve the following programming problem and make sure to consider edge cases. Respond with only a Python function called 'solution'. Do not include explanations, comments, or any markdown or formatting tags.
Question:
{question}
"""


if __name__ == '__main__':
    directory = Path("./problems")
    benchmark_data = []

    max_iteration_id = 0
    # Loop through all .txt or .py or any files in the directory
    for file_path in directory.glob("*"):  # change the pattern as needed
        with file_path.open("r", encoding="utf-8") as file:
            iteration_id = file_path.name[file_path.name.find('t') + 1:]
            max_iteration_id = max(max_iteration_id, int(iteration_id))
            if int(iteration_id) != 0:
                continue
            content = file.read()
            entry = extract_question_solution_tests(content)
            entry["file_name"] = file_path.name[:-len(iteration_id) - 3]
            benchmark_data.append(entry)

    print(f"Original benchmark size is {len(benchmark_data)}")
    benchmark_data = [d for d in benchmark_data if None not in d.values()]
    print(f"Parseable benchmark size is {len(benchmark_data)}")

    for i, data in enumerate(benchmark_data):
        solution = get_problem_solution(data['question'])
        file_name = data["file_name"]
        for id in range(0, max_iteration_id + 1):
            with open(f"./test_taker_sols/{file_name}_it{id}_sol", "w") as f:
                f.write(solution)

        print(f"Done solving {i + 1}th problem")

