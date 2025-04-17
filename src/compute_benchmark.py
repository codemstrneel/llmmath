import os
import subprocess
import tempfile
import re
from pathlib import Path
import sys
import types

def extract_question_solution_tests(guess_content, sol_content):
    # Extract the question section
    question_match = re.search(r'QUESTION\n(.*?)\nSOL & TESTS', sol_content, re.DOTALL)
    question = question_match.group(1).strip() if question_match else None

    # Extract the solution code between <|Solution Begin|> and <|Solution End|>
    solution_match = re.search(r'<\|Solution Begin\|>\s*```python\n(.*?)\n```',
                               sol_content, re.DOTALL)
    solution_code = solution_match.group(1).strip() if solution_match else None

    # Extract the test code between <|Test Begin|> and <|Test End|>
    test_match = re.search(r'<\|Test Begin\|>\s*```python\n(.*?)\n```',
                           sol_content, re.DOTALL)
    test_code = test_match.group(1).strip() if test_match else None



    return {"question": question, "solution": solution_code, "tests": test_code,
            "guess": guess_content}

def create_pytest_file(problem_title, solution_code, test_cases):
    """
    Creates a temporary pytest file with both the solution and the o1-generated code.

    Parameters:
        problem_title (str): The title of the problem.
        solution_code (str): The correct solution code.
        generated_code (str): The generated code to be tested.
        test_cases (str): The pytest-compatible test cases.

    Returns:
        str: The path to the temporary pytest file.
    """
    temp_dir = tempfile.mkdtemp()
    file_path = os.path.join(temp_dir, f"test_{problem_title.replace(' ', '_')}.py")

    with open(file_path, "w") as f:
        f.write(f"# Test cases for: {problem_title}\n\n")
        f.write("# Solution code\n")
        f.write(solution_code + "\n\n")
        f.write("# Test cases\n")
        f.write(test_cases + "\n")

    return file_path

def run_pytest(file_path):
    """
    Runs the generated pytest file and captures the output.

    Parameters:
        file_path (str): Path to the pytest file.

    Returns:
        str: The output from pytest.
    """
    result = subprocess.run(["pytest", file_path, "-v"], capture_output=True, text=True)
    return result.stdout


def run_in_memory_tests(solution_code: str, test_code: str) -> bool:
    """
    1) Create a fake module named 'solution' in memory.
    2) Exec the solution code into that module's namespace.
    3) Add the module to sys.modules so an import line like
       'from solution import ...' will work.
    4) Exec the test code in a fresh namespace.
    5) Locate test_ functions in that namespace and run them.
    6) Print pass/fail results, then clean up the fake module.
    7) Return True if all tests passed, otherwise False.
    """

    # Step 1 & 2: Create in-memory 'solution' module and exec the solution code
    solution_module = types.ModuleType("solution")
    exec(solution_code, solution_module.__dict__)

    # Step 3: Register it so "from solution import ..." works
    sys.modules["solution"] = solution_module

    # Step 4: Exec the test code
    test_namespace = {}
    exec(test_code, test_namespace)

    # Step 5: Gather test_* functions
    test_functions = []
    for name, obj in test_namespace.items():
        if callable(obj) and name.startswith('test_'):
            test_functions.append(obj)

    # Step 5: Run the tests
    all_passed = True
    for test_func in test_functions:
        try:
            test_func()
            print(f"✅ {test_func.__name__} passed")
        except AssertionError:
            print(f"❌ {test_func.__name__} FAILED")
            all_passed = False

    # Step 6: Print summary, then clean up
    if all_passed:
        print("\n🎉 All tests passed!")
    else:
        print("\n❗Some tests failed.")

    del sys.modules["solution"]

    # Return whether all tests passed
    return all_passed

def run_benchmark(benchmark):
    """
    Runs the benchmark for each problem in the given list.

    Parameters:
        benchmark (list): A list of problems with title, solution, generated code, and pytest cases.
    """
    for i, problem in enumerate(benchmark):
        title = f"problem_run_{i}"
        print(f"\nTesting Problem: {title}\n")

        solution_code = problem["solution"]
        test_cases = problem["tests"]

        # Create a temporary pytest file
        pytest_file = create_pytest_file(title, solution_code, test_cases)

        # Run pytest and print the results
        result = run_pytest(pytest_file)
        print(result)

if __name__ == '__main__':
    directory = Path("./solutions")
    benchmark_data = []
    # Loop through all .txt or .py or any files in the directory
    for file_path in directory.glob("*"):  # change the pattern as needed
        with file_path.open("r", encoding="utf-8") as file:
            guess_content = file.read()
            sol_content = open(f"./problems/{file_path.name[:-4]}").read()
            benchmark_data.append(extract_question_solution_tests(guess_content,
                                                                  sol_content))

    benchmark_data = [v for v in benchmark_data if None not in v.values()]
    benchmark_data = benchmark_data[33:34]
    print(benchmark_data[0]["question"])
    print(benchmark_data[0]["solution"])
    print(benchmark_data[0]["tests"])

    total_passed = 0
    total_err = 0
    for i in range(len(benchmark_data)):
        try:
            result = run_in_memory_tests(benchmark_data[i]["solution"],
                            benchmark_data[i]["tests"])
            if result:
                total_passed += 1

        except Exception as e:
            total_err += 1



    print(f"{total_passed} passed, {total_err} erred,  out of {len(benchmark_data)}")
