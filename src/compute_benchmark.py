import os
import subprocess
import tempfile

def create_pytest_file(problem_title, solution_code, generated_code, test_cases):
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
        f.write("# o1 Generated code\n")
        f.write(generated_code + "\n\n")
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

def run_benchmark(benchmark):
    """
    Runs the benchmark for each problem in the given list.

    Parameters:
        benchmark (list): A list of problems with title, solution, generated code, and pytest cases.
    """
    for problem in benchmark:
        title = problem.get("title", "Untitled Problem")
        print(f"\nTesting Problem: {title}\n")

        solution_code = problem["solution"]
        generated_code = problem["o1_generated_code"]
        test_cases = problem["pytest_cases"]

        # Create a temporary pytest file
        pytest_file = create_pytest_file(title, solution_code, generated_code, test_cases)

        # Run pytest and print the results
        result = run_pytest(pytest_file)
        print(result)

if __name__ == '__main__':
    # Example benchmark data with pytest format test cases
    benchmark_data = [
        {
           "title": "Double a Number",
           "solution":
"""def solve(x):
    return x * 2""",
           "o1_generated_code":
"""def solve(x):
    return 2 * x""",
           "pytest_cases":
"""
def test_double_number():
    assert solve(3) == 6
    assert solve(5) == 10
    assert solve(0) == 0
    assert solve(-2) == -4
"""
        },
        # Add more problems as needed
    ]

    run_benchmark(benchmark_data)
