import os
import subprocess
import tempfile
import re
from pathlib import Path
import sys
import types
import signal
from _pytest.outcomes import Failed
import functools

from parse import extract_question_solution_tests

TIMEOUT_PERIOD = 2 # Timeout a test after set amount of time

def suppress_print(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        try:
            result = func(*args, **kwargs)
        finally:
            sys.stdout = original_stdout
        return result
    return wrapper


def timeout_handler(signum, frame):
    raise TimeoutError("Test execution exceeded time limit.")


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
    test_code = "from solution import solution\n" + test_code
    exec(test_code, test_namespace)

    signal.signal(signal.SIGALRM, timeout_handler)
    # Step 5: Gather test_* functions
    test_functions = []
    for name, obj in test_namespace.items():
        if callable(obj) and name.startswith('test_'):
            test_functions.append(obj)

    outputs = []

    # Step 5: Run the tests
    all_passed = True
    for test_func in test_functions:
        signal.alarm(TIMEOUT_PERIOD)
        try:
            test_func()
            print(f"✅ {test_func.__name__} passed")
            outputs.append(f"{test_func.__name__} passed")
        except Failed as e:
            print(f"❌ {test_func.__name__} FAILED")
            all_passed = False
            outputs.append(f"{test_func.__name__} failed: {e}")
        except AssertionError as e:
            print(f"❌ {test_func.__name__} FAILED")
            all_passed = False
            outputs.append(f"{test_func.__name__} failed: {e}")
        except Exception as e:
            print(f"❌ {test_func.__name__} FAILED")
            all_passed = False
            outputs.append(f"{test_func.__name__} failed: {e}")
        finally:
            signal.alarm(0)


    # Step 6: Print summary, then clean up
    if all_passed:
        print("\n🎉 All tests passed!")
    else:
        print("\n❗Some tests failed.")

    del sys.modules["solution"]

    # Return whether all tests passed
    return all_passed, outputs


@suppress_print
def compute_statistics(benchmark_data, sol_key):
    total_passed = 0
    total_err = 0
    total_fail = 0
    for i in range(len(benchmark_data)):
        try:
            result, outputs = run_in_memory_tests(benchmark_data[i][sol_key], benchmark_data[i]["tests"])

            if result:
                total_passed += 1

            print(outputs)
        except Exception as e:
            total_err += 1

    total_fail = len(benchmark_data) - total_passed - total_err
    return (total_passed, total_err, total_fail)

if __name__ == '__main__':
    directory = Path("./test_taker_sols")
    benchmark_data = {}
    # Loop through all .txt or .py or any files in the directory
    for file_path in directory.glob("*"):  # change the pattern as needed
        with file_path.open("r", encoding="utf-8") as file:
            guess_content = file.read()
            iteration_id = file_path.name[:-4][file_path.name[:-4].find('t') + 1:]
            sol_content = open(f"./problems/{file_path.name[:-4]}").read()
            data = extract_question_solution_tests(sol_content)
            data["guess"] = guess_content

            if iteration_id not in benchmark_data:
                benchmark_data[iteration_id] = []
            if None not in data.values():
                benchmark_data[iteration_id].append(data)

    for id in sorted(benchmark_data.keys()):
        print(f"Stats for iteration {id}")
        true_stats = compute_statistics(benchmark_data[id], "solution")
        guess_stats = compute_statistics(benchmark_data[id], "guess")
        print(f"True: {true_stats[0]} passed, {true_stats[1]} erred, {true_stats[2]} failed out of {len(benchmark_data[id])}")
        print(f"Guess: {guess_stats[0]} passed, {guess_stats[1]} erred, {guess_stats[2]} failed out of {len(benchmark_data[id])}")
