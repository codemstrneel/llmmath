
import re

def extract_question_solution_tests(sol_content):
    # Extract the question section
    question_match = re.search(r'QUESTION\n(.*?)\nSOLUTION/TESTS', sol_content, re.DOTALL)
    question = question_match.group(1).strip() if question_match else None

    # Extract the solution code between <|Solution Begin|> and <|Solution End|>
    solution_match = re.search(r'<\|Solution Begin\|>\n(.*?)\n<\|Solution End\|>',
                               sol_content, re.DOTALL)
    solution_code = solution_match.group(1).strip() if solution_match else None

    # Extract the test code between <|Test Begin|> and <|Test End|>
    test_match = re.search(r'<\|Test Begin\|>\n(.*?)\n<\|Test End\|>',
                           sol_content, re.DOTALL)
    test_code = test_match.group(1).strip() if test_match else None

    return {"question": question, "solution": solution_code, "tests": test_code}

def extract_solution_tests(sol_content):
    # Extract the solution code between <|Solution Begin|> and <|Solution End|>
    solution_match = re.search(r'<\|Solution Begin\|>\n(.*?)\n<\|Solution End\|>',
                               sol_content, re.DOTALL)
    solution_code = solution_match.group(1).strip() if solution_match else None

    # Extract the test code between <|Test Begin|> and <|Test End|>
    test_match = re.search(r'<\|Test Begin\|>\n(.*?)\n<\|Test End\|>',
                           sol_content, re.DOTALL)
    test_code = test_match.group(1).strip() if test_match else None

    return {"solution": solution_code, "tests": test_code}


