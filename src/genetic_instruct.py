'''
TODO/Misc Ideas:
    - Provide the seed problems AND their solutions for generation of problems
      and solution/test generation for improved CoT
    - Incorporating rationales into problem generation
'''

import numpy as np
from typing import List, Dict
from openai import OpenAI
import os
from dotenv import load_dotenv
from datasketch import MinHash, MinHashLSH
from collections import defaultdict
import random

load_dotenv()

task_ids = {0: "cross-over", 1: "mutation"}
task_prob = [0.5, 0.5]  # 50% chance for either task
difficulty_levels = ["easy", "medium", "hard"]

# Initialize OpenAI Client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Function to get AI response
def get_openai_response(prompt, model="gpt-4-turbo", log_file="./api_log.txt"):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )
        msg = response.choices[0].message.content.strip()

        if log_file != None:
            with open(log_file, "a") as f:
                f.write("PROMPT:\n\n" + prompt + "\n\n" + "RESPONSE\n\n" + msg + "\n\n")

        return msg
    except Exception as e:
        return f"Error: {e}"


# Step 2: Solution and Test Generation
def generate_solution_and_tests(question):
    """ Generate Python code solutions and corresponding unit tests """
    return get_openai_response(solution_and_tests_prompt(question))


# Genetic Instruct - Main Function
def genetic_instruct(seed_instructions: List[Dict[str, str]], num_samples,
                     num_colonies, crossover_batch_size, num_crossover):
    """ Genetically Instruct for one generation, run sequentially """
    max_samples = num_samples // num_colonies
    current_samples = []

    for i in range(num_colonies):
        print(f"Starting colony {i}")
        results = genetic_instruct_one_colony(seed_instructions, max_samples,
                                              crossover_batch_size, num_crossover)
        current_samples.extend(results)

    # Deduplication after merging results from all colonies
    current_samples = deduplicate(current_samples)

    return current_samples  # Return the final array of samples


def genetic_instruct_one_colony(seed_instructions: List[Dict[str, str]],
                                max_samples, crossover_batch_size, num_crossover):
    all_samples = [v for v in seed_instructions]
    new_samples = []
    while len(new_samples) < max_samples:
        print(f"Curr dataset size in colony: {len(new_samples)}")
        results = genetic_instruct_one_iter(all_samples, crossover_batch_size,
                                            num_crossover)
        results = deduplicate(results)
        all_samples.extend([result[0] for result in results])
        new_samples.extend(results)
    return new_samples


# One iteration of the Genetic Instruct process
def genetic_instruct_one_iter(seed_instructions: List[Dict[str, str]],
                              crossover_batch_size, num_crossover):
    """Generate coding questions, solutions, and tests for one iteration"""
    task_id = np.random.choice(list(task_ids.keys()), p=task_prob)  # Randomly select task

    if task_id == 0:
        new_instructions = []
        for _ in range(num_crossover):
            new_instructions.append(crossover(seed_instructions,
                                              crossover_batch_size))
    elif task_id == 1:
        new_instructions = mutate(seed_instructions)

    # Step 2: Code generation for the newly generated instructions
    final_samples = []
    for instruction in new_instructions:
        solution_and_tests = generate_solution_and_tests(instruction)
        print("Finished solution and test gen")

        # Add the diversified triplet to final samples
        final_samples.append((instruction, solution_and_tests))

    return final_samples

def mutate(seed_instructions):
    instruction = sample_instruction_batch(seed_instructions, 1)[0]
    problems = []
    for i in range(len(difficulty_levels)):
        problem = get_openai_response(mutation_prompt(instruction, difficulty_levels[i]))
        problems.append(problem)
    return problems

def crossover(seed_instructions, batch_size):
    instructions = sample_instruction_batch(seed_instructions, batch_size)
    problem = get_openai_response(crossover_prompt(instructions))
    return problem

# def filter_code_generations(new_instructions, code_generations, test_cases):
#     """ Filter and validate code using unit tests and correctness checks """
#     # Validate code using test cases (could be executed in a Python interpreter in a real scenario)
#     return [(instruction, code, test) for instruction, code, test in zip(new_instructions, code_generations, test_cases) if validate_code_with_tests(code, test)]
#
#
# def validate_code_with_tests(code, test_cases):
#     """ Validate generated code against unit tests """
#     # Simulate code validation (can use a real code execution environment here)
#     return True  # For now, always returns True as a placeholder


def minhash_signature(text, num_perm=250):
    m = MinHash(num_perm=num_perm)
    for word in text.split():
        m.update(word.encode('utf8'))
    return m


def deduplicate(instructions, threshold=0.75):
    """ Deduplicate the instruction samples across nodes. Instructions is an
    array of 2-element tuples representing (question, solution_and_test)"""
    lsh = MinHashLSH(threshold=threshold, num_perm=250)
    deduplicated = []
    signature_map = defaultdict(list)

    # Generate signatures and add to LSH
    for idx, pair in enumerate(instructions):
        m = minhash_signature(pair[0])
        if lsh.query(m):
            # Check for similarity with existing entries
            similar = lsh.query(m)
            signature_map[similar[0]].append(pair)
        else:
            lsh.insert(str(idx), m)
            signature_map[str(idx)].append(pair)

    # Collect deduplicated instructions
    for group in signature_map.values():
        deduplicated.append(group[0])  # Keep only one from each group

    print(f"Deduplication went from {len(instructions)} to {len(deduplicated)} problems")
    return deduplicated

def sample_instruction_batch(instructions, batch_size):
    '''Randomly samples batch_size elements from instructions'''
    if batch_size >= len(instructions):
        return instructions
    return random.sample(instructions, batch_size)


'''Functions for generating prompts'''


def mutation_prompt(instruction, difficulty_level):
    if difficulty_level == 'easy':
        return f'''Please decrease the difficulty of the given programming test
question a bit. The new problem should be conceptually similar to the given
question, but should not simply paraphrase it. Do not provide any hints, solutions or outputs. Only one new instruction is allowed.
Original Question: {instruction}
New Question:
'''
    if difficulty_level == 'medium':
        return f'''Please create a new programming problem of the same
difficulty as the given programming test question. The new problem should be
conceptually similar to the given question, but should not simply paraphrase
it. Do not provide any hints, solutions or outputs. Only one new instruction is allowed.
Original Question: {instruction}
New Question:
'''
    if difficulty_level == 'hard':
        return f'''Please increase the difficulty of the given programming test question a bit. Do not provide any hints, solutions or outputs. Only one new instruction is allowed.
Original Question: {instruction}
New Question:
'''


def crossover_prompt(instructions):
    prompt_str = '''I will provide you with a set of coding questions. Please give
me a new coding question that combines core concepts from two or more of the
given questions. Please ensure that the new question is novel and does not
simply paraphrase any of the problems I am giving you.\n'''
    question_str = ""
    for (i, question) in enumerate(instructions):
        question_str += f'Question {i + 1}:\n{question}\n'
    return prompt_str + question_str + "New Question:\n"


def solution_and_tests_prompt(problem):
    return f'''You are an expert in Python coding.
## Task:
Please Answer the question and generate unit tests to verify your answer.
## Output Format:
Your solution and unit tests should be presented in markdown Python code format within the
specified sections below. Ensure your code is within code blocks. For the tests, use
pytest style by defining individual test functions (without classes) and using assert
statements. Your tests should be implementation independent.
<|Solution Begin|>
[Solution Code in Python]
<|Solution End|>
<|Test Begin|>
[Unit Test Code in Python]
<|Test End|>
## Example
Below is an example output format implementing a simple a + b function.
<|Solution Begin|> ```python
def add(a, b):
    """
    Returns the sum of a and b.
    """
    return a + b
```
<|Solution End|>
<|Test Begin|>
```python
from solution import add
def test_add_positive_numbers():
    assert add(2, 3) == 5
def test_add_with_zero():
    assert add(0, 5) == 5
    assert add(5, 0) == 5
def test_add_negative_numbers():
    assert add(-1, -1) == -2
def test_add_mixed_sign_numbers():
    assert add(-1, 3) == 2
```
<|Test End|>
## Question: {problem}
'''
