import numpy as np
from typing import List, Dict
from openai import OpenAI
import os
from dotenv import load_dotenv
from datasketch import MinHash, MinHashLSH
from collections import defaultdict
import random

from compute_benchmark import run_in_memory_tests
from parse import extract_solution_tests

load_dotenv()

task_ids = {0: "cross-over", 1: "mutation"}
task_prob = [0.5, 0.5]  # 50% chance for either task
difficulty_levels = ["easy", "medium", "hard"]

# Initialize OpenAI Client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Function to get AI response
def get_openai_response(prompt, model="gpt-4o", log_file="./api_log.txt"):
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


def generate_solution_and_tests_with_feedback(question, num_iterations):
    """Generate Python code solutions and unit tests with feedback; when
    num_iterations is zero, this just normally generates solutions and tests"""
    sol_history = []
    output_history = []
    sol_history.append(generate_solution_and_tests(question))

    for _ in range(num_iterations):
        sol_test = extract_solution_tests(sol_history[-1])
        result = False
        try:
            result, outputs = run_in_memory_tests(sol_test["solution"], sol_test["tests"])
            output_history.append("\n".join(outputs))
        except Exception as e:
            output_history.append(f"Compilation error: {e}")

        if result:
            return sol_history

        prompt = solution_and_tests_prompt_with_feedback(question, sol_history, output_history)
        sol_history.append(get_openai_response(prompt))
    return sol_history


# Genetic Instruct - Main Function
def genetic_instruct(seed_instructions: List[Dict[str, str]], num_samples,
                     num_colonies, crossover_batch_size, num_crossover,
                     seed_batch_size, num_code_iterations):
    """ Genetically Instruct for one generation, run sequentially """
    max_samples = num_samples // num_colonies
    current_samples = []

    for i in range(num_colonies):
        print(f"Starting colony {i}")
        colony_seed_instr = sample_instruction_batch(seed_instructions, seed_batch_size)
        results = genetic_instruct_one_colony(colony_seed_instr, max_samples,
                                              crossover_batch_size,
                                              num_crossover, num_code_iterations)
        current_samples.extend(results)

    # Deduplication after merging results from all colonies
    current_samples = deduplicate(current_samples)

    return current_samples  # Return the final array of samples


def genetic_instruct_one_colony(seed_instructions: List[Dict[str, str]],
                                max_samples, crossover_batch_size,
                                num_crossover, num_code_iterations):
    all_samples = [v for v in seed_instructions]
    new_samples = []
    while len(new_samples) < max_samples:
        print(f"Curr dataset size in colony: {len(new_samples)}")
        results = genetic_instruct_one_iter(all_samples, crossover_batch_size,
                                            num_crossover, num_code_iterations)
        new_samples.extend(results)
        new_samples = deduplicate(new_samples)
        all_samples = seed_instructions + [sample[0] for sample in new_samples]
    return new_samples


# One iteration of the Genetic Instruct process
def genetic_instruct_one_iter(seed_instructions: List[Dict[str, str]],
                              crossover_batch_size, num_crossover,
                              num_code_iterations):
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
        solution_and_tests = generate_solution_and_tests_with_feedback(instruction, num_code_iterations)

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
        return f'''Please decrease the difficulty of the given programming test question a bit. The new problem should be conceptually similar to the given question, but should not simply paraphrase it. Do not provide any hints, solutions or outputs. Only one new instruction is allowed.
Original Question: {instruction}
New Question:
'''
    if difficulty_level == 'medium':
        return f'''Please create a new programming problem of the same difficulty as the given programming test question. The new problem should be conceptually similar to the given question, but should not simply paraphrase it. Do not provide any hints, solutions or outputs. Only one new instruction is allowed.
Original Question: {instruction}
New Question:
'''
    if difficulty_level == 'hard':
        return f'''Please increase the difficulty of the given programming test question a bit. Do not provide any hints, solutions or outputs. Only one new instruction is allowed.
Original Question: {instruction}
New Question:
'''


def crossover_prompt(instructions):
    prompt_str = '''I will provide you with a set of coding questions. Please give me a new coding question that combines core concepts from two or more of the given questions. Please ensure that the new question is novel and does not simply paraphrase any of the problems I am giving you. Do not include any extra information that would help a test-taker other than the problem statement itself.\n'''
    question_str = ""
    for (i, question) in enumerate(instructions):
        question_str += f'Question {i + 1}:\n{question}\n'
    return prompt_str + question_str + "New Question:\n"


def solution_and_tests_prompt(problem):
    return f'''You are an expert in Python coding.
## Task:
Please Answer the question and generate unit tests to verify your answer.
## Output Format:
Your solution and unit tests should be presented in the format within the specified sections below. Ensure your code is within code blocks. For the tests, use pytest style by defining individual test functions (without classes) and using assert statements. Your tests should be implementation independent. Ensure that you include the <|Solution Begin|>, <|Solution End|>, <|Test Begin|>, <|Test End|> tags as depicted. The solution function must be named 'solution'.
<|Solution Begin|>
[Solution Code in Python]
<|Solution End|>
<|Test Begin|>
[Unit Test Code in Python]
<|Test End|>
## Example
Below is an example output format implementing a simple a + b function.
<|Solution Begin|>
def add(a, b):
    """
    Returns the sum of a and b.
    """
    return a + b
<|Solution End|>
<|Test Begin|>
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
<|Test End|>
## Question: {problem}
'''

def solution_and_tests_prompt_with_feedback(problem, sol_history, output_history):
    history_str = ""
    for i in range(len(sol_history)):
        history_str += f"Attempt {i+1} Solution:\n{sol_history[i]}\n\n"
        history_str += f"Attempt {i+1} Code Execution Output:\n{output_history[i]}\n\n"
    return f'''You are an expert in Python coding.
## Task:
Please Answer the question and generate unit tests to verify your answer. The entire chat history of your previous attempts to generate questions and unit tests is presented below in the "Chat History" section, along with the output of running your solution against your tests in a code execution environment. Please modify only your tests and/or solution to be more correct.

## Output Format:
Your solution and unit tests should be presented in the format within the specified sections below. Ensure your code is within code blocks. For the tests, use pytest style by defining individual test functions (without classes) and using assert statements. Your tests should be implementation independent. Ensure that you include the <|Solution Begin|>, <|Solution End|>, <|Test Begin|>, <|Test End|> tags as depicted. The solution function must be named 'solution'.
<|Solution Begin|>
[Solution Code in Python]
<|Solution End|>
<|Test Begin|>
[Unit Test Code in Python]
<|Test End|>
## Example
Below is an example output format implementing a simple a + b function.
<|Solution Begin|>
def add(a, b):
    """
    Returns the sum of a and b.
    """
    return a + b
<|Solution End|>
<|Test Begin|>
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
<|Test End|>

## Question: {problem}

## Chat History
{history_str}
'''

