import numpy as np
from typing import List, Dict
from openai import OpenAI

# Constants for the process (can be tuned)
#print("c")
NUM_SAMPLES = 20
NUM_COLONIES = 4
NUM_SUBCOLONY = 2
task_ids = {0: "cross-over", 1: "mutation"}
task_prob = [0.5, 0.5]  # 50% chance for either task


#print("b")
# Initialize OpenAI Client
client = OpenAI(api_key="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")

# Function to get AI response
def get_openai_response(prompt, model="o1-mini"):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {e}"

# Step 1: Coding Question Synthesis
def generate_coding_question(seed_instructions):
    """ Generate coding questions using existing seed instructions """
    levels = ["easy", "medium", "hard"]
    
    instructions = []
    for i in range(3):
        prompt = f"Generate a diverse and challenging coding question based on the following examples: {seed_instructions}. Include only one {levels[i]} question and no others."
        instructions.append(get_openai_response(prompt))
    return instructions
# Step 2: Solution and Test Generation
def generate_solution_and_tests(question):
    """ Generate Python code solutions and corresponding unit tests """
    solution_prompt = f"Write a Python solution for this question: {question}"
    test_prompt = f"Generate unit tests for the following Python solution: {question}"
    
    solution = get_openai_response(solution_prompt)
    test_cases = get_openai_response(test_prompt)
    
    return solution, test_cases

# Step 3: Post-training Data Synthesis
def post_training_data_synthesis(question, solution, test_cases):
    """ Reformat and diversify question-solution-test triplets """
    style_conversion_prompt = f"Convert this question, solution, and test cases into a different format: {question}, {solution}, {test_cases}"
    diversified_triplet = get_openai_response(style_conversion_prompt)
    
    return diversified_triplet

# Genetic Instruct - Main Function
def genetic_instruct(seed_instructions: List[Dict[str, str]]):
    """ Genetically Instruct for one generation, run sequentially """
    max_samples = NUM_SAMPLES // NUM_COLONIES
    current_samples = []
    all_samples = [v for v in seed_instructions]

    # Sequentially generate samples instead of parallel processing
    while len(current_samples) <= max_samples:
        #print("a")
        # Perform the genetic instruct process for each sub-colony sequentially
        for _ in range(NUM_SUBCOLONY):
            #print(_)
            results = genetic_instruct_one_iter(all_samples)  # Call the generation function
            current_samples.extend(results)
            all_samples.extend(results)

        # Deduplication after merging results from all colonies
        current_samples = global_deduplicate_across_nodes(current_samples)

    return current_samples  # Return the final array of samples

# One iteration of the Genetic Instruct process
def genetic_instruct_one_iter(seed_instructions: List[Dict[str, str]]):
    #print("hi")
    """Generate coding questions, solutions, and tests for one iteration"""
    task_id = np.random.choice(list(task_ids.keys()), p=task_prob)  # Randomly select task

    # Step 1: Instruction generation based on task type (mutation/crossover)
    new_instructions = generate_coding_question(seed_instructions)
    #print(new_instructions)
    # Step 2: Code generation for the newly generated instructions
    final_samples = []
    for instruction in new_instructions:
        #print(instruction)
        solution, test_cases = generate_solution_and_tests(instruction)
        
        # Step 3: Post-training Data Synthesis (diversify)
        diversified_triplet = post_training_data_synthesis(instruction, solution, test_cases)
        
        # Add the diversified triplet to final samples
        final_samples.append(diversified_triplet)
    
    return final_samples

# Helper functions for generating prompts

def encode_inst_generation_prompts(seed_instructions, task_id):
    """ Generate prompts to create new instructions from seed data """
    if task_id == 0:  # Crossover
        return f"Generate new instructions by combining the following examples: {seed_instructions}"
    elif task_id == 1:  # Mutation
        return f"Generate a new, mutated version of the following question: {seed_instructions}"
    else:
        return f"Generate new coding instructions based on: {seed_instructions}"

def encode_codegen_prompts(new_instructions, task_id):
    """ Generate prompts for generating code solutions based on instructions """
    return f"Generate Python code for the following instruction: {new_instructions}"

def encode_test_generation_prompts(code_generations):
    """ Generate unit test prompts for generated code """
    return f"Generate unit tests for the following Python code: {code_generations}"

def filter_code_generations(new_instructions, code_generations, test_cases):
    """ Filter and validate code using unit tests and correctness checks """
    # Validate code using test cases (could be executed in a Python interpreter in a real scenario)
    return [(instruction, code, test) for instruction, code, test in zip(new_instructions, code_generations, test_cases) if validate_code_with_tests(code, test)]

def validate_code_with_tests(code, test_cases):
    """ Validate generated code against unit tests """
    # Simulate code validation (can use a real code execution environment here)
    return True  # For now, always returns True as a placeholder

def encode_judge_prompts(new_inst_code_test):
    """ Generate prompts for the Judge-LLM to assess the correctness of the instruction, code, and test triplets """
    return f"Judge the correctness of this instruction, code, and test: {new_inst_code_test}"

def global_deduplicate_across_nodes(current_samples):
    """ Deduplicate the instruction samples across nodes """
    # Implement MinHash or similar method for deduplication
    return current_samples

# Testing the genetic instruct pipeline
seed_instructions = [
    {"instruction": "Write a function to find the factorial of a number.", "code": "def factorial(n): return 1 if n == 0 else n * factorial(n-1)"},
    {"instruction": "Write a function to calculate Fibonacci sequence.", "code": "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)"}
]

# Run the genetic instruct process and print the results
final_samples = genetic_instruct(seed_instructions)
print(final_samples)  # This will print the final array of samples (instruction, code, test triplets)

print("hi")