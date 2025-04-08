from openai import OpenAI
import os
from dotenv import load_dotenv
import re

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def parse_concepts(concepts):
    try:
        # Find the first and last brackets
        start_idx = concepts.find('[')
        end_idx = concepts.rfind(']')

        if start_idx == -1 or end_idx == -1:
            print("Error: Brackets not found in the response.")
            return None

        # Extract the substring within the brackets
        concept_str = concepts[start_idx + 1:end_idx].strip()

        # Use regex to find all quoted strings, considering both single and double quotes
        parsed_concepts = re.findall(r'["\'](.*?)["\']', concept_str)

        assert parsed_concepts != None, f"No parsable concepts, raw concepts {concepts}"
        return parsed_concepts
    except Exception as e:
        print("Error parsing concepts:", e)
        print("Raw response:", concepts)
        return None

def generate_concepts(problem, solution, num_concepts, model):
    '''Extracts foundational concepts from a problem'''
    # Difficulty level
    level = "Olympiad-level"
    concept_prompt = f""" As an expert in educational assessment, analyze this problem with given solution.
Question: {problem}
Solution: {solution}
Break down and identify {num_concepts} foundational concepts being tested. List these knowledge
points that:
• Are core curriculum concepts typically taught in standard courses,
• Are precise and measurable (not vague like 'understanding math'),
• Are essential building blocks needed to solve this problem,
• Represent fundamental principles rather than problem-specific techniques.
Think through your analysis step by step, then format your response as a Python code snippet
containing a list of {num_concepts} strings, where each string clearly describes one fundamental
knowledge point. Ensure that these concept descriptions are concise, short, and general enough that they apply to a wide range of math problems.
"""
    response = client.chat.completions.create(model=model,
    messages=[{"role": "user", "content": concept_prompt}])

    concepts = response.choices[0].message.content
    parsed_concepts = parse_concepts(concepts)
    return parsed_concepts

def generate_rationales(problem, concepts, model):
    '''Generate rationale based on extracted concepts'''
    level = "Olympiad-level"
    rationale_prompt = f""" Imagine you are an expert in educational problem design.
You are shown these components:
Problem: {problem}
Fundamental Concepts: {concepts}
Difficulty Level: {level}

Your task is to reverse-engineer a clear thinking process that shows how a teacher might design
this problem. This thinking process should:
• Show how combining the given foundational concepts naturally leads to a problem at the specified difficulty level.
• Include all key decisions and reasoning that shaped the problem design.
• Be so precise and detailed that another teacher following these exact steps would recreate the identical problem.
• Be so natural and logical that another teacher could derive the same thinking process using only the foundational concepts and difficulty level.

Present your answer after “Thinking Process:” with the complete step-by-step thinking process described above.
"""

    response_rationale = client.chat.completions.create(model=model,
    messages=[{"role": "user", "content": rationale_prompt}])

    rationale = response_rationale.choices[0].message.content
    return rationale

if __name__ == "__main__":
    problem = "Find the least odd prime factor of 2019^8 + 1."
    num_concepts = 5

    concepts = generate_concepts(problem, "", num_concepts, "gpt-4-turbo")
    # rationales = generate_rationales(problem, concepts, "gpt-4-turbo")

    print("GENERATED CONCEPTS")
    print(concepts)
    # print("GENERATED RATIONALES")
    # print(rationales)

