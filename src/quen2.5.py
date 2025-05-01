import os, random, torch
from typing import List, Dict
from dotenv import load_dotenv
from openai import OpenAI

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from trl import AutoModelForCausalLMWithValueHead, PPOTrainer, PPOConfig

from compute_benchmark import run_in_memory_tests
from parse import extract_solution_tests

# ── 0) Environment & Globals ─────────────────────────────────────────────────
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
SAVE_DIR = "problems"
NUM_EPISODES = 400
MAX_TEST_ITERS    = 3
MAX_CODE_ITERS    = 3
os.makedirs(SAVE_DIR, exist_ok=True)

# ── 1) Helpers & Prompts ──────────────────────────────────────────────────────
difficulty_levels = ["easy","medium","hard"]

def get_openai_response(prompt: str, model: str = "gpt-4o") -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role":"user","content":prompt}]
    )
    return resp.choices[0].message.content.strip()

def mutation_prompt(instruction: str, level: str) -> str:
    methods = {
        "easy":   "decrease the difficulty slightly, no hints or solutions.",
        "medium": "keep the same difficulty, no hints or solutions.",
        "hard":   "increase the difficulty slightly, no hints or solutions."
    }
    return (
        f"Please {methods[level]} Original Question: {instruction}\n"
        "New Question:"
    )

def tests_prompt(problem: str) -> str:
    return (
        "You are a Python testing expert.\n"
        "Generate pytest‐style tests (no solution) for the following problem:\n\n"
        f"{problem}\n\n"
        "Only output the tests, and."
    )

def tests_prompt_with_feedback(problem: str, history: List[str]) -> str:
    hist = ""
    for i, t in enumerate(history):
        hist += f"Attempt {i+1} Tests:\n{t}\n\n"
    return (
        "You are a Python testing expert.\n"
        "Improve the pytest tests below based on the problem.\n\n"
        f"Problem:\n{problem}\n\n"
        f"{hist}"
        "Generate the improved tests only."
    )

def solution_and_tests_prompt(problem: str) -> str:
    return (
        "You are a Python coding expert.\n"
        "Generate a solution and pytest‐style tests in the following format.\n"
        "<|Solution Begin|>\n# solution code\n<|Solution End|>\n"
        "<|Test Begin|>\n# pytest tests\n<|Test End|>\n\n"
        f"## Question: {problem}\n"
    )

def solution_and_tests_prompt_with_feedback(
    problem: str,
    sol_history: List[str],
    output_history: List[str]
) -> str:
    hist = ""
    for i, (sol, out) in enumerate(zip(sol_history, output_history)):
        hist += (
            f"Attempt {i+1} Solution:\n{sol}\n\n"
            f"Attempt {i+1} Test Output:\n{out}\n\n"
        )
    return (
        "You are an expert in Python coding. Fix failures below, if any.\n\n"
        f"Question: {problem}\n\nChat History:\n{hist}\n\n"
        "Generate a solution and pytest‐style tests using the exact <> tags:\n"
        "<|Solution Begin|>\n# solution code\n<|Solution End|>\n"
        "<|Test Begin|>\n# pytest tests\n<|Test End|>\n"
    )

# ── 2) Scoring ────────────────────────────────────────────────────────────────
_SC_NOTE = "Reply ONLY `1` (good) or `0` (bad)."

def _score(prompt: str) -> int:
    txt = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role":"user","content":prompt}]
    ).choices[0].message.content.strip()
    return 1 if txt.startswith("1") else 0

def judge_problem(p: str, p_old: str) -> int:
    prompt = (
        "Is the programming problem below clear, sufficiently different "
        f"from the original question ({p_old}), solvable, and algorithmic in nature? {_SC_NOTE}\n\n"
        f"{p}"
    )
    return _score(prompt)

def judge_tests(p: str, t: str) -> int:
    prompt = (
        "Do these pytest tests give good coverage and avoid trivial asserts? "
        f"{_SC_NOTE}\n\nProblem:\n{p}\n\nTests:\n{t or ''}"
    )
    return _score(prompt)

# ── 3) Models & PPO Setup ─────────────────────────────────────────────────────
POLICY_MODEL = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(POLICY_MODEL, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

policy_model = AutoModelForCausalLMWithValueHead.from_pretrained(
    POLICY_MODEL, trust_remote_code=True
).to(DEVICE)
ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
    POLICY_MODEL, trust_remote_code=True
).to(DEVICE)

ppo_cfg = PPOConfig(batch_size=1, mini_batch_size=1, learning_rate=1e-5)
ppo = PPOTrainer(
    config=ppo_cfg,
    model=policy_model,
    ref_model=ref_model,
    tokenizer=tokenizer
)

GENCFG = GenerationConfig(
    max_new_tokens=256, do_sample=True, temperature=0.9, top_p=0.95
)

def qwen_generate(prompt: str) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    outs = policy_model.generate(**inputs, generation_config=GENCFG)
    return tokenizer.decode(outs[0, inputs.input_ids.shape[-1]:], skip_special_tokens=True)

# ── 4) RL Episode (NEW PIPELINE) ──────────────────────────────────────────────
def rl_episode(seed: str, episode_idx: int) -> Dict[str, int]:
    # 1) Mutate seed → hard problem
    problem = qwen_generate(mutation_prompt(seed, "hard"))

    # 2) Single TEST generation
    final_tests = qwen_generate(tests_prompt(problem))
    r_tests = judge_tests(problem, final_tests)

    # 3) Iterative SOLUTION+TEST GEN (up to 3 iters)
    sol_hist, out_hist = [], []
    passed = False
    for i in range(MAX_CODE_ITERS):
        if i == 0:
            candidate = qwen_generate(solution_and_tests_prompt(problem))
        else:
            candidate = qwen_generate(solution_and_tests_prompt_with_feedback(
                problem, sol_hist, out_hist
            ))
        sol_hist.append(candidate)

        parsed = extract_solution_tests(candidate)
        sol   = parsed.get("solution", "")
        tests = parsed.get("tests", final_tests)

        try:
            passed, outs = run_in_memory_tests(sol, tests)
        except Exception as e:
            passed, outs = False, [f"runtime error: {e}"]

        out_hist.append("\n".join(outs))
        if passed:
            break

    final_solution_tests = sol_hist[-1]
    r_sol = 1 if passed else 0

    # 4) Judge problem clarity/novelty vs original seed
    r_prob = judge_problem(problem, seed)

    # 5) PPO update
    q_ids = tokenizer(problem,               return_tensors="pt").input_ids.squeeze(0).to(DEVICE)
    r_ids = tokenizer(final_solution_tests, return_tensors="pt").input_ids.squeeze(0).to(DEVICE)
    reward = torch.tensor(r_prob + r_tests + r_sol, dtype=torch.float, device=DEVICE)
    ppo.step([q_ids], [r_ids], [reward])

    # Save if all passed
    if r_prob and r_tests and r_sol:
        with open(os.path.join(SAVE_DIR, f"prob_{episode_idx}.txt"), "w") as f:
            f.write(f"QUESTION\n\n{problem}\n\nSOLUTION & TESTS\n\n{final_solution_tests}")

    return {"problem": r_prob, "tests": r_tests, "solution": r_sol}


# ── 5) Main Loop ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    seeds = [ex["text"] for ex in load_dataset("mbpp", split="train")]
    saved = 0
    for ep in range(NUM_EPISODES):
        res = rl_episode(random.choice(seeds), ep)
        if all(res.values()):
            saved += 1
            print(f"[{saved}] saved")
    print("FINISHED")
