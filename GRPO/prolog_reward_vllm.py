import os
import re
import subprocess
import tempfile
import json
from typing import List

from math_verify import parse, verify
from math_verify_prolog import (
    prolog_term_to_latex,
)

SYSTEM_PROMPT = """Below is an instruction that describes a task. Write a response that appropriately completes the request.
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
<prolog>
...
</prolog>
"""

SYSTEM_PROMPT_Q = "Below is an instruction that describes a task. Write a response that appropriately completes the request."

XML_COT_FORMAT = """\
<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>
<prolog>
{prolog}
</prolog>
"""

DUMMY_ANSWER = -99999


def prolog_output_matches_answer_term(prolog_output: str, expected_answer: str) -> bool:
    if prolog_output is None or expected_answer is None:
        return False
    try:
        lhs = parse(f"${prolog_term_to_latex(prolog_output)}$")
        rhs = parse(f"${prolog_term_to_latex(str(expected_answer))}$")
        return verify(lhs, rhs)
    except Exception as e:
        print(f"structured match failed: {e}")
        return str(prolog_output).strip() == str(expected_answer).strip()

def log_correct_model_output(input_text: str, domain: str, model_output: str, save_path: str):
    record = {
        "input": input_text.strip(),
        "domain": domain.strip(),
        "model_output": model_output.strip()
    }
    with open(save_path, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def save_prolog_code_to_file(prolog_code, file_path):
    with open(file_path, 'w') as file:
        file.write(prolog_code)

def evaluate_prolog_code(prolog_code: str) -> dict:
    """Run Prolog, retuen stdout, stderr"""
    with tempfile.NamedTemporaryFile("w", suffix=".pl", delete=False) as tmp:
        tmp.write(prolog_code)
        tmp_path = tmp.name

    try:
        result = subprocess.run(
            ["swipl", "-t", "halt", "-f", tmp_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=5
        )
        stdout = result.stdout.decode().strip()
        stderr = result.stderr.decode().strip()
        cleaned_stdout = re.sub(r"\s+", "", stdout)

        return {
            "result": cleaned_stdout if cleaned_stdout else None,
            "singleton_warning": "singleton" in stderr.lower(),
            "raw_stderr": stderr,
        }

    except subprocess.TimeoutExpired:
        return {
            "result": None,
            "singleton_warning": False,
            "raw_stderr": "Timeout",
        }
    finally:
        os.unlink(tmp_path)

def extract_xml_answer(text: str) -> str:
    return text.split("<answer>")[-1].split("</answer>")[0].strip()

def extract_prolog_code(text: str) -> str:
    return text.split("<prolog>")[-1].split("</prolog>")[0].strip()

def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()

def extract_suggested_predicates_from_prompt(prompt: str) -> set[str]:
    return set(re.findall(r'-\s*([a-z_][a-zA-Z0-9_]*)\s*/\d+', prompt))

def extract_used_predicates_from_prolog(code: str) -> set[str]:
    return set(re.findall(r'\b([a-z_][a-zA-Z0-9_]*)\s*\(', code))

# -------------------  Reward  -------------------
def prolog_correctness_reward_func(prompts, completions, answer, **kwargs) -> List[float]:
    if answer == DUMMY_ANSWER:
        return [0.0]

    prompt_text = prompts[0][-1]['content']
    suggested_predicates = extract_suggested_predicates_from_prompt(prompt_text)
    responses = [completion[0]['content'] for completion in completions]
    extracted_codes = [extract_prolog_code(r) for r in responses]
    save_path = "outputs/grpo_correct_model_outputs_full_data_MATH_vllm.jsonl"
    domain = kwargs.get("domain", "unknown")
    if isinstance(domain, list):
        domain = domain[0] if domain else "unknown"

    rewards = []
    for code, expected, full_response in zip(extracted_codes, answer, responses):
        eval_result = evaluate_prolog_code(code)
        used_predicates = extract_used_predicates_from_prolog(code)

        match = prolog_output_matches_answer_term(eval_result["result"], expected)
        eval_result["match"] = match

        reward = 0.0
        if not eval_result["result"]:
            reward = 0.0
        elif eval_result["match"]:
            reward = 2.0
            if eval_result["singleton_warning"]:
                reward -= 0.1
            if reward >= 1.9:
                log_correct_model_output(prompt_text, domain, full_response, save_path)

        rewards.append(reward)

    return rewards

def prolog_partial_reward(prompts, completions, answer, **kwargs) -> List[float]:
    if answer == DUMMY_ANSWER:
        return [0.0]

    prompt_text = prompts[0][-1]['content']
    suggested_predicates = extract_suggested_predicates_from_prompt(prompt_text)
    responses = [completion[0]['content'] for completion in completions]
    extracted_codes = [extract_prolog_code(r) for r in responses]

    domain = kwargs.get("domain", "unknown")
    if isinstance(domain, list):
        domain = domain[0] if domain else "unknown"

    rewards = []
    for code, expected, full_response in zip(extracted_codes, answer, responses):
        eval_result = evaluate_prolog_code(code)
        used_predicates = extract_used_predicates_from_prolog(code)
        match = prolog_output_matches_answer_term(eval_result["result"], expected)
        eval_result["match"] = match

        reward = 0.0
        if eval_result["result"] and not match:
            if suggested_predicates & used_predicates:
                reward = 0.5

        rewards.append(reward)

    return rewards

# ------------------- COT Reward -------------------
def compare_values(a, b) -> bool:
    try:
        return parse(f"${a}$") == parse(f"${b}$")
    except:
        return str(a).strip() == str(b).strip()

def cot_correctness_reward_func(prompts, completions, answer, **kwargs) -> List[float]:
    if answer == DUMMY_ANSWER:
        return [0.0]
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    extracted_responses = [extract_xml_answer(r) for r in responses]
    print('-'*20, f"Question:\n{q}", f"\nAnswer:\n{answer[0]}", f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0]}")
    return [2.0 if compare_values(r, a) else 0.0 for r, a in zip(extracted_responses, answer)]

# ------------------- Format -------------------
def strict_format_reward_func(completions, **kwargs) -> List[float]:
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n<prolog>\n.*?\n</prolog>\n$"
    responses = [completion[0]["content"] for completion in completions]
    return [0.5 if re.match(pattern, r) else 0.0 for r in responses]

def soft_format_reward_func(completions, **kwargs) -> List[float]:
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>\s*<prolog>.*?</prolog>"
    responses = [completion[0]["content"] for completion in completions]
    return [0.5 if re.match(pattern, r) else 0.0 for r in responses]

def count_xml(text) -> float:
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    if text.count("<answer>\n") == 1:
        count += 0.125
    if text.count("\n</answer>\n") == 1:
        count += 0.125
    if text.count("\n<prolog>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</prolog>\n")[-1]) * 0.001
    if text.count("\n</prolog>") == 1:
        count += 0.125
        count -= (len(text.split("\n</prolog>")[-1]) - 1) * 0.001
    return count

def xmlcount_reward_func(completions, **kwargs) -> List[float]:
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]
