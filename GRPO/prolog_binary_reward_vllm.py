import os
import re
import subprocess
import tempfile
import json
import subprocess

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
from math_verify import parse, verify
from math_verify_prolog import prolog_term_to_latex

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
    """Log successful model generations with domain, input, and output."""
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


import tempfile
import subprocess
import os
import re

def evaluate_prolog_code(prolog_code: str) -> dict:
    """Run Prolog code and return the stdout result, along with warnings and stderr."""

    with tempfile.NamedTemporaryFile("w", suffix=".pl", delete=False) as tmp:
        tmp.write(prolog_code)
        tmp_path = tmp.name

    try:
        result = subprocess.run(
            ["swipl", "-t", "halt", "-f", tmp_path],  # -q 可加快，不建议省略错误输出
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
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

def extract_prolog_code(text: str) -> str:
    answer = text.split("<prolog>")[-1]
    answer = answer.split("</prolog>")[0]
    return answer.strip()

def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()

# Reward functions
def extract_suggested_predicates_from_prompt(prompt: str) -> set[str]:
    """
    Extract suggested predicate names from prompt.
    Matches lines like: - predicate_name/arity:
    """
    return set(re.findall(r'-\s*([a-z_][a-zA-Z0-9_]*)\s*/\d+\s*:', prompt))

def extract_used_predicates_from_prolog(code: str) -> set[str]:
    """
    Extract all predicate names used in Prolog code.
    Matches calls like: name(...)
    """
    return set(re.findall(r'\b([a-z_][a-zA-Z0-9_]*)\s*\(', code))

#     return rewards
def prolog_correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    if answer == DUMMY_ANSWER:
        return [0.0]

    prompt_text = prompts[0][-1]['content']
    responses = [completion[0]['content'] for completion in completions]
    extracted_codes = [extract_prolog_code(r) for r in responses]

    save_path = "outputs/grpo_correct_model_outputs_binary_vllm.jsonl"
    domain = kwargs.get("domain", "unknown")

    if isinstance(domain, list):
        domain = domain[0] if domain else "unknown"

    rewards = []
    for code, expected, full_response in zip(extracted_codes, answer, responses):
        eval_result = evaluate_prolog_code(code)
        match = prolog_output_matches_answer_term(eval_result["result"], expected)

        if match:
            log_correct_model_output(prompt_text, domain, full_response, save_path)
            reward = 2.0
        else:
            reward = 0.0

        print("-" * 40)
        print(f"Prompt:\n{prompt_text}")
        print(f"Expected answer: {expected}")
        print(f"Prolog output: {eval_result['result']}")
        print(f"Match: {match}")
        print(f"Reward: {reward}\n")

        rewards.append(reward)

    return rewards




def cot_correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    if answer == DUMMY_ANSWER:
        return 0
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    extracted_responses = [extract_xml_answer(r) for r in responses]
    print('-'*20, f"Question:\n{q}", f"\nAnswer:\n{answer[0]}", f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0]}")
    return [2.0 if compare_values(r, a) else 0.0 for r, a in zip(extracted_responses, answer)]

def consistent_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    prolog_extracted_responses = [extract_prolog_code(r) for r in responses]
    cot_extracted_responses = [extract_xml_answer(r) for r in responses]
    rewards = [2.0 if compare_values(evaluate_prolog_code(r), a) else 0.5 if evaluate_prolog_code(r) else 0.0 for r, a in zip(prolog_extracted_responses, cot_extracted_responses)]
    prolog_ans_0 = evaluate_prolog_code(prolog_extracted_responses[0])
    print('-'*20, f"Question:\n{q}", f"\nAnswer:\n{answer[0]}", f"\nResponse:\n{responses[0]}", f"\nExtracted COT:\n#{cot_extracted_responses[0]}#\nExtracted Prolog code:\n{prolog_extracted_responses[0]} evaluated_prolog_ans=#{prolog_ans_0}# consistent reward={rewards[0]}\n")
    return rewards

def int_reward_func(completions, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]

def solve_reward_func(completions, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    pattern = r"solve\(([^)]+)\).*?\b\1\b"
    return [0.5 if re.search(pattern, r, re.DOTALL) else 0.0 for r in responses]


def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n<prolog>\n.*?\n</prolog>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>\s*<prolog>.*?</prolog>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

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
        count -= len(text.split("\n</prolog>\n")[-1])*0.001
    if text.count("\n</prolog>") == 1:
        count += 0.125
        count -= (len(text.split("\n</prolog>")[-1]) - 1)*0.001
    return count

def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]
