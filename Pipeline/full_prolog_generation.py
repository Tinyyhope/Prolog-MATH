import os
import json
import re
import subprocess
import tempfile
import argparse
from tqdm import tqdm
from openai import OpenAI

class DeepSeekPrologSolver:
    def __init__(self, api_key, predicate_jsonl, error_file):
        self.client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        self.error_file = error_file
        self.predicate_map = self._load_predicates(predicate_jsonl)

    def _load_predicates(self, jsonl_path):
        mapping = {}
        with open(jsonl_path, 'r') as f:
            for line in f:
                entry = json.loads(line)
                mapping[entry["problem"]] = entry  # use problem text as key
        return mapping

    def _run_check(self, code, expected):
        with tempfile.NamedTemporaryFile("w", suffix=".pl", delete=False) as tmp:
            tmp.write(code)
            tmp_path = tmp.name
        try:
            result = subprocess.run(
                ["swipl", "-q", "-t", "halt", "-f", tmp_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=5
            )
            output = result.stdout.decode().strip()
            return expected in output, output  # return both result and output for logging
        except subprocess.TimeoutExpired:
            return False, "TIMEOUT"
        finally:
            os.unlink(tmp_path)

    def _build_prompt(self, problem, cot_solution, predicate_entry):
        lines = []
        for op in predicate_entry.get("predefined_operators", []):
            lines.append(f"<< {op} >>")
        for op in predicate_entry.get("generated_operators", []):
            call = op["call"]
            definition = op["definition"]
            example = op.get("example", "")
            lines.append(f"<< {call} >>\n{definition}\n% Example: {example}")
        pred_text = "\n".join(lines) or "None"
        answer_type_hint = """
The final answer should be written as a structured Prolog term.
You must use one of the following formats, depending on the type of answer:

- complex(Re, Im)
- frac(Numer, Denom)
- mixed(Whole, frac(N, D))
- eq(Variable, Value)
- expr(...)
- symbolic_constant
- interval(A, B)
- interval_union([I1, I2, ...])
- solution([X1, X2, ...])
- plus_minus(A, sqrt(B))
- sqrt(X)
- base(Base, Digits)
- vector([X, Y, Z])
- matrix([[A, B], [C, D], ...])
- trig(Function, Arg)
- decimal(Value)
- formatted_number("...")
- unit(Value, UnitType)

Each answer must be returned as a valid Prolog term. Do not print explanations or natural language.
"""
        return [
            {"role": "system", "content": "You are a helpful Prolog programming assistant."},
            {"role": "user", "content": f"""
Problem:
{problem}

CoT Solution:
{cot_solution}

Use the following predicates (define them if needed):
{pred_text}
{answer_type_hint}
Now generate a single complete Prolog program that solves the problem.
Do NOT use markdown formatting (no ```prolog).
Do NOT include any natural language or explanation.
Only output pure Prolog code, with % comments if needed.
End your code with `:- solve, halt.` so the file is runnable.
"""}
        ]

    def retry_errors(self, rescued_file, max_retry=10):
        with open(self.error_file, 'r') as f:
            lines = f.readlines()

        remaining_lines = lines[max_retry:]
        selected = lines[:max_retry]
        updated_errors = []

        print(f"🔍 Loaded {len(lines)} error entries, retrying {len(selected)}...")

        for line in tqdm(selected, desc="Retrying"):
            entry = json.loads(line)
            problem = entry["problem"]
            cot_solution = entry["solution"]
            expected = entry["expected"]

            
            predicate_entry = self.predicate_map.get(problem, None)
            if not predicate_entry:
                print(f"⚠️ No predicate entry found for problem:\n{problem[:60]}...")
                updated_errors.append(line)
                continue

            final_code = ""
            success = False
            output = ""

            for attempt in range(3):
                try:
                    prompt = self._build_prompt(problem, cot_solution, predicate_entry)
                    resp = self.client.chat.completions.create(
                        model="deepseek-chat",
                        messages=prompt,
                        temperature=0.3,
                        max_tokens=1500
                    )
                    code = resp.choices[0].message.content.strip()
                    final_code = code
                    passed, actual_output = self._run_check(code, expected)
                    output = actual_output
                    if passed:
                        success = True
                        break
                except Exception as e:
                    print(f"Exception during generation: {e}")
                    final_code = f"% Exception: {str(e)}"

            new_entry = {
                "problem": problem,
                "solution": final_code,
                "expected": expected,
                "actual": output,
                "reason": None if success else "Retry failed"
            }

            if success:
                with open(rescued_file, 'a') as f_out:
                    json.dump(new_entry, f_out)
                    f_out.write("\n")
            else:
                updated_errors.append(json.dumps(entry) + "\n")

            
            with open(self.error_file, 'w') as f_err:
                f_err.writelines(updated_errors + remaining_lines)

        print(f"✅ Retried {len(selected)} problems. Rescued: {len(selected) - len(updated_errors)}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_retry", type=int, default=10, help="Number of problems to retry.")
    args = parser.parse_args()

    API_KEY = "Your api_key here."
    PREDICATE_FILE = "data/rolog_math_output.jsonl"
    ERROR_FILE = "data/error_output.jsonl"
    RESCUED_FILE = "data/rescued_problem.jsonl"


    solver = DeepSeekPrologSolver(
        api_key=API_KEY,
        predicate_jsonl=PREDICATE_FILE,
        error_file=ERROR_FILE
    )

    solver.retry_errors(
        rescued_file=RESCUED_FILE,
        max_retry=args.max_retry
    )
