import os
import json
import re
import subprocess
import tempfile
from openai import OpenAI
import chardet

class DeepSeekPrologSolver:
    def __init__(self, api_key, problems_dir, operators_file, output_file="output.jsonl", max_problems=10):
        self.client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        self.problems_dir = problems_dir
        self.operators_file = operators_file
        self.output_file = output_file
        self.max_problems = max_problems
        self.operators_list = self._load_operators()
        self.seen_generated_definitions = set()

    def detect_encoding(self, file_path):
        with open(file_path, "rb") as f:
            raw_data = f.read(1024)
        result = chardet.detect(raw_data)
        return result["encoding"]

    def _load_operators(self):
        operators = []
        encoding = self.detect_encoding(self.operators_file)
        with open(self.operators_file, "r", encoding=encoding, errors="replace") as file:
            for line in file:
                if "(" in line and ")" in line:
                    op_name = line.split("(")[0].strip()
                    operators.append(op_name)
        return operators

    def _check_prolog_syntax(self, code_str):
        with tempfile.NamedTemporaryFile("w", suffix=".pl", delete=False) as tmp_file:
            tmp_file.write(code_str)
            tmp_path = tmp_file.name

        result = subprocess.run(
            ["swipl", "-q", "-t", "halt", "-f", tmp_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        os.unlink(tmp_path)
        return result.returncode == 0, result.stderr.decode()

    def _run_all_definitions(self, all_def_code):
        with tempfile.NamedTemporaryFile("w", suffix=".pl", delete=False) as tmp_file:
            tmp_file.write(all_def_code)
            tmp_path = tmp_file.name

        try:
            result = subprocess.run(
                ["swipl", "-q", "-t", "halt", "-f", tmp_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=5
            )
            os.unlink(tmp_path)
            return result.returncode == 0, result.stderr.decode().strip()
        except subprocess.TimeoutExpired:
            os.unlink(tmp_path)
            return False, "Timeout: Prolog execution took too long"

    def _extract_numbers(self, text):
        return set(re.findall(r'\d+', text))

    def _generate_prolog_interface(self, problem, solution, numbers):
        messages = [
            {"role": "system", "content": "You are a symbolic reasoning assistant. Define any new Prolog predicates needed to solve the problem below, and reuse the given background predicates when appropriate."},
            {"role": "user", "content": f"""
Given the problem:

{problem}

and its detailed solution:

{solution}

Using the available Prolog operators:
{', '.join(self.operators_list)}

Generate a list of Prolog calls needed to solve the problem. You may also define new operators if necessary.

For each operator call, enclose it in double angle brackets << >>.

Then clearly separate the results into:

1. Predefined Operators: just a list of calls like << add(2,3,X) >>
2. Generated Operators: for each, include:
   - the call (in << >>)
   - its full Prolog definition (e.g., max(A,B,Max) :- ...)
   - one usage example that will succeed

DO NOT use any helper predicate (like sum_list, pow, etc.) unless you also provide its full definition.
You must define every predicate that is used inside any new operator you write.
Do not solve the problem. Do not include natural language. Only output the Prolog
predicate definitions.
"""}
        ]

        response = self.client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            temperature=0.3,
            max_tokens=1500
        )

        if not response or not getattr(response, "choices", None):
            print("❌ DeepSeek API response is empty or invalid!")
            return [], [], [{"call": None, "definition": "", "example": "", "reason": "API returned no response"}]

        raw_response = response.choices[0].message.content.strip()
        print(f"\n📤 Raw Response from DeepSeek:\n{'='*60}\n{raw_response}\n{'='*60}\n")

        return self._parse_deepseek_output(raw_response)

    def _parse_deepseek_output(self, response_text):
        predefined_ops = []
        valid_generated_ops = []
        invalid_generated_ops = []

        all_calls = re.findall(r'<<\s*(.*?)\s*>>', response_text)
        all_calls_clean = [c for c in all_calls if ":-" not in c and "=" not in c and "/" not in c]

        predefined_ops = [c for c in all_calls_clean if c.split("(")[0].strip() in self.operators_list]
        generated_calls = [c for c in all_calls_clean if c.split("(")[0].strip() not in self.operators_list]

        chunks = re.split(r"<<\s*(.*?)\s*>>", response_text)
        def_map = {}
        all_def_code = ""

        for i in range(1, len(chunks), 2):
            call = chunks[i].strip()
            definition = chunks[i + 1].strip() if i + 1 < len(chunks) else ""
            name = call.split("(")[0].strip()
            if ":-" in definition and name not in self.seen_generated_definitions:
                self.seen_generated_definitions.add(name)
                def_map[name] = definition
                all_def_code += definition + "\n"

        all_ok, compile_err = self._run_all_definitions(all_def_code)

        for call in generated_calls:
            name = call.split("(")[0].strip()
            full_def = def_map.get(name, "")

            if not full_def:
                invalid_generated_ops.append({
                    "call": call,
                    "definition": "",
                    "example": "",
                    "reason": "Missing definition"
                })
                continue

            example = next((e for e in all_calls if e.startswith(name + "(") and e != call), call)

            if not all_ok:
                invalid_generated_ops.append({
                    "call": call,
                    "definition": full_def,
                    "example": example,
                    "reason": f"Global compile error: {compile_err}"
                })
                continue

            valid_generated_ops.append({
                "call": call,
                "definition": full_def,
                "example": example,
                "output": "Compiled OK"
            })

        return predefined_ops, valid_generated_ops, invalid_generated_ops

    def process_problems(self):
        output_data = []

        problem_files = [f for f in os.listdir(self.problems_dir) if f.endswith(".json")]
        sorted_files = sorted(problem_files, key=lambda x: int(x.split(".")[0]))
        selected_files = sorted_files[:self.max_problems]

        for filename in selected_files:
            file_path = os.path.join(self.problems_dir, filename)
            with open(file_path, "r") as f:
                problem_data = json.load(f)

            problem = problem_data.get("problem", "")
            solution = problem_data.get("solution", "")
            numbers = self._extract_numbers(problem)

            predefined_ops, valid_generated_ops, invalid_generated_ops = self._generate_prolog_interface(problem, solution, numbers)

            output_data.append({
                "problem": problem,
                "id": filename.split(".")[0],
                "predefined_operators": predefined_ops,
                "generated_operators": valid_generated_ops,
                "invalid_generated_operators": invalid_generated_ops
            })

            print(f"Problem ID: {filename.split('.')[0]}")
            print(f"Valid Generated Operators: {[op['call'] for op in valid_generated_ops]}")
            print(f"Invalid Generated Operators: {[op['call'] for op in invalid_generated_ops]}\n")

        with open(self.output_file, "w", encoding="utf-8") as f:
            for entry in output_data:
                json.dump(entry, f, ensure_ascii=False)
                f.write("\n")

        print(f"✅ Processing complete! Results saved to {self.output_file}")


if __name__ == "__main__":
    API_KEY = "Your api_key here"
    ## Example problem domain, change to your task
    PROBLEMS_DIR = "data/number_theory"
    OPERATORS_FILE = "data/operator.pl"
    OUTPUT_FILE = "outputs/prolog_math_output.jsonl"

    solver = DeepSeekPrologSolver(API_KEY, PROBLEMS_DIR, OPERATORS_FILE, OUTPUT_FILE)
    solver.process_problems()
