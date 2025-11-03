import re
from datasets import load_dataset, Dataset
import torch
from unsloth import FastLanguageModel, PatchFastRL
from unsloth import is_bfloat16_supported
from trl import GRPOConfig, GRPOTrainer
from prolog_reward_vllm import SYSTEM_PROMPT,prolog_partial_reward, prolog_correctness_reward_func
import os
def main():
    PatchFastRL("GRPO", FastLanguageModel)
    max_seq_length = 2048
    lora_rank = 32

    def get_questions(split="train"):
        # TODO: specify your dataset path, e.g. data_files="data/grpo_train.jsonl"

        data = load_dataset("json", data_files=None, split="train")

        def format_prompt(example):
            format_instruction = (
                "The final answer must be a valid Prolog term using one of the following formats:\n"
                "- 17 (plain number)\n"
                "- decimal(Value) | complex(Re, Im) | frac(N, D) | mixed(W, frac(N, D))\n"
                "- eq(Var, Val) | expr(...) | symbolic_constant\n"
                "- interval(A, B) | interval_union([I1, I2, ...]) | solution([X1, X2, ...])\n"
                "- plus_minus(A, sqrt(B)) | sqrt(X) | base(Base, Digits)\n"
                "- vector([...]) | matrix([[...], [...]]) | trig(Func, Arg)\n"
                "- formatted_number(\"...\") | unit(Value, UnitType)\n"
                "Use write(...) to output the final answer. Do NOT include explanations or natural language."
            )

            base_input = example["input"].strip() + "\n\n" + format_instruction
            predicates = example.get("predicates", [])

            if predicates:
                pred_lines = ["Suggested predicates:"]
                for p in predicates:
                    if "name" in p and "arity" in p and "defined" in p:
                        header = f"- {p['name']}/{p['arity']}:"
                        definition = p["defined"].strip()
                        indented_def = "\n".join("  " + line for line in definition.split("\n"))
                        pred_lines.append(f"{header}\n{indented_def}")
                pred_hint = "\n\n" + "\n".join(pred_lines)
            else:
                pred_hint = "\n\nSuggested predicates: generate necessary predicates to solve the problem."

            return {
                'prompt': [
                    {'role': 'system', 'content': SYSTEM_PROMPT},
                    {'role': 'user', 'content': base_input + pred_hint}
                ],
                'answer': example['answer']
            }

        data = data.map(format_prompt)
        return data

    model, tokenizer = FastLanguageModel.from_pretrained(
        # TODO: specify your model path, e.g. "models/qwen2.5_3B_sft_prolog"
        model_name = None,
        max_seq_length = max_seq_length,
        load_in_4bit = True,
        fast_inference = True,
        max_lora_rank = lora_rank,
        gpu_memory_utilization = 0.5,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r = lora_rank,
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha = lora_rank,
        use_gradient_checkpointing = "unsloth",
        random_state = 3407,
    )

    dataset = get_questions()
    print("DONE loading questions")

    training_args = GRPOConfig(
        use_vllm = False,
        learning_rate = 3e-6,
        adam_beta1 = 0.9,
        adam_beta2 = 0.95,
        weight_decay = 0.1,
        warmup_ratio = 0.1,
        lr_scheduler_type = "cosine",
        logging_steps = 1,
        bf16 = is_bfloat16_supported(),
        fp16 = not is_bfloat16_supported(),
        per_device_train_batch_size = 6,
        gradient_accumulation_steps = 8,
        num_generations = 6,
        max_prompt_length = 640,
        max_completion_length = 1408,
        num_train_epochs = 20,
        save_steps = 100,
        max_grad_norm = 0.1,
        report_to = "wandb",
        output_dir = "outputs/prolog_epich_10_rank32_vllm",
        vllm_gpu_memory_utilization = 0.4,
    )

    print("GRPOTrainer time:")
    trainer = GRPOTrainer(
        model = model,
        processing_class = tokenizer,
        reward_funcs = [
            prolog_correctness_reward_func,
	prolog_partial_reward,
        ],
        args = training_args,
        train_dataset = dataset,
    )
    trainer.train()

if __name__ == "__main__":
    main()
