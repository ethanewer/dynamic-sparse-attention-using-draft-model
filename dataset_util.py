import re
from string import ascii_uppercase

import pandas as pd  # type: ignore
import torch
from torch import Tensor


def encode_longbench_example(
    example,
    tokenizer,
    device,
) -> tuple[Tensor, Tensor, Tensor]:
    prompt = f"""\
Please read the following text and answer the question below.

{example["context"]}

What is the correct answer to this question: {example["question"]}
Choices:
(A) {example["choice_A"]}
(B) {example["choice_B"]}
(C) {example["choice_C"]}
(D) {example["choice_D"]}

Format your response as follows: "The correct answer is (insert answer here)".

Let's think step by step.
"""

    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        padding=True,
        padding_side="left",
        return_tensors="pt",
        return_dict=True,
    )
    input_ids: Tensor = inputs["input_ids"].to(device)  # type: ignore
    attention_mask: Tensor = inputs["attention_mask"].to(device)  # type: ignore
    position_ids = torch.arange(input_ids.shape[1], device=device)[None]
    return input_ids, attention_mask, position_ids


def format_mmlu_question(example):
    prompt = "Given the following question and candidate answers, choose the best answer.\nQuestion: "
    question = example["question"]
    options = example["options"]
    prompt += question + "\n"
    for i, option in enumerate(options):
        prompt += f"{ascii_uppercase[i]}. {option}\n"
    prompt += "\n"
    prompt += 'Your response should end with "The answer is (X)." where X is a letter from the provided choices.\n'
    prompt += "Each reasoning step in your response should be delimited by two newline characters\n\n"
    prompt += "Let's think step by step.\n\n"
    return prompt


def encode_mmlu_example(
    example,
    fewshot_df_path,
    tokenizer,
    device,
) -> tuple[Tensor, Tensor, Tensor]:
    subject = example["category"]
    fewshot_df = pd.read_json(fewshot_df_path)
    filtered_fewshot_df = fewshot_df[fewshot_df["category"] == subject]

    messages = []
    for i in range(len(filtered_fewshot_df)):
        fewshot_example = filtered_fewshot_df.iloc[i]
        messages.append(
            {"role": "user", "content": format_mmlu_question(fewshot_example)}
        )
        messages.append(
            {"role": "assistant", "content": f"{fewshot_example['cot_content']}\n\n"}
        )
    messages.append({"role": "user", "content": format_mmlu_question(example)})

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        padding=True,
        padding_side="left",
        return_tensors="pt",
        return_dict=True,
    )

    input_ids: Tensor = inputs["input_ids"].to(device)  # type: ignore
    attention_mask: Tensor = inputs["attention_mask"].to(device)  # type: ignore
    position_ids = torch.arange(input_ids.shape[1], device=device)[None]
    return input_ids, attention_mask, position_ids


def extract_answer_multiple_choice(text: str, start_tag="<|im_start|>") -> str | None:
    text = text[text.rfind(start_tag) :]

    pattern = r"answer is \(?([A-J])\)?"
    match = re.search(pattern, text)
    if match:
        return match.group(1).upper()
    else:
        match = re.search(r".*[aA]nswer:\s*([A-J])", text)
        if match:
            return match.group(1).upper()
        else:
            return None


def grade_multiple_choice(text: str, ground_truth_answer: str | None) -> bool:
    answer = extract_answer_multiple_choice(text)
    if ground_truth_answer is None:
        return True
    elif answer is None:
        return False
    elif answer.upper() == ground_truth_answer.upper():
        return True
    else:
        return False
