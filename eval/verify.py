#!/usr/bin/env python3
"""
Script to evaluate VLM predictions using CompassVerifier model via vLLM.
Reads jsonl files from a directory and evaluates predictions, grouping by dataset.
"""

import argparse
import os
import glob
import json
import pandas as pd
from typing import List, Dict
from collections import defaultdict
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from collections import defaultdict
import re

CV_PROMPT = """
Please as a grading expert, judge whether the final answers given by the candidates below are consistent with the standard answers, that is, whether the candidates answered correctly. 
Here are some evaluation criteria:
1. Please refer to the given standard answer. You don't need to re-generate the answer to the question because the standard answer has been given. You only need to judge whether the candidate's answer is consistent with the standard answer according to the form of the question. THE STANDARD ANSWER IS ALWAYS CORRECT AND THE QUESTION IS PERFECTLY VALID. NEVER QUESTION THEM.
2. ONLY compare the FINAL ANSWER - COMPLETELY IGNORE any potential errors in the REASONING PROCESSES.
3. Some answers may be expressed in different ways, such as some answers may be a mathematical expression, some answers may be a textual description, as long as the meaning expressed is the same. Before making a judgment, please understand the question and the standard answer first, and then judge whether the candidate's answer is correct.
4. Some answers may consist of multiple items, such as multiple-choice questions, multiple-select questions, fill-in-the-blank questions, etc. Regardless of the question type, the final answer will be considered correct as long as it matches the standard answer, regardless of whether the reasoning process is correct. For multiple-select questions and multi-blank fill-in-the-blank questions, all corresponding options or blanks must be answered correctly and match the standard answer exactly to be deemed correct.
5. If the prediction is given with \\boxed{{}}, please ignore the \\boxed{{}} and only judge whether the candidate's answer is consistent with the standard answer.
6. If the candidate's answer is invalid (e.g., incomplete (cut off mid-response), lots of unnormal repetitive content, or irrelevant to the question, saying it can't answer the question because some irresistible factors, like ethical issues, no enough information, etc.), select option C (INVALID).Please judge whether the following answers are consistent with the standard answer based on the above criteria. Grade the predicted answer of this new question as one of:
A: CORRECT 
B: INCORRECT
C: INVALID
Just return the letters "A", "B", or "C", with no text around it.
Here is your task. Simply reply with either CORRECT, INCORRECT, or INVALID. Don't apologize or correct yourself if there was a mistake; we are just trying to grade the answer.
<Original Question Begin>:
{question}
<Original Question End>
<Standard Answer Begin>:
{gold_answer}
<Standard Answer End>
<Candidate's Answer Begin>: 
{llm_response}
<Candidate's Answer End>
Judging the correctness of the candidate's answer:
"""


CV_COT_PROMPT = """As a grading expert, your task is to determine whether the candidate's final answer matches the provided standard answer. Follow these evaluation guidelines precisely:

Evaluation Protocol:
1. Reference Standard:
   - The standard answer is definitive and always correct
   - The question is perfectly valid - never question them
   - Do not regenerate answers; only compare with the given standard

2. Comparison Method:
   - Carefully analyze the question's requirements and the standard answer's structure
     * Determine whether the question expects exact matching of the entire standard answer or allows partial matching of its components.
     * This determination must be made based on the question's phrasing and the nature of the standard answer.
   - Compare ONLY the candidate's final answer (ignore all reasoning/explanation errors)
   - Disregard any differences in formatting or presentation style
   - For mathematical expressions: calculate step by step whether the two formulas are equivalent
   - For multiple-choice questions: compare only the final choice and corresponding option content

3. Multi-part Answers:
   - For questions requiring multiple responses (e.g., multi-select):
   - All parts must match the standard answer exactly. 
   - Compare each sub-answer step by step. Partial matches are considered incorrect.

4. Validity Check:
   - Reject answers that are:
     * Incomplete (cut off mid-sentence in the final sentence, lacking a complete response) → Label as INCOMPLETE
     * Repetitive (repetition of words or phrases in a loop) → Label as REPETITIVE
     * Explicit refusals (e.g., directly return "I cannot answer/provide/access ...") → Label as REFUSAL
   - For invalid answers, specify the type in the judgment (e.g., \\boxed{{C}} - INCOMPLETE).

Grading Scale:
\\boxed{{A}} - CORRECT: 
   - Answer matches standard exactly (including equivalent expressions)
   - For numerical answers: consider as equivalent if values match when rounded appropriately
   - Semantically equivalent responses

\\boxed{{B}} - INCORRECT:
   - Any deviation from standard answer
   - Partial matches for multi-part questions

\\boxed{{C}} - INCOMPLETE/REPETITIVE/REFUSAL:
   - Fails validity criteria above (must specify: INCOMPLETE/REPETITIVE/REFUSAL)

Execution Steps and Output Formats:

Analysis step by step: [
Thoroughly evaluate the candidate's answer including:
(1) First check if the answer is INCOMPLETE (cut off mid-sentence), REPETITIVE (looping repetition), or a REFUSAL (explicit denial) - if so, immediately classify as \\boxed{{C}} with the corresponding type.
(2) Analyze the question's core requirements and the standard answer's structure, for example:
- Strict requirements: Identify mandatory constraints (e.g., simplification, answer order, multi-part completeness)
- Tolerant allowances: Ignore non-critical deviations (e.g., missing option labels in MCQs, equivalent but unformatted expressions)
- Required answer type, precision level, etc.
(3) Perform a detailed comparison between the candidate's final answer and the standard answer, for example:
- Content equivalence
- Permitted variations in numerical precision
- Allowed expression formats]
Final Judgment: \\boxed{{A/B/C}} - <CORRECT/INCORRECT/INCOMPLETE/REPETITIVE/REFUSAL>

Here is your task.
<Original Question Begin>
{question}
<Original Question End>

<Standard Answer Begin>
{gold_answer}
<Standard Answer End>

<Candidate's Answer Begin>
{llm_response}
<Candidate's Answer End>

Analysis step by step and Final Judgment:
"""

SYSTEM_PROMPT = "Please as a grading expert, judge whether the final answers given by the candidates below are consistent with the standard answers, that is, whether the candidates answered correctly."


def load_model(model_path: str, tensor_parallel_size: int = 8):
    """Load the CompassVerifier model using vLLM."""
    # Load model with vLLM
    model = LLM(
        model=model_path,
        trust_remote_code=True,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=32768,
    )
    
    # Get tokenizer
    tokenizer = model.get_tokenizer()
    
    print("Model loaded successfully!")
    return model, tokenizer


def format_prompt(question: str, answer: str, prediction: str) -> List[dict]:
    """Format the prompt for the verifier model as chat messages."""
    prompt = CV_COT_PROMPT.format(
        question=question,
        gold_answer=answer,
        llm_response=prediction[-5000:]
    )
    
    # Format as chat messages
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ]
    
    return messages


def load_jsonl(file_path: str) -> List[Dict]:
    """Load a jsonl file and return list of dictionaries."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def save_jsonl(data: List[Dict], file_path: str):
    """Save list of dictionaries to a jsonl file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def process_judgment(judgment_str: str) -> str:
    # First try to find the exact \boxed{letter} pattern
    boxed_matches = re.findall(r'boxed{([A-C])}', judgment_str)
    if boxed_matches:
        return boxed_matches[-1]
    
    # Directly return the judgment if it is A, B, or C
    if judgment_str in ["A", "B", "C"]:
        return judgment_str
    else:
        final_judgment_str = judgment_str.split("Final Judgment:")[-1]
        matches = re.findall(r'\(([A-C])\)*', final_judgment_str)
        if matches:
            return matches[-1]
        matches = re.findall(r'([A-C])', final_judgment_str)
        if matches:
            return matches[-1]
        return ""


def process_jsonl_file(file_path: str, model: LLM, tokenizer) -> List[Dict]:
    """Process a single jsonl file and add verification results."""
    print(f"\nProcessing file: {file_path}")
    
    # Load the jsonl file
    data = load_jsonl(file_path)
    print(f"  Loaded {len(data)} records")
    
    if not data:
        print(f"  Warning: No data in {file_path}. Skipping...")
        return []
    
    # Check if required fields exist
    required_fields = ['question', 'answer', 'prediction']
    sample = data[0]
    missing_fields = [field for field in required_fields if field not in sample]
    if missing_fields:
        print(f"  Warning: Missing fields {missing_fields} in {file_path}. Skipping...")
        return data
    
    # Prepare prompts and format them using tokenizer's chat template
    formatted_prompts = []
    for item in data:
        question = str(item.get('question', ''))
        answer = str(item.get('answer', ''))
        prediction = str(item.get('prediction', ''))
        
        messages = format_prompt(question, answer, prediction)
        # Apply chat template to format messages
        formatted_prompt = tokenizer.apply_chat_template(
            messages, 
            add_generation_prompt=True, 
            tokenize=False
        )
        
        # Remove bos_token if present (vLLM handles this automatically)
        if tokenizer.bos_token and formatted_prompt.startswith(tokenizer.bos_token):
            formatted_prompt = formatted_prompt.removeprefix(tokenizer.bos_token)
        
        formatted_prompts.append(formatted_prompt)
    
    # Configure sampling parameters
    sampling_params = SamplingParams(
        temperature=1e-6,
        top_p=0.9,
        top_k=1,
        max_tokens=2048,
    )
    
    # Generate predictions in batches
    batch_size = 64
    all_results = []
    
    for i in range(0, len(formatted_prompts), batch_size):
        batch = formatted_prompts[i:i + batch_size]
        print(f"  Processing batch {i//batch_size + 1}/{(len(formatted_prompts) + batch_size - 1)//batch_size} ({len(batch)} items)")
        
        try:
            # Call vLLM model
            outputs = model.generate(batch, sampling_params)
            
            # Extract text from outputs
            batch_results = []
            for output in outputs:
                generated_text = output.outputs[0].text
                batch_results.append(generated_text)
            
            all_results.extend(batch_results)
        except Exception as e:
            print(f"  Error processing batch: {e}")
            import traceback
            traceback.print_exc()
            # Add empty results for failed batch
            all_results.extend([""] * len(batch))
    
    # Add judge results to data
    for i, item in enumerate(data):
        item['judge'] = process_judgment(all_results[i])
        item['judge_prompt'] = all_results[i]
    
    return data


def calculate_accuracy_by_dataset(data: List[Dict], output_dir: str):
    """Group data by dataset and calculate accuracy for each."""
    # Group by dataset
    dataset_groups = defaultdict(list)
    for item in data:
        dataset_name = item.get('dataset', 'unknown')
        dataset_groups[dataset_name].append(item)
    
    print("\n" + "=" * 60)
    print("Accuracy Results by Dataset")
    print("=" * 60)
    
    overall_results = []
    
    for dataset_name, items in sorted(dataset_groups.items()):
        total_count = len(items)
        a_count = sum(1 for item in items if item.get('judge') == 'A')
        b_count = sum(1 for item in items if item.get('judge') == 'B')
        c_count = sum(1 for item in items if item.get('judge') == 'C')
        other_count = total_count - a_count - b_count - c_count
        
        accuracy = a_count / total_count if total_count > 0 else 0.0
        
        print(f"\n{dataset_name}:")
        print(f"  Total: {total_count}")
        print(f"  A (Correct): {a_count}")
        print(f"  B (Incorrect): {b_count}")
        print(f"  C (Invalid): {c_count}")
        if other_count > 0:
            print(f"  Other: {other_count}")
        print(f"  Accuracy: {accuracy:.4f} ({a_count}/{total_count})")
        
        # Save results for this dataset
        results_path = os.path.join(output_dir, f"{dataset_name}_results.jsonl")
        save_jsonl(items, results_path)
        print(f"  Saved results to: {results_path}")
        
        # Save accuracy summary for this dataset
        acc_data = {
            'dataset': dataset_name,
            'total': total_count,
            'A_count': a_count,
            'B_count': b_count,
            'C_count': c_count,
            'accuracy': accuracy
        }
        acc_path = os.path.join(output_dir, f"{dataset_name}_acc.json")
        with open(acc_path, 'w', encoding='utf-8') as f:
            json.dump(acc_data, f, indent=2, ensure_ascii=False)
        print(f"  Saved accuracy to: {acc_path}")
        
        overall_results.append(acc_data)
    
    # Save overall summary
    summary_path = os.path.join(output_dir, "overall_summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(overall_results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved overall summary to: {summary_path}")
    
    # Calculate and print overall accuracy
    total_all = sum(r['total'] for r in overall_results)
    correct_all = sum(r['A_count'] for r in overall_results)
    overall_acc = correct_all / total_all if total_all > 0 else 0.0
    print(f"\nOverall Accuracy: {overall_acc:.4f} ({correct_all}/{total_all})")
    
    return overall_results


def main():
    parser = argparse.ArgumentParser(description='Evaluate VLM predictions using CompassVerifier')
    parser.add_argument(
        '--model_name',
        type=str,
        default='starriver030515/ChartVerse-8B',
        help='Path to CompassVerifier model'
    )
    parser.add_argument(
        '--verifier_model_path',
        type=str,
        default='opencompass/CompassVerifier-7B',
        help='Path to CompassVerifier model'
    )
    parser.add_argument(
        '--tensor_parallel_size',
        type=int,
        default=1,
        help='Number of GPUs for tensor parallelism'
    )
    
    args = parser.parse_args()
    
    model = os.path.basename(args.model_name)
    output_dir = f"./outputs/{model}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    model, tokenizer = load_model(args.verifier_model_path, tensor_parallel_size=args.tensor_parallel_size)

    all_data = process_jsonl_file(os.path.join(output_dir,"rollout.jsonl"), model, tokenizer)
    
    # Calculate accuracy by dataset
    if all_data:
        calculate_accuracy_by_dataset(all_data, output_dir)
    
    print("\n" + "=" * 50)
    print("All files processed!")


if __name__ == "__main__":
    main()