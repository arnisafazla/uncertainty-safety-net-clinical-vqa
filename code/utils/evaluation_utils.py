import pandas as pd
import re, ast
from typing import List
from collections import Counter
from tqdm import tqdm

special_tokens = [
        "<|begin_of_text|>",
        "<|eot_id|>"
    ]

def re_search(text):
    """Extract answer using regex. Returns single letter (A-F) or None."""
    # first, remove the special tokens from the text
    for token in special_tokens:
        text = text.replace(token, "")
    text = text.strip()

    if text in ["A", "B", "C", "D", "E", "F"]:
        return text

    # Find the first occurrence of '['
    first_bracket_index = text.find("[")
    if first_bracket_index == -1:
        return None

    # Extract substring from the first '[' onward (up to 5 characters, or more if needed)
    text_from_first_bracket = text[first_bracket_index:first_bracket_index + 5]

    # Check for explicit empty brackets []
    if text_from_first_bracket.strip().startswith("[]"):
        return "[]"

    # Apply regex to capture one or more bracketed uppercase letters
    matches = re.findall(r"\[\s*([A-F])\.?(?:\s+|$|\s*\])", text_from_first_bracket)
    if len(matches) == 1:
        return matches[0]
    elif len(matches) > 1:
        # Multiple matches found - return [multiple] to indicate ambiguity
        return None

    return None

class EvaluateModelOutputs:
    def __init__(self, model_id: str, model_config: dict):
        from scripts.utils.transformers_models import InferenceManager

        # Initialize inference manager
        self.inference_manager = InferenceManager(model_config=model_config)
        self.inference_manager.load_model(model_id)

    def extract_with_llm(self, input_data: List[dict]):
        def format_prompt(item: dict):
            """Format the prompt for LLM extraction."""
            response = item["model_response"]
            question = item["question"]
            # Format options
            options = ast.literal_eval(item["shuffled_options"])
            formatted_options = "\n".join(
                f"[{letter}] {option}"
                for letter, option in options.items()
            )
            prompt = f"""You will be given a multiple-choice question with between 2 and 6 answer options (A-F) and a free-form response.

        Question: {question}
        Options:
        {formatted_options}
        Response: {response}
        Your task is to determine which of the options (A-F) the response refers to. Output only the corresponding letter in square brackets, like [X]. Do not include any explanation or additional text.
        Your annotation must be exactly one of the following: [A], [B], [C], [D], [E], [F] or [] where [] is used if no single valid option is chosen, if multiple options are indicated, or if the response does not clearly map to any option.
        If the response includes a different letter, you must still map it to one of the valid options above, based on the content of the response. Keep in mind that the response may be poorly formatted or contain irrelevant letters —focus only on identifying the most likely intended option.
        Your annotation:"""
            return {"prompt": prompt}

        input_data_formatted = [format_prompt(item) for item in input_data]
        # batches of 8
        batch_size = 8
        model_responses = []
        for i in tqdm(range(0, len(input_data_formatted), batch_size)):
            batch_prompts = input_data_formatted[i:i+batch_size]
            outputs = self.inference_manager.run_inference(batch_prompts)
            model_responses.extend(outputs)

        output_rows = []
        for row, model_response in zip(input_data, model_responses):
            # Use the same regex method to extract from LLM response
            extracted = re_search(model_response["model_response"])
            new_row = row.copy()
            new_row["extracted_model_response"] = extracted
            new_row["extraction_method"] = "llm_extracted"
            output_rows.append(new_row)

        return output_rows

    def save_results(self, output_rows, output_path):
        """Save the results to a TSV file."""
        df_results = pd.DataFrame(output_rows)
        df_results.to_csv(output_path, sep="\t", index=False)
        print(f"Results saved to {output_path}")

    def extract_results_pipeline(self, df_responses, output_path, start_index=-1):
        output_rows = []
        for_llm_extraction = []

        for _, row in df_responses.iterrows():
            if row["sample_idx"] <= start_index:
                continue

            extracted_answer = None
            
            # Handle empty/null responses
            if extracted_answer is None and (pd.isna(row["model_response"]) or row["model_response"] is None or row["model_response"].strip() == ""):
                extracted_answer, method = "[]", "empty_answer"

            # Try regex extraction first
            if extracted_answer is None:
                regex_result = re_search(row["model_response"])
                if regex_result is not None:
                    extracted_answer, method = regex_result, "re_search"
                # If regex failed, try LLM extraction
                else:
                    for_llm_extraction.append(row)
                    continue

            # If everything failed
            if extracted_answer is None:
                extracted_answer, method = "[]", "llm_extracted"

            new_row = row.copy()
            new_row["extracted_model_response"] = extracted_answer
            new_row["extraction_method"] = method
            output_rows.append(new_row)

        # Process LLM extractions in batch
        print(f"Processing {len(for_llm_extraction)} entries with LLM extraction...")
        output_rows.extend(self.extract_with_llm(for_llm_extraction))
        self.save_results(output_rows, output_path)

# ----------------------------- Reverse map the answer options --------------------------

def add_initial_options(df_raw: pd.DataFrame) -> pd.DataFrame:
    option_cols = [col for col in df_raw.columns if col in list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")]

    def _make_dict(row):
        return {col: row[col] for col in option_cols if pd.notna(row[col])}

    out = df_raw.copy()
    out["original_options"] = out.apply(_make_dict, axis=1)
    return out

def map_answers(df_extracted: pd.DataFrame, df_raw: pd.DataFrame, strict: bool = False) -> pd.DataFrame:
    """
    Map predictions and correct answers back to canonical labels
    for the entire df_extracted at once.
    """

    def _map_group(df_subset):
        # Get initial_options for this index
        idx = df_subset["sample_idx"].iloc[0]
        initial_options = df_raw.loc[df_raw["sample_idx"] == idx, "initial_options"].iloc[0]

        # Build reverse map for faster lookup
        rev_map = {v: k for k, v in initial_options.items()}

        def _map_row(row):
            # Ensure shuffled_options is a dict
            opts = row["shuffled_options"]
            if isinstance(opts, str):
                opts = ast.literal_eval(opts)

            # Get texts
            pred_text = opts.get(row["extracted_model_response"])
            corr_text = opts.get(row["shuffled_correct_answer"])

            if strict:
                backup = None
            else:
                backup = "[]"
            return pd.Series({
                'extracted_model_response_mapped': rev_map.get(pred_text, backup),
                'original_correct_answer': rev_map.get(corr_text)
            })

        mapped = df_subset.apply(_map_row, axis=1)
        return pd.concat([df_subset, mapped], axis=1)

    return pd.concat([_map_group(g) for _, g in df_extracted.groupby("sample_idx")])

def final_prediction(group: pd.DataFrame, strict: bool = False):
    """
    Perform majority voting for a group. Select random prediction in case of tie.
    Returns:
      - final_pred (str or None)
      - is_correct (bool or None)
    """
    preds = group['extracted_model_response_mapped'].dropna().tolist()
    if not preds:
        return None, None

    counts = Counter(preds)
    # find if we have a strict majority
    most_common = counts.most_common()
    if len(most_common) > 1 and most_common[0][1] == most_common[1][1]:
        if strict:
            return None, None
        else:
            final_pred = most_common[0][0]
    else:
        final_pred = most_common[0][0]

    # all rows in a group should have the same correct label
    correct_labels = group['original_correct_answer'].dropna().unique()
    correct_label = correct_labels[0] if len(correct_labels) == 1 else None

    is_correct = (final_pred == correct_label) if correct_label else None

    return final_pred, is_correct