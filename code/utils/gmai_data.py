from typing import List, Dict, Any, Tuple, Generator, Optional
import random
import string, ast
import pandas as pd
from math import factorial

random.seed(42)  # For reproducibility

# This script processes GMAI-MMBench data, shuffles options, and prepares prompts for model inference.
# GMAI-MMBench specific functions: shuffle options given either 4/5 options
# load images in string format and process (?)

QUESTION_PROMPTS = {
    "RAW": """Question: {question}
Options:
{options}
""",
    "GMAI": """Question: {question}
Options:
{options}
Please select the correct answer from the options above.
""",
    "GMAI_with_example": """Provide a concise answer to the given question. Start your response with the letter of the correct option in square brackets. If needed, briefly explain why.
Example: [B] since the option matches the symptoms described.

Question: {question}
Options:
{options}
Your answer:
""",
    "default": """Question: ({question})
Provide a concise answer and start your response with the letter of the selected option in square brackets. Options:
(
{options}
)
Your Answer:
""",
    "describe_image": "Describe what you see in the image."
}

def generate_unique_shuffles(options: Dict[str, str], 
        correct_answer: str, k: int) -> List[Tuple[Dict[str, str], str, str]]:
    """
    Generate k unique shuffles of options.
    
    Args:
        options: Dictionary mapping letters to option texts (e.g., {"A": "option 1", "B": "option 2"}).
        correct_answer: The correct answer letter (e.g., "B").
        k: Number of unique shuffles to generate.
    
    Returns:
        List of tuples (shuffled_options_dict, new_correct_letter).
    """
    if k == 0:
        return [(options, correct_answer, "no replace")]
    
    # Extract texts
    option_texts = list(options.values())

    # Compute max unique permutations
    max_permutations = factorial(len(option_texts))
    k = min(k, max_permutations)

    shuffles = set()
    attempts = 0
    max_attempts = k * 20  # safety cap

    while len(shuffles) < k and attempts < max_attempts:
        attempts += 1

        shuffled_texts = option_texts.copy()
        random.shuffle(shuffled_texts)

        # Map back to letters
        shuffled_options = {string.ascii_uppercase[i]: text for i, text in enumerate(shuffled_texts)}

        # Find new correct answer letter
        correct_text = options[correct_answer]
        new_correct_letter = next(letter for letter, text in shuffled_options.items() if text == correct_text)

        # Use tuple of texts in order as uniqueness key
        shuffle_key = tuple(shuffled_texts)
        shuffles.add((shuffle_key, new_correct_letter))

    # Build final result
    result = []
    for shuffle_key, correct_letter in shuffles:
        shuffled_dict = {string.ascii_uppercase[i]: text for i, text in enumerate(shuffle_key)}
        result.append((shuffled_dict, correct_letter, "default"))

    return result

def generate_unique_shuffles_replace_option(
    options: Dict[str, str],
    correct_answer: str,
    k: int,
    new_option: str = "none of the options are correct"
) -> List[Tuple[Dict[str, str], str, str]]:
    """
    Generate k unique shuffles of options, replacing one option with `new_option` before each shuffle.

    Args:
        options: Dictionary mapping letters to option texts (e.g., {"A": "option 1", "B": "option 2"}).
        correct_answer: The correct answer letter (e.g., "B").
        k: Number of unique shuffles to generate.
        new_option: Before each shuffle, replace one of the options with this new option.
            If num_options > k, replace each option once (cycling through replacements).

    Returns:
        List of tuples: (shuffled_options_dict, new_correct_letter, label)
            where label is:
              - "none" if the correct answer was replaced
              - "confusing" if the correct answer was not replaced
    """
    letters = list(options.keys())
    num_options = len(options)

    # Special case: if k == 0 → replace only the correct answer
    if k == 0:
        modified_options = options.copy()
        modified_options[correct_answer] = new_option
        return [(modified_options, correct_answer, "none")]

    # Compute how many unique shuffles are possible
    max_permutations = factorial(num_options)
    k = min(k, max_permutations)

    results = []
    replaced_indices = list(range(num_options))
    random.shuffle(replaced_indices)

    for i in range(k):
        modified_options = options.copy()

        # Choose which option to replace
        if num_options > k:
            replace_idx = replaced_indices[i % num_options]
        else:
            replace_idx = i % num_options

        letter_to_replace = letters[replace_idx]
        correct_text = options[correct_answer]

        # Replace with new option
        modified_options[letter_to_replace] = new_option

        # Determine label and correct letter
        if letter_to_replace == correct_answer:
            label = "none"
        else:
            label = "confusing"
        new_correct_letter = correct_answer

        # Shuffle
        shuffled_items = list(modified_options.items())
        random.shuffle(shuffled_items)
        shuffled_dict = {string.ascii_uppercase[i]: text for i, (old_letter, text) in enumerate(shuffled_items)}

        # Find new correct letter
        correct_text = options[correct_answer]
        new_correct_letter = next((ltr for ltr, txt in shuffled_dict.items() if txt == correct_text), None)

        if new_correct_letter is None:
            new_correct_letter = next((ltr for ltr, txt in shuffled_dict.items() if txt == new_option), None)

        results.append((shuffled_dict, new_correct_letter, label))

    return results


def construct_prompt(question: str, options: Dict[str, str], prompt_template: str) -> str:
    """
    Construct the prompt with question and options.
    
    Args:
        question: The question text
        options: List of option texts
    
    Returns:
        Formatted prompt string
    """
    formatted_options = "\n".join(
        f"[{letter}] {option}"
        for letter, option in options.items()
    )
    
    return QUESTION_PROMPTS[prompt_template].format(
        question=question,
        options=formatted_options
    )

def format_and_upsample_data(data: List[Dict[str, Any]], sample_s: int = 1, prompt_template: str = "default") -> List[Dict[str, Any]]:
    """
    Generator that yields processed data items with shuffled options.
    
    Args:
        data: Original data dictionary
        k: Number of unique shuffles per item
        max_items: Maximum number of items to process
    
    Outputs:
        List of Dictionary containing processed item data
    """
    total_items = 0
    processed_items = []
    
    for item in data:
        question = item["question"]
        options = ast.literal_eval(item["shuffled_options"])

        # Construct the prompt
        prompt = construct_prompt(question, options, prompt_template)
        item["prompt"] = prompt
        sampled_items = []
        for i in range(sample_s):
            tmp_item = item.copy()
            tmp_item["generation_idx"] = i
            sampled_items.append(tmp_item)
        processed_items.extend(sampled_items)
        total_items += 1

    return processed_items

def batch_generator(processed_data: List[Dict[str, Any]], batch_size: int) -> Generator[List[Dict[str, Any]], None, None]:
    """
    Generator that yields batches of processed items.
    
    Args:
        data_gen: Data generator
        batch_size: Size of each batch
    
    Yields:
        List of batch items
    """
    batch = []
    
    for item in processed_data:
        batch.append(item)
        
        if len(batch) >= batch_size:
            yield batch
            batch = []
    
    # Yield remaining items
    if batch:
        yield batch

