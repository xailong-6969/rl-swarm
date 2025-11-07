import logging
import re
import json

logger = logging.getLogger(__name__)

def parse_json_from_fence(text):
    """
    Parses a JSON fence block from a given text string.
    Args:
        text (str): The input text containing a JSON fence block.
    Returns:
        dict or list or None: The parsed JSON object, or None if no valid JSON block is found.
    """
    # Regex to find a block starting with ```json and ending with ```
    # The `?` makes the match non-greedy, so it stops at the first closing fence.
    # The `re.DOTALL` flag allows the `.` to match newlines.
    match = re.search(r'```json(.*?)```', text, re.DOTALL)
    if match:
        json_string = match.group(1).strip()
        try:
            # Use json.loads to parse the cleaned string
            parsed_json = json.loads(json_string)
            return parsed_json
        except json.JSONDecodeError as e:
            logger.info(f"Unable to decode JSON: {e}")
            return None
    else:
        logger.info(f'proposal cannot be parsed from fence')
    return None

def extract_question_name(question: str):
    """
    Extracts the function name from prompts like:
    - Write a function is_even(n)
    - Write a function 'is_even'
    - Write a function `is_even(n)`
    - Write a function "is_even"
    """
    question_pattern = re.compile(
        r'^Write a function\s+'
        r'(?P<quote>[`"\'])?'                           # optional quote: ', ", or `
        r'(?P<name>[A-Za-z_][A-Za-z0-9_]*)'            # function name
        r'(?:\s*\(\s*'                                  # optional opening parenthesis
        r'(?P<params>[A-Za-z_][A-Za-z0-9_]*(?:\s*,\s*[A-Za-z_][A-Za-z0-9_]*)*)?'  # optional params
        r'\s*\))?'                                      # optional closing parenthesis
        r'(?P=quote)?'                                  # optional matching closing quote
    )
    try:
        match = question_pattern.match(question)
    except:
        logger.info(f"Failed to extract question name from question: {question}")
        return None
    if match:
        func_name = match.group('name')
        return func_name
    logger.info(f"Failed to extract question name from question: {question}")
    return None
