from genrl.state import GameState
from typing import Any, List
import re
import json

def get_solutions(game_state: GameState, stage: int) -> dict[Any, dict[Any, List[Any]]]:
    actions = game_state.get_stage_actions(stage)
    solutions = {}
    for agent in actions:
        solutions[agent] = {}  
        for batch_id in actions[agent]:
            solutions[agent][batch_id] = []
            for node, _ in enumerate(actions[agent][batch_id]):
                solutions[agent][batch_id].append(actions[agent][batch_id][node])
    return solutions  # Indices are [Agent][Batch Item][Node Idx][Solution]


def get_unittests(game_state: GameState, stage: int) -> dict[Any, dict[Any, List[Any]]]:
    world_states = game_state.get_stage_state(stage)
    unittests = {}  # Key per agent
    for agent in world_states:
        unittests[agent] = {} 
        for batch_id in world_states[agent]:
            unittests[agent][batch_id] = []
            for node, _ in enumerate(world_states[agent][batch_id]):
                unittests[agent][batch_id].append(
                    world_states[agent][batch_id][node].environment_states["test"]
                )
    return unittests  # Indices are [Agent][Batch Item][Node Idx]


def get_questions(game_state: GameState, stage: int) -> dict[Any, dict[Any, List[Any]]]:
    world_states = game_state.get_stage_state(stage)
    questions = {}
    for agent in world_states:
        questions[agent] = {} 
        for batch_id in world_states[agent]:
            questions[agent][batch_id] = []
            for node, _ in enumerate(world_states[agent][batch_id]):
                questions[agent][batch_id].append(
                    world_states[agent][batch_id][node].environment_states["question"]
                )
    return questions  # Indices are [Agent][Batch Item][Node Idx]


def get_dataset(game_state: GameState, stage: int) -> dict[Any, dict[Any, List[Any]]]:
    world_states = game_state.get_stage_state(stage)
    dataset = {}
    for agent in world_states:
        dataset[agent] = {} 
        for batch_id in world_states[agent]:
            dataset[agent][batch_id] = []
            for node, _ in enumerate(world_states[agent][batch_id]):
                dataset[agent][batch_id].append(
                    world_states[agent][batch_id][node].environment_states["metadata"]["dataset"]
                )
    return dataset  # Indices are [Agent][Batch Item][Node Idx]


def parse_response(text: str):
    """
    Extracts a numeric score from various response formats.
    Handles:
    - Fenced JSON (```json ... ```)
    - Plain JSON ({'score': 0.85})
    - Plain JSON with 'is_correct' (boolean) or legacy 'score' (float)
    - Python-style dicts
    - Extra text or explanations
    Returns 1.0 for true/correct, 0.0 for false/incorrect.
    """
    if not text or not isinstance(text, str):
        return None
    cleaned = text.strip() # Remove leading and trailing whitespace
    # 1) Extract JSON inside code fences, if any
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL | re.IGNORECASE)
    if match:
        cleaned = match.group(1).strip()

    # 2) Check if `cleaned` is in valid JSON format; if so, return score
    try:
        data = json.loads(cleaned)
        # Priority: is_correct (boolean) -> score (float)
        if "is_correct" in data:
            return 1.0 if data["is_correct"] else 0.0
        if "score" in data:
            return float(data["score"])
    except Exception:
        pass

    # 3) Convert single quotes to double quotes and retry
    try:
        alt = cleaned.replace("'", '"')
        # Also handle Python True/False -> JSON true/false
        alt = alt.replace("True", "true").replace("False", "false")
        data = json.loads(alt)
        if "is_correct" in data:
            return 1.0 if data["is_correct"] else 0.0
        if "score" in data:
            return float(data["score"])
    except Exception:
        pass

    # 4) Fallback regex to find is_correct or score pattern anywhere
    # Try is_correct first
    match = re.search(r"['\"]?is_correct['\"]?\s*[:=]\s*(true|false)", cleaned, re.IGNORECASE)
    if match:
        return 1.0 if match.group(1).lower() == "true" else 0.0
    # Try legacy score pattern
    match = re.search(r"['\"]?score['\"]?\s*[:=]\s*(-?\d+(?:\.\d+)?)", cleaned)
    if match:
        return float(match.group(1))

    # 5) If still not found return None
    return None


def parse_python_fence(text):
    """
    Parses a PYTHON fence block from a given text string.
    Args:
        text (str): The input text.
    Returns:
        string or None: The parsed Python string, or None if no valid Python block is found.
    """
    match = re.search(r'```python(.*?)```', text, re.DOTALL)
    if match:
        python_string = match.group(1).strip()
        return python_string
    return None


def check_eos(solution, tokenizer, max_new_tokens):
    """
    Tokenizes solution and checks if the length is < max_new_tokens (for marking whether eos is found).
    Args:
        solution (str): The input text.
        tokenizer (transformers.AutoTokenizer): Tokenizer.
        max_new_tokens (int): Number of (new) tokens allowed for generation.
    Returns:
        bool: Whether the solution contains less number of tokens than max_new_tokens
    """
    completion_ids = tokenizer(solution, return_tensors="pt")['input_ids']
    if len(completion_ids[0]) < max_new_tokens: # Tokenizer returns batched outputs, where the batch size is 1 because solution is a string 
        return True
    else:
        return False 