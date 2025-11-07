from dataclasses import dataclass
from typing import Any
import ollama
from transformers import AutoTokenizer
from code_gen_exp.src.utils.solver_utils import (
    check_eos,
    parse_python_fence,
    parse_response,
    get_solutions,
    get_unittests,
    get_questions,
    get_dataset,
)

@dataclass
class RewardsOllamaConfig:
    model: str = "qwen2.5-coder:1.5b-instruct"
    temperature: float = 0.0
    num_predict: int = 512


class CodeGenerationRewards:
    def __init__(self, solver_tokenizer_path: str, solver_token_lim: int, ollama_config: RewardsOllamaConfig = RewardsOllamaConfig()):
        self.stage = 0
        self.model = ollama_config.model
        self.temperature = ollama_config.temperature
        self.num_predict = ollama_config.num_predict
        self.tokenizer = AutoTokenizer.from_pretrained(solver_tokenizer_path, padding_side="left")
        self.solver_token_lim = solver_token_lim


    def _build_prompt(self, dataset: str, solution_code: str, unit_tests: str, question: str) -> str:
        if dataset == 'mbpp':
            return ("You are an expert programming evaluator who needs to decide whether the given solution will pass all the given unit tests.\n"
                 "You will be given a problem, a list of unit tests, and a solution.\n"
                 "Walk through each unit test and dry run the solution.\n"
                 "If the solution passes all the unit tests, put is_correct as true.\n"
                 "If the solution fails even a single unit test, put is_correct as false.\n"
                 "Put you final answer in a JSON fenced block. The JSON should have only one key: is_correct. It should be a boolean.\n"
                 "Its format should be as follows:\n"
                 "```json\n{\n  \"is_correct\": true | false\n}\n```\n\n"
                 "--- Problem ---\n"
                 f"{question}\n\n"
                 "--- Unit Tests ---\n"
                 f"{unit_tests}\n\n"
                 "--- Solution ---\n"
                 f"{solution_code}\n\n"
            )
        elif dataset == 'code_contests':
            return ("You are an expert programming evaluator who needs to decide whether the given solution will pass all the given unit tests.\n"              
                "With the code you will be given the inputs for unit tests along with the associated outputs. \n"
                "Decide if the given python code will produce the given outputs when run against the inputs. \n"
                "Put you final answer in a JSON fenced block. The JSON should have only one key: is_correct. It should be a boolean.\n"
                "Its format should be as follows:\n"
                "```json\n{\n  \"is_correct\": true | false\n}\n```\n\n"
                "--- Problem ---\n"
                f"{question}\n\n"
                "--- Unit Tests ---\n"
                f"{unit_tests}\n\n"
                "--- Solution ---\n"
                f"{solution_code}\n\n"
            )

    def _extract_json(self, text: str) -> Any:
        match = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", text, re.IGNORECASE)
        if match:
            return json.loads(match.group(1))
        return json.loads(text)

    def reward_fn(self, dataset, solutions, unittests, question):
        rewards = []
        for solution in solutions:
            if not isinstance(solution, str):
                reward = -1.2
            else:
                parsed_code = parse_python_fence(solution)
                eos_found = check_eos(solution, self.tokenizer, self.solver_token_lim)
                if parsed_code is None: # No fenced code found
                    reward = -1.0
                else:
                    try:
                        prompt = self._build_prompt(dataset, str(solution), str(unittests), str(question))
                        response = ollama.generate(model=self.model, prompt=prompt, options={"temperature": self.temperature, "num_predict": self.num_predict})
                        raw_text = response.response
                        reward = parse_response(raw_text)
                        if reward is None:
                            reward = 0.0
                    except:
                        reward = 0.0
                reward += 0.2 if eos_found else -0.2
            rewards.append(reward)

        return rewards



    def __call__(self, game_state):
        solutions_by_agent = get_solutions(game_state, self.stage)
        unittests_by_agent = get_unittests(game_state, self.stage)
        questions = get_questions(game_state, self.stage)
        datasets_by_agent = get_dataset(game_state, self.stage)
        
        rewards = {}  # Key per agent
        try:
            for agent in solutions_by_agent:
                rewards[agent] = {}  # Will store a list per batch item
                for batch_id in solutions_by_agent[agent]:
                    rewards[agent][batch_id] = []
                    for node_idx, _ in enumerate(solutions_by_agent[agent][batch_id]):
                        rewards[agent][batch_id].append(
                            self.reward_fn(
                                datasets_by_agent[agent][batch_id][node_idx],
                                solutions_by_agent[agent][batch_id][node_idx],
                                unittests_by_agent[agent][batch_id][node_idx],
                                questions[agent][batch_id][node_idx]
                            )
                        )
            return rewards
        except Exception as e:
            return {}
