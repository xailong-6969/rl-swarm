
from dataclasses import dataclass, field
from collections import deque
import copy
import logging
import os
import shutil
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer

from code_gen_exp.src.utils.proposer_utils import parse_json_from_fence, extract_question_name
from genrl.misc_utils.sandbox_executor import CodeSandboxExecutor
logger = logging.getLogger(__name__)


try:
    from vllm import LLM as VLLMEngine
    from vllm import SamplingParams as VLLMSamplingParams
    _VLLM_AVAILABLE = True
except Exception:
    VLLMEngine = None
    VLLMSamplingParams = None
    _VLLM_AVAILABLE = False

SYSTEM_PROMPT = "You are an expert Python developer and technical interviewer specializing in creating high-quality coding challenges. Your role is to design Python programming problems with comprehensive test cases that accurately assess a developer's coding abilities, problem-solving skills, and understanding of Python best practices."
# Level 1: Beginner - Basic Python concepts
LEVEL_1_PROMPT_TEXT = (
            "You are to propose a BEGINNER Python coding question and matching unit tests.\n"
            "Constraints:\n"
            "- Do NOT provide any solution or implementation.\n"
            "- Choose a single clear function name and write a one-sentence description of what it should do.\n"
            "- Provide runnable unit tests only (pytest or unittest) that validate expected behavior of the function.\n"
            "- The tests should be self-contained and reference the function name but not implement it.\n"
            "- Limit the number of included tests to at most 4.\n"
            "- The output MUST be valid JSON with exactly these keys: question (string) and tests (string).\n"
            "- The tests field must contain ONLY Python code.\n"
            "- Do not include any additional keys or commentary.\n"
            "- The question should contain the function name that the tests refer to.\n"
            "- IMPORTANT: Wrap your entire output in a Markdown fenced code block that starts with ```json on its own line and ends with ``` on its own line.\n"
            "- Do not include any text outside the fenced block.\n\n"
            "DIFFICULTY REQUIREMENTS (Level 1 - Beginner):\n"
            "- Focus on basic Python concepts: variables, simple data types, basic loops, conditionals\n"
            "- Simple arithmetic operations, string manipulation, basic list operations\n"
            "- Single parameter functions with straightforward logic\n"
            "- No complex algorithms or data structures required\n"
            "- Examples: string formatting, basic calculations, simple list processing\n\n"
            "Example output format (fenced JSON):\n"
            "```json\n{\n  \"question\": \"Write a function foo(x) that ...\",\n  \"tests\": \"import pytest\\n\\n def test_foo(): ...\"\n}\n```"
            "The following are problems you have already proposed along with the number of times they have been proposed, please do not repeat:\n\n"
        )

# Level 2: Easy - Basic algorithms and data structures
LEVEL_2_PROMPT_TEXT = (
            "You are to propose an EASY Python coding question and matching unit tests.\n"
            "Constraints:\n"
            "- Do NOT provide any solution or implementation.\n"
            "- Choose a single clear function name and write a one-sentence description of what it should do.\n"
            "- Provide runnable unit tests only (pytest or unittest) that validate expected behavior of the function.\n"
            "- The tests should be self-contained and reference the function name but not implement it.\n"
            "- Limit the number of included tests to at most 4.\n"
            "- The output MUST be valid JSON with exactly these keys: question (string) and tests (string).\n"
            "- The tests field must contain ONLY Python code.\n"
            "- Do not include any additional keys or commentary.\n"
            "- The question should contain the function name that the tests refer to.\n"
            "- IMPORTANT: Wrap your entire output in a Markdown fenced code block that starts with ```json on its own line and ends with ``` on its own line.\n"
            "- Do not include any text outside the fenced block.\n\n"
            "DIFFICULTY REQUIREMENTS (Level 2 - Easy):\n"
            "- Basic algorithms: linear search, simple sorting, basic string/list operations\n"
            "- Simple data structures: lists, dictionaries, sets\n"
            "- Basic iteration and simple recursive thinking\n"
            "- Multiple parameters but straightforward logic\n"
            "- Examples: finding max/min, counting occurrences, simple transformations\n\n"
            "Example output format (fenced JSON):\n"
            "```json\n{\n  \"question\": \"Write a function foo(x) that ...\",\n  \"tests\": \"import pytest\\n\\n def test_foo(): ...\"\n}\n```"
            "The following are problems you have already proposed along with the number of times they have been proposed, please do not repeat:\n\n"
        )

# Level 3: Medium - Intermediate algorithms and problem solving
LEVEL_3_PROMPT_TEXT = (
            "You are to propose a MEDIUM Python coding question and matching unit tests.\n"
            "Constraints:\n"
            "- Do NOT provide any solution or implementation.\n"
            "- Choose a single clear function name and write a one-sentence description of what it should do.\n"
            "- Provide runnable unit tests only (pytest or unittest) that validate expected behavior of the function.\n"
            "- The tests should be self-contained and reference the function name but not implement it.\n"
            "- Limit the number of included tests to at most 4.\n"
            "- The output MUST be valid JSON with exactly these keys: question (string) and tests (string).\n"
            "- The tests field must contain ONLY Python code.\n"
            "- Do not include any additional keys or commentary.\n"
            "- The question should contain the function name that the tests refer to.\n"
            "- IMPORTANT: Wrap your entire output in a Markdown fenced code block that starts with ```json on its own line and ends with ``` on its own line.\n"
            "- Do not include any text outside the fenced block.\n\n"
            "DIFFICULTY REQUIREMENTS (Level 3 - Medium):\n"
            "- Intermediate algorithms: binary search, merge sort, basic graph traversal\n"
            "- More complex data structures: trees, heaps, stacks, queues\n"
            "- Moderate algorithmic thinking and optimization\n"
            "- Multiple parameters with some complexity\n"
            "- Examples: tree operations, matrix manipulation, moderate optimization problems\n\n"
            "Example output format (fenced JSON):\n"
            "```json\n{\n  \"question\": \"Write a function foo(x) that ...\",\n  \"tests\": \"import pytest\\n\\n def test_foo(): ...\"\n}\n```"
            "The following are problems you have already proposed along with the number of times they have been proposed, please do not repeat:\n\n"
        )

# Level 4: Hard - Advanced algorithms and complex problem solving
LEVEL_4_PROMPT_TEXT = (
            "You are to propose a HARD Python coding question and matching unit tests.\n"
            "Constraints:\n"
            "- Do NOT provide any solution or implementation.\n"
            "- Choose a single clear function name and write a one-sentence description of what it should do.\n"
            "- Provide runnable unit tests only (pytest or unittest) that validate expected behavior of the function.\n"
            "- The tests should be self-contained and reference the function name but not implement it.\n"
            "- Limit the number of included tests to at most 4.\n"
            "- The output MUST be valid JSON with exactly these keys: question (string) and tests (string).\n"
            "- The tests field must contain ONLY Python code.\n"
            "- Do not include any additional keys or commentary.\n"
            "- The question should contain the function name that the tests refer to.\n"
            "- IMPORTANT: Wrap your entire output in a Markdown fenced code block that starts with ```json on its own line and ends with ``` on its own line.\n"
            "- Do not include any text outside the fenced block.\n\n"
            "DIFFICULTY REQUIREMENTS (Level 4 - Hard):\n"
            "- Advanced algorithms: dynamic programming, complex graph algorithms, advanced sorting\n"
            "- Complex data structures: advanced trees, graphs, hash tables with collision handling\n"
            "- Sophisticated algorithmic thinking and optimization\n"
            "- Multiple parameters with complex interactions\n"
            "- Examples: pathfinding algorithms, complex optimization, advanced data manipulation\n\n"
            "Example output format (fenced JSON):\n"
            "```json\n{\n  \"question\": \"Write a function foo(x) that ...\",\n  \"tests\": \"import pytest\\n\\n def test_foo(): ...\"\n}\n```"
            "The following are problems you have already proposed along with the number of times they have been proposed, please do not repeat:\n\n"
        )

# Level 5: Expert - Very challenging problems requiring deep algorithmic knowledge
LEVEL_5_PROMPT_TEXT = (
            "You are to propose an EXPERT Python coding question and matching unit tests.\n"
            "Constraints:\n"
            "- Do NOT provide any solution or implementation.\n"
            "- Choose a single clear function name and write a one-sentence description of what it should do.\n"
            "- Provide runnable unit tests only (pytest or unittest) that validate expected behavior of the function.\n"
            "- The tests should be self-contained and reference the function name but not implement it.\n"
            "- Limit the number of included tests to at most 4.\n"
            "- The output MUST be valid JSON with exactly these keys: question (string) and tests (string).\n"
            "- The tests field must contain ONLY Python code.\n"
            "- Do not include any additional keys or commentary.\n"
            "- The question should contain the function name that the tests refer to.\n"
            "- IMPORTANT: Wrap your entire output in a Markdown fenced code block that starts with ```json on its own line and ends with ``` on its own line.\n"
            "- Do not include any text outside the fenced block.\n\n"
            "DIFFICULTY REQUIREMENTS (Level 5 - Expert):\n"
            "- Expert-level algorithms: advanced dynamic programming, complex graph theory, advanced optimization\n"
            "- Sophisticated data structures: advanced trees, complex graph representations, specialized structures\n"
            "- Deep algorithmic thinking, multiple algorithm combinations, edge case handling\n"
            "- Complex parameter interactions and advanced problem decomposition\n"
            "- Examples: advanced pathfinding, complex optimization problems, specialized algorithms\n\n"
            "Example output format (fenced JSON):\n"
            "```json\n{\n  \"question\": \"Write a function foo(x) that ...\",\n  \"tests\": \"import pytest\\n\\n def test_foo(): ...\"\n}\n```"
            "The following are problems you have already proposed along with the number of times they have been proposed, please do not repeat:\n\n"
        )

# Backward compatibility - keep original prompts as aliases
PROMPT_TEXT = LEVEL_2_PROMPT_TEXT  # Default to Level 2 (Easy)
HARD_PROMPT_TEXT = LEVEL_4_PROMPT_TEXT  # Hard maps to Level 4


CLEANUP_PROMPT = "please reformat the following to ensure that it is valid json in a fenced code block with keys question and tests.  All other output should be removed.\n"

@dataclass
class PPOConfig:
    clip_range: float = 0.2
    ppo_epochs: int = 2
    learning_rate: float = 1e-5
    beta: float = 0.01
    lookback: int = 5

@dataclass
class VllmConfig:
    use_vllm: bool = False
    engine: VLLMEngine = None
    max_tokens: int = 1024
    sampling: VLLMSamplingParams | None = field(default=None)
    # vLLM resource controls (tunable to mitigate OOM)
    gpu_memory_utilization: float = 0.9  # fraction of available GPU memory vLLM may use
    max_model_len: int = 4096          # reduce KV cache footprint if needed
    swap_space: int = 4                # GB of CPU swap for KV cache spillover
    tensor_parallel_size: int = 1      # keep to 1 unless you shard across GPUs
    pipeline_parallel_size: int = 1

    def __post_init__(self):
        if _VLLM_AVAILABLE and self.sampling is None:
            self.sampling = VLLMSamplingParams(max_tokens=self.max_tokens)

@dataclass
class PromptUpdateConfig:
    """Configuration for prompt difficulty updates and adaptive learning."""
    reward_history_size: int = 20  # Number of recent rewards to track for difficulty adjustment
    initial_difficulty_level: int = 2  # Starting difficulty level (1-5)
    recent_window_size: int = 10  # Number of recent iterations to compare
    earlier_window_size: int = 10  # Number of earlier iterations to compare against
    difficulty_change_threshold: float = 0.05  # Minimum performance difference to trigger difficulty change
    prompt_update_frequency: int = 5  # Update prompt difficulty every N iterations


class Proposer:
    def __init__(self, 
                model_path: str, 
                ppo_config: PPOConfig = PPOConfig(), 
                vllm_config: VllmConfig = VllmConfig(), 
                second_pass: bool = True,
                prompt_update_config: PromptUpdateConfig = None):

        # Use default PromptUpdateConfig if none provided
        if prompt_update_config is None:
            prompt_update_config = PromptUpdateConfig()
        
        self.second_pass = second_pass
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.train_dtype = torch.bfloat16 if self.device == 'cuda' else torch.float32
        self.sandbox = CodeSandboxExecutor()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        # PPO config and optimizer
        self.ppo_config = ppo_config

        # vLLM related attributes
        self.vllm_config = vllm_config
        self._vllm_available = _VLLM_AVAILABLE and self.vllm_config.use_vllm
        logger.info(f"Using Vllm for inference: {self._vllm_available}")
        self._vllm_engine = None

        # Track current model path (HF id or local dir after training)
        self._current_model_path = model_path

        # Prompt update configuration
        self.prompt_update_config = prompt_update_config

        # Difficulty level system (1-5)
        self.current_difficulty_level = prompt_update_config.initial_difficulty_level
        self.difficulty_prompts = {
            1: LEVEL_1_PROMPT_TEXT,
            2: LEVEL_2_PROMPT_TEXT,
            3: LEVEL_3_PROMPT_TEXT,
            4: LEVEL_4_PROMPT_TEXT,
            5: LEVEL_5_PROMPT_TEXT
        }

        self.prompts = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": self.difficulty_prompts[self.current_difficulty_level]},
        ]

        self.previous_problems = deque(maxlen=self.ppo_config.lookback)
        self.previous_func_names = {}

        # Adaptive difficulty parameters
        self.reward_history = deque(maxlen=self.prompt_update_config.reward_history_size)

        if not self._vllm_available:
            self.model = AutoModelForCausalLM.from_pretrained(model_path)
            self.model.to(self.device, dtype=self.train_dtype)
            self.optimizer = AdamW(self.model.parameters(), lr=self.ppo_config.learning_rate)

    def _ensure_vllm_engine(self):
        """Create vLLM engine if available and not already initialized."""
        if not self._vllm_available:
            return
        if self._vllm_engine is None:
            # Instantiate vLLM engine with the current model path
            self._vllm_engine = VLLMEngine(
                model=self._current_model_path,
                dtype='auto',
                gpu_memory_utilization=self.vllm_config.gpu_memory_utilization,
                max_model_len=self.vllm_config.max_model_len,
                swap_space=self.vllm_config.swap_space,
                tensor_parallel_size=self.vllm_config.tensor_parallel_size,
            )

    def _shutdown_vllm_engine(self):
        """Tear down vLLM engine to free GPU memory before training."""
        if self._vllm_engine is not None:
            try:
                # Best-effort teardown; vLLM frees on GC
                del self._vllm_engine
            finally:
                self._vllm_engine = None
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

    def update_prompt_difficulty(self, rewards: list[float]):
        """
        Update the prompt based on recent vs earlier performance comparison.
        If recent performance is significantly better than earlier performance, increase difficulty.
        If recent performance is significantly worse than earlier performance, decrease difficulty.
        """
        # Add new rewards to history
        self.reward_history.extend(rewards)
        
        # Need enough data for both recent and earlier windows
        min_required_samples = self.prompt_update_config.recent_window_size + self.prompt_update_config.earlier_window_size
        if len(self.reward_history) < min_required_samples:
            logger.info(f"Not enough reward history ({len(self.reward_history)} samples, need {min_required_samples}), keeping current difficulty level {self.current_difficulty_level}")
            return
        
        # Calculate recent and earlier performance
        recent_performance = self._calculate_recent_performance()
        earlier_performance = self._calculate_earlier_performance()
        
        logger.info(f"Recent performance (last {self.prompt_update_config.recent_window_size}): {recent_performance:.3f}")
        logger.info(f"Earlier performance (previous {self.prompt_update_config.earlier_window_size}): {earlier_performance:.3f}")
        
        # Determine if we should change difficulty
        performance_diff = recent_performance - earlier_performance
        new_difficulty_level = self._determine_difficulty_change(performance_diff)
        
        # Update difficulty level if it changed
        if new_difficulty_level != self.current_difficulty_level:
            old_level = self.current_difficulty_level
            self.current_difficulty_level = new_difficulty_level
            self.prompts[1]["content"] = self.difficulty_prompts[self.current_difficulty_level]
            
            difficulty_names = {1: "Beginner", 2: "Easy", 3: "Medium", 4: "Hard", 5: "Expert"}
            direction = "increased" if new_difficulty_level > old_level else "decreased"
            logger.info(f"Difficulty level {direction} from {old_level} ({difficulty_names[old_level]}) to {new_difficulty_level} ({difficulty_names[new_difficulty_level]}) based on performance diff: {performance_diff:.3f}")
        else:
            difficulty_names = {1: "Beginner", 2: "Easy", 3: "Medium", 4: "Hard", 5: "Expert"}
            logger.info(f"Maintaining current difficulty level {self.current_difficulty_level} ({difficulty_names[self.current_difficulty_level]}) with performance diff: {performance_diff:.3f}")

    def _calculate_recent_performance(self) -> float:
        """Calculate average performance over the most recent window."""
        recent_rewards = list(self.reward_history)[-self.prompt_update_config.recent_window_size:]
        return sum(recent_rewards) / len(recent_rewards)

    def _calculate_earlier_performance(self) -> float:
        """Calculate average performance over the earlier window (before recent window)."""
        start_idx = -(self.prompt_update_config.recent_window_size + self.prompt_update_config.earlier_window_size)
        end_idx = -self.prompt_update_config.recent_window_size
        earlier_rewards = list(self.reward_history)[start_idx:end_idx]
        return sum(earlier_rewards) / len(earlier_rewards)

    def _determine_difficulty_change(self, performance_diff: float) -> int:
        """
        Determine new difficulty level based on performance difference.
        Returns the new difficulty level (1-5).
        """
        current_level = self.current_difficulty_level
        
        # If performance improved significantly, increase difficulty
        if performance_diff > self.prompt_update_config.difficulty_change_threshold:
            return min(current_level + 1, 5)  # Cap at level 5
        
        # If performance declined significantly, decrease difficulty
        elif performance_diff < -self.prompt_update_config.difficulty_change_threshold:
            return max(current_level - 1, 1)  # Cap at level 1
        
        # If performance change is within threshold, maintain current level
        else:
            return current_level

    def checkpoint_model(self, save_dir: str = "./proposer_ckpt"):
        # Save to disk
        if os.path.isdir(save_dir):
            shutil.rmtree(save_dir)
        os.makedirs(save_dir, exist_ok=True)
        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
        self._current_model_path = save_dir

    def _reload_vllm_engine_from_hf(self):
        """
        Save HF model/tokenizer to a directory and recreate vLLM engine from it.
        """
        # Recreate engine
        if not self._vllm_available:
            return
        self._shutdown_vllm_engine()
        self._ensure_vllm_engine()
 
    def _process_proposal(self, proposal_raw: str):
        proposal = parse_json_from_fence(proposal_raw)
        if proposal and 'question' in proposal and 'tests' in proposal:
            result, success = self.sandbox.execute_with_validation(str(proposal["tests"])) # should run through interpreter without errors
            if not success:
                proposal = None
        else:
            logger.info(f'proposal cannot be parsed from fence')
        return proposal

    def generate_proposal(self):
        proposal = None
        proposal_raw = None

        prompt = copy.deepcopy(self.prompts)
        previous_func_names_with_only_repeated_questions = sorted([func_name for func_name, count in self.previous_func_names.items() if count > 1], key=lambda x: x[1], reverse=True)
        previous_func_names_with_only_repeated_questions = previous_func_names_with_only_repeated_questions[:min(len(previous_func_names_with_only_repeated_questions), 20)]
        prompt[1]['content'] += '\n'.join(str(previous_func_names_with_only_repeated_questions)) + '\n'
        if self._vllm_available:
            self._ensure_vllm_engine()
            prompt_text = self.tokenizer.apply_chat_template(
                prompt,
                tokenize=False,
                add_generation_prompt=True,
                return_tensors=None,
            )
            while proposal is None:
                outputs = self._vllm_engine.generate([prompt_text], self.vllm_config.sampling)
                proposal_raw = outputs[0].outputs[0].text
                if self.second_pass:
                    cleanup_prompt = [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": CLEANUP_PROMPT + proposal_raw}
                        ]
                    cleanup_prompt_text = self.tokenizer.apply_chat_template(
                        cleanup_prompt,
                        tokenize=False,
                        add_generation_prompt=True,
                        return_tensors=None,
                        enable_thinking=False
                    )
                    outputs = self._vllm_engine.generate([cleanup_prompt_text], self.vllm_config.sampling)
                    proposal_raw = outputs[0].outputs[0].text
                proposal = self._process_proposal(proposal_raw)

        else:
            # Fallback to HF generate if vLLM not available
            input_ids = self.tokenizer.apply_chat_template(
                prompt,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )
            prompt_length = input_ids.size(1)
            input_ids = input_ids.to(self.model.device)
            while proposal is None:
                response = self.model.generate(
                    input_ids,
                    max_new_tokens=4096,
                    do_sample=True,
                    top_k=300,
                    top_p=0.9,
                    temperature=1
                )
                generated_ids = response[0][prompt_length:]
                proposal_raw = self.tokenizer.decode(
                    generated_ids, skip_special_tokens=True,
                )
                if self.second_pass:
                    cleanup_prompt = [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": CLEANUP_PROMPT + proposal_raw}
                        ]
                    cleanup_input_ids = self.tokenizer.apply_chat_template(
                        cleanup_prompt,
                        tokenize=True,
                        add_generation_prompt=True,
                        return_tensors="pt",
                    )
                    cleanup_prompt_length = cleanup_input_ids.size(1)
                    cleanup_input_ids = cleanup_input_ids.to(self.model.device)
                    cleanup_response = self.model.generate(
                        cleanup_input_ids,
                        max_new_tokens=4096,
                        do_sample=True,
                        top_k=300,
                        top_p=0.9,
                        temperature=1
                    )
                    cleanup_generated_ids = cleanup_response[0][cleanup_prompt_length:]
                    proposal_raw = self.tokenizer.decode(
                        cleanup_generated_ids, skip_special_tokens=True,
                    )
                proposal = self._process_proposal(proposal_raw)

        # Add this check to ensure proposal is not None
        if proposal is None or 'question' not in proposal or 'tests' not in proposal:
            logger.error("Failed to generate a valid proposal after multiple attempts")
            return None

        self.previous_problems.append(proposal['question'])

        func_name = extract_question_name(proposal['question'])
        if func_name is not None:
            if func_name in self.previous_func_names:
                self.previous_func_names[func_name] += 1
            else:
                self.previous_func_names[func_name] = 1
        logger.info(f"Previously asked questions: {self.previous_func_names}")
        
        proposal["proposal_raw"] = proposal_raw
        return proposal

    def reward_fn(self, rewards: list[float]) -> float:            
        if len(rewards) == 0 or rewards is None:
            return None
        elif len(rewards) == 1:
            return rewards[0]
        else:
            avg_reward = sum(rewards) / len(rewards)
            if avg_reward in [0.0, 1.0]:
                return 0.0
            else:
                return 1 - avg_reward

    def _logprob_sum_for_generated(
        self,
        input_ids: torch.Tensor,
        generated_ids: torch.Tensor,
        gen_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute log-probabilities for the generated token sequence under the model.
        input_ids: [B, T_in] or [1, T_in]
        generated_ids: [B, T_gen] or [1, T_gen]
        gen_attention_mask: optional [B, T_gen] (1 for real tokens, 0 for pad)
        Returns: [B, T_gen, V] log-probs for each generated token position.
        """
        if generated_ids.numel() == 0:
            # Return an empty logprob tensor with correct batch dims
            batch = input_ids.shape[0]
            return torch.zeros((batch, 0, self.model.config.vocab_size), device=self.device, dtype=self.train_dtype)

        batch = input_ids.shape[0]

        prompt_attention = torch.ones_like(input_ids, device=input_ids.device)
        if gen_attention_mask is None:
            gen_attention_mask = torch.ones_like(generated_ids, device=input_ids.device)

        context = torch.cat([input_ids, generated_ids], dim=1)  # [B, T_in + T_gen]
        attention_mask = torch.cat([prompt_attention, gen_attention_mask], dim=1)  # [B, T_in + T_gen]
        outputs = self.model(context, attention_mask=attention_mask)
        logits = outputs.logits[:, -generated_ids.shape[1]:, :]  # [B, T_gen, V]
        logprobs = F.log_softmax(logits, dim=-1)
        return logprobs  # [B, T_gen, V]

    def train(self, rewards, proposal):
        """
        Perform a PPO update using the proposed trajectory and the provided reward(s).
        Padding and masking are applied for batched proposals.
        """
        if proposal is None:
            logger.info("No proposal provided to train on.")
            return
        logger.info(f'proposals: {len(proposal)}')
        # Before training, shut down vLLM to free memory; we'll reload after.
        self._shutdown_vllm_engine()

        # Tokenize the chat prompt and the generated completion(s)
        self.model.eval()
        with torch.no_grad():
            base_input_ids = self.tokenizer.apply_chat_template(
                self.prompts,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            ).to(self.device)

            gen_batch = self.tokenizer(
                proposal,
                add_special_tokens=False,
                return_tensors="pt",
                padding=True,
            )
            gen_ids = gen_batch["input_ids"].to(self.device)  # [B, T_gen]
            batch_size = gen_ids.size(0)
            input_ids = base_input_ids.repeat(batch_size, 1)  # [B, T_in]

            gen_attn = gen_batch.get("attention_mask", torch.ones_like(gen_ids)).to(self.device)
            old_logprob = self._logprob_sum_for_generated(input_ids, gen_ids, gen_attention_mask=gen_attn).detach().to(self.device)
            selected_old_logprobs = old_logprob.gather(dim=-1, index=gen_ids.unsqueeze(-1)).squeeze(-1)  # [B, T_gen]

        # rewards: list[list[float]] -> compute per-sample scalar
        adv_list = []
        for rlist in rewards:
            rs = self.reward_fn(rlist)
            adv_list.append(0.0 if rs is None else rs)
        advantage = torch.tensor(adv_list, dtype=self.train_dtype, device=self.device)  # [B]
        advantage -= advantage.mean()
        
        mask = gen_attn.float()  # [B, T_gen]

        valid_counts = mask.sum(dim=1).clamp_min(1.0)  # [B]

        self.model.train()
        for _ in range(self.ppo_config.ppo_epochs):
            self.optimizer.zero_grad()
            # Recompute current logprob under updated policy
            new_logprob = self._logprob_sum_for_generated(input_ids, gen_ids, gen_attention_mask=gen_attn)
            selected_new_logprobs = new_logprob.gather(dim=-1, index=gen_ids.unsqueeze(-1)).squeeze(-1)  # [B, T_gen]

            # PPO ratio per token
            kl_per_token = (selected_new_logprobs - selected_old_logprobs)  # [B, T_gen]
            ratio = torch.exp(kl_per_token)  # [B, T_gen]

            # Clipped objective per token, scale by per-sample advantage
            adv_expanded = advantage.unsqueeze(-1)  # [B, 1]
            unclipped = ratio * adv_expanded  # [B, T_gen]
            clipped_ratio = torch.clamp(ratio, 1.0 - self.ppo_config.clip_range, 1.0 + self.ppo_config.clip_range)
            clipped = clipped_ratio * adv_expanded  # [B, T_gen]
            per_token_loss = -torch.min(unclipped, clipped)  # [B, T_gen]
            # Mask padding and compute mean per sequence, then mean over batch
            masked_loss = (per_token_loss * mask).sum(dim=1) / valid_counts  # [B]
            policy_loss = masked_loss.mean()  # scalar

            # KL penalty
            masked_kl = (kl_per_token * mask).sum(dim=1) / valid_counts  # [B]
            mean_kl = masked_kl.mean()  # scalar

            tot_loss = policy_loss + self.ppo_config.beta * mean_kl
            tot_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

        self.checkpoint_model()
        # After training, reload vLLM engine from updated HF weights
        self._reload_vllm_engine_from_hf()
