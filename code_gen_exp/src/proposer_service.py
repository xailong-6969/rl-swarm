from dataclasses import dataclass
import logging
import random
from genrl.communication.hivemind.hivemind_backend import HivemindBackend, HivemindRendezvouz
from code_gen_exp.src.coordinator import ModalSwarmCoordinator
from code_gen_exp.src.proposer import Proposer, PPOConfig, VllmConfig, PromptUpdateConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ProposerServiceConfig:
    model: str
    num_proposals: int
    train_batch_size: int
    identity_path: str
    startup_timeout: int
    beam_size: int
    get_retries: int
    max_round: int
    do_training: bool = False
    second_pass: bool = True
    prompt_update_config: PromptUpdateConfig = None


class ProposerClientDHT:
    def __init__(self, backend: HivemindBackend):
        self.backend = backend

    def insert_proposal(self, proposer_model: str, proposals: list[dict]):
        objs = [{
            "proposer_model": proposer_model,
            "proposal_question": proposal_dict["question"],
            "proposal_tests": proposal_dict["tests"],
            "proposal_raw": proposal_dict["proposal_raw"],
        } for proposal_dict in proposals]
        try:
            self.backend.put(objs, sub_key="proposer".encode())
        except Exception as e:
            get_logger().debug(f"Failed to insert proposals: {e}")

    def request_training_data(self, train_batch_size: int) -> list[dict]:
        data = []
        try:
            obj_ = self.backend.get(sub_key="solver".encode())
        except Exception as e:
            get_logger().debug(f"Failed to get training data: {e}")

        if obj_ is None or len(obj_) == 0:
            return data

        objs = list(obj_.values())

        # Batching data so this is a nested list
        for list_of_samples in objs:
            for sample in list_of_samples:
                if sample['dataset'] == 'proposer':
                    data.append(sample)
                    
        if len(data) > train_batch_size:
            data = random.sample(data, train_batch_size)
        return data


class ProposerService:
    def __init__(self,
                 service_config: ProposerServiceConfig,
                 ppo_config: PPOConfig,
                 vllm_config: VllmConfig,
                 prompt_update_config: PromptUpdateConfig,
                 coordinator: ModalSwarmCoordinator = None):
        
        initial_peers = coordinator.get_bootnodes() if coordinator is not None else None
        backend = HivemindBackend(
            identity_path=service_config.identity_path,
            startup_timeout=service_config.startup_timeout,
            beam_size=service_config.beam_size,
            get_retries=service_config.get_retries,
            initial_peers=initial_peers,
        )
        proposer_client = ProposerClientDHT(backend)
        
        if vllm_config.use_vllm:
            assert not service_config.do_training, "VLLM is not compatible with training"
        
        self.proposer_client = proposer_client
        self.coordinator = coordinator
        self.proposer = Proposer(
            service_config.model, 
            ppo_config, 
            vllm_config, 
            second_pass=service_config.second_pass,
            prompt_update_config=prompt_update_config
        )
        logger.info(f'Proposer initialized with model {service_config.model}')
        self.config = service_config
        self.prompt_update_frequency = prompt_update_config.prompt_update_frequency
    
    def insert(self):
        try:
            model_name = self.proposer.model.name_or_path
        except AttributeError:
            model_name = "none"
        proposals = []
        for _ in range(self.config.num_proposals):
            proposal = self.proposer.generate_proposal()
            if proposal is not None:
                proposals.append(proposal)
        self.proposer_client.insert_proposal(model_name, proposals)
        logger.info(f"{len(proposals)} proposals inserted")

    def update_proposer_prompt(self):
        """
        Fetch recent training data, extract rewards, and update prompt difficulty
        based on solver performance.
        """
        training_data = self.proposer_client.request_training_data(self.config.train_batch_size)
        if len(training_data) == 0:
            logger.info("No training data found")
            return
            
        # Flatten all rewards from training data
        # Each sample may have a list of rewards or a single reward
        rewards = []
        for sample in training_data:
            reward = sample.get("reward", None)
            if reward is not None:
                if isinstance(reward, list):
                    # If reward is a list, flatten it
                    rewards.extend([r for r in reward if r is not None])
                else:
                    rewards.append(reward)

        if len(rewards) == 0:
            logger.info("No rewards found in training data")
            return

        # Update the proposer's prompt based on recent rewards
        self.proposer.update_prompt_difficulty(rewards)
        logger.info(f"Prompt difficulty updated based on {len(rewards)} reward samples")


    def train(self):

        training_data = self.proposer_client.request_training_data(self.config.train_batch_size)
        if len(training_data) == 0:
            logger.info("No training data found")
            return
        elif len(training_data) > self.config.train_batch_size:
            logger.info("Training data is larger than batch size")
            training_data = training_data[:self.config.train_batch_size]
            
        rewards = []
        proposals = []
        for sample in training_data:
            rewards.append(sample["reward"])
            proposals.append(sample["proposal_raw"])


        if len(rewards) == 0:
            logger.info("No training data found")
            return

        logger.info(f"Training with {len(proposals)} proposals")

        self.proposer.train(rewards, proposals)
        logger.info(f"Training completed")


    def run(self):
        logger.info("Starting proposer service")
        iteration = 0

        while True:
            self.insert()
            
            # Update prompt difficulty based on recent rewards
            if iteration % self.prompt_update_frequency == 0:
                self.update_proposer_prompt()
            
            if self.config.do_training:
                self.train()
            
            iteration += 1

