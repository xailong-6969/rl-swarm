log_dir: ${oc.env:ROOT,.}/logs

hydra:
  run:
    dir: ${log_dir}
  job_logging:
    handlers:
      console:
        level: INFO
    root:
      level: DEBUG

training:
  max_round: 1000000
  max_stage: 1
  hf_push_frequency: 1
  num_generations: 2
  num_transplant_trees: 2
  seed: 42
  dtype: 'float32'
  max_new_tokens: 256

reward_config:
  ollama_model: qwen2.5-coder:1.5b-instruct
  temperature: 0.0
  num_predict: 256

blockchain:
  alchemy_url: "https://gensyn-testnet.g.alchemy.com/public"
  swarm_contract_address: ${oc.env:SWARM_CONTRACT,null} # This is set by modal-login in run_rl_swarm.sh
  org_id: ${oc.env:ORG_ID,null} # This is set by modal-login in run_rl_swarm.sh
  mainnet_chain_id: 685685 # currently unused, will be used with WalletSwarmCoordinator
  modal_proxy_url: "http://localhost:3000/api/"
  swarm_coordinator_abi_path: "code_gen_exp/contracts/SwarmCoordinator_0.4.2.json"


eval:
  judge_base_url: "https://codezero-judge.gensyn.ai"

game_manager:
  _target_: code_gen_exp.src.manager.SwarmGameManager
  max_stage: ${training.max_stage}
  max_round: ${training.max_round}
  log_dir: ${log_dir}
  hf_token: ${oc.env:HUGGINGFACE_ACCESS_TOKEN,null}
  hf_push_frequency: ${training.hf_push_frequency}
  rewards_ollama_model: ${reward_config.ollama_model}
  run_mode: "train_and_evaluate"
  game_state: 
    _target_: genrl.state.game_state.GameState
    round: 0
    stage: 0
  trainer:
    _target_: code_gen_exp.src.trainer.GRPOTrainerModule
    models:
      - _target_: transformers.AutoModelForCausalLM.from_pretrained
        pretrained_model_name_or_path: ${oc.env:MODEL_NAME, ${gpu_model_choice:${default_large_model_pool},${default_small_model_pool}}} 
    config:
      _target_: genrl.trainer.grpo_trainer.GRPOTrainerConfig
      dtype: ${training.dtype}
      epsilon: 0.2
      epsilon_high: 0.28
      max_new_tokens: ${training.max_new_tokens}
      num_generations: ${training.num_generations}
    log_with: wandb
    log_dir: ${log_dir}
    judge_base_url: ${eval.judge_base_url}
  reward_manager:
    _target_: genrl.rewards.DefaultRewardManager
    reward_fn_store:
      _target_: genrl.rewards.reward_store.RewardFnStore
      max_rounds: ${training.max_round}
      reward_fn_stores:
        - _target_: genrl.rewards.reward_store.RoundRewardFnStore
          num_stages: ${training.max_stage}
          reward_fns:
            - _target_: code_gen_exp.src.solver_rewards.CodeGenerationRewards
              solver_tokenizer_path: ${game_manager.trainer.models.0.pretrained_model_name_or_path}
              solver_token_lim: ${training.max_new_tokens}
              ollama_config:
                _target_: code_gen_exp.src.solver_rewards.RewardsOllamaConfig
                model: ${reward_config.ollama_model}
                temperature: ${reward_config.temperature}
                num_predict: ${reward_config.num_predict}
  data_manager:
    _target_: code_gen_exp.src.solver_data.CodeGenerationDataManager
    system_prompt: 'solver'
    batch_size: 2
    local_batch_size: 1
    proposer_batch_size: 1
    num_generations: ${training.num_generations}
    num_transplant_trees: ${training.num_transplant_trees}
  communication_kwargs:
    identity_path: ${oc.env:IDENTITY_PATH,null}
    startup_timeout: 120
    beam_size: 10
    get_retries: 1
  coordinator:
    _target_: code_gen_exp.src.coordinator.ModalSwarmCoordinator
    web3_url: ${blockchain.alchemy_url}
    contract_address: ${blockchain.swarm_contract_address}
    org_id: ${blockchain.org_id}
    modal_proxy_url: ${blockchain.modal_proxy_url}
    swarm_coordinator_abi_json: ${blockchain.swarm_coordinator_abi_path}

default_large_model_pool: 
  - deepseek-ai/deepseek-coder-1.3b-instruct
  - Qwen/Qwen2.5-Coder-1.5B-Instruct

default_small_model_pool:
  - Qwen/Qwen2.5-Coder-0.5B-Instruct

proposer:
  _target_: code_gen_exp.src.proposer_service.ProposerService
  service_config:
    _target_: code_gen_exp.src.proposer_service.ProposerServiceConfig
    model: ${oc.env:MODEL_NAME, ${gpu_model_choice:${default_large_model_pool},${default_small_model_pool}}} 
    num_proposals: 3
    train_batch_size: 3   
    identity_path: ${oc.env:IDENTITY_PATH,null}
    startup_timeout: 120
    beam_size: 10
    get_retries: 0
  ppo_config:
    _target_: code_gen_exp.src.proposer.PPOConfig
  vllm_config:
    _target_: code_gen_exp.src.proposer.VllmConfig
  coordinator:
    _target_: code_gen_exp.src.coordinator.ModalSwarmCoordinator
    web3_url: ${blockchain.alchemy_url}
    contract_address: ${blockchain.swarm_contract_address}
    org_id: ${blockchain.org_id}
    modal_proxy_url: ${blockchain.modal_proxy_url}
    swarm_coordinator_abi_json: ${blockchain.swarm_coordinator_abi_path}

    
    
