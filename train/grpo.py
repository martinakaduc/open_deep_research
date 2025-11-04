import os
from utils import run_server, shutdown_server

# launch the master node of ray in container
RAY_MASTER_CMD = (
    "ray start --head --node-ip-address 0.0.0.0 --num-gpus {num_gpus} --port 6379"
)

GRPO_CMD = (
    "ray job submit --address='http://127.0.0.1:8265' "
    '--runtime-env-json=\'{ "working_dir": "/openrlhf" }\' '
    "-- python3 -m openrlhf.cli.train_ppo_ray "
    "--ref_num_nodes 1 "
    "--ref_num_gpus_per_node 1 "
    "--reward_num_nodes 1 "
    "--reward_num_gpus_per_node 1 "
    "--critic_num_nodes 1 "
    "--critic_num_gpus_per_node 1 "
    "--actor_num_nodes 1 "
    "--actor_num_gpus_per_node 1 "
    "--vllm_num_engines 4 "
    "--vllm_tensor_parallel_size 1 "
    "--colocate_all_models "
    "--vllm_gpu_memory_utilization 0.5 "
    "--pretrain {model_path} "
    "--reward_pretrain {reward_model_path} "
    "--save_path {save_path} "
    "--ckpt_path {ckpt_path} "
    "--save_hf_ckpt "
    "--micro_train_batch_size {batch_size} "
    "--train_batch_size 128 "
    "--micro_rollout_batch_size {rollout_batch_size} "
    "--rollout_batch_size 1024 "
    "--n_samples_per_prompt 1 "
    "--max_epochs {max_epochs} "
    "--prompt_max_len 1024 "
    "--max_samples 100000 "
    "--generate_max_len 1024 "
    "--zero_stage 3 "
    "--bf16 "
    "--actor_learning_rate 5e-7 "
    "--critic_learning_rate 9e-6 "
    "--init_kl_coef 0.01 "
    "--prompt_data {data_path} "
    "--input_key query "
    "--label_key response "
    "--normalize_reward "
    "--gradient_checkpointing "
    "--packing_samples "
    "--vllm_sync_backend nccl "
    "--enforce_eager "
    "--vllm_enable_sleep "
    "--deepspeed_enable_sleep "
    "--advantage_estimator group_norm "
    "--lora_rank 16 "
    "--lora_alpha 32 "
    "--target_modules ['q_proj', 'k_proj', 'v_proj'] "
)

EXPORT_CMD = (
    "python3 -m openrlhf.cli.export_hf_model "
    "--model_path {model_path} "
    "--lora_path {lora_path} "
    "--output_path {output_path} "
    "--bf16 "
)

# Enable wandb logging
# --use_wandb {wandb_token}

# Support REINFORCE++  | RLOO | REINFORCE++-baseline | GRPO | Dr. GRPO
# --advantage_estimator reinforce | rloo | reinforce_baseline | group_norm | dr_grpo

# Set --init_kl_coef to 0 will not launch the reference model

# Support remote reward model (HTTP)
# --remote_rm_url http://localhost:5000/get_reward

# Support N samples
# --n_samples_per_prompt 4


def run_grpo_training(
    model_path: str,
    reward_model_path: str,
    save_path: str,
    ckpt_path: str,
    data_path: str,
    batch_size: int,
    rollout_batch_size: int,
    max_epochs: int,
    num_gpus: int,
):
    ray_pid = run_server(RAY_MASTER_CMD.format(num_gpus=num_gpus))

    grpo_command = GRPO_CMD.format(
        model_path=model_path,
        reward_model_path=reward_model_path,
        save_path=save_path,
        ckpt_path=ckpt_path,
        batch_size=batch_size,
        rollout_batch_size=rollout_batch_size,
        max_epochs=max_epochs,
        data_path=data_path,
    )

    print("Running GRPO training with command:")
    print(grpo_command)
    os.system(grpo_command)

    print("GRPO training completed.")
    shutdown_server(ray_pid)


def export_grpo_model(
    model_path: str,
    lora_path: str,
    output_path: str,
):
    export_command = EXPORT_CMD.format(
        model_path=model_path,
        lora_path=lora_path,
        output_path=output_path,
    )

    print("Exporting GRPO model with command:")
    print(export_command)
    os.system(export_command)

    print("Model export completed.")
