#!/usr/bin/env bash
set -xeuo pipefail

ID=${1:-"dapo-qwen3-8b-megatron-sglang-baseline"}
HOME_DIR=/apps

project_name=${PROJECT_NAME:-DAPO-mxfp8-rebase}
exp_name=$ID

################################################### quick config ###################################################

rollout_mode="async"
rollout_name="sglang" # sglang or vllm
return_raw_chat="False"
if [ "$rollout_mode" = "async" ]; then
    # export VLLM_USE_V1=1
    return_raw_chat="True"
fi
dtype="bfloat16" # ["bfloat16", "float16"]

adv_estimator=grpo

use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0

clip_ratio_low=0.2
clip_ratio_high=0.28

rollout_is=token
rollout_is_threshold=2.0
rollout_rs=null
rollout_rs_threshold=null
rollout_rs_threshold_lower=null
rollout_token_veto_threshold=null

max_prompt_length=${MAX_PROMPT_LENGTH:-1024}
max_response_length=${MAX_RESPONSE_LENGTH:-$((1024 * 4))}
enable_overlong_buffer=${ENABLE_OVERLONG_BUFFER:-True}
overlong_buffer_len=${OVERLONG_BUFFER_LEN:-512}
overlong_penalty_factor=${OVERLONG_PENALTY_FACTOR:-1.0}

loss_agg_mode="token-mean"

enable_filter_groups=${ENABLE_FILTER_GROUPS:-True}
filter_groups_metric=${FILTER_GROUPS_METRIC:-acc}
max_num_gen_batches=${MAX_NUM_GEN_BATCHES:-10}
train_prompt_bsz=${TRAIN_PROMPT_BSZ:-32}
gen_prompt_bsz=${GEN_PROMPT_BSZ:-96}
n_resp_per_prompt=${N_RESP_PER_PROMPT:-16}
train_prompt_mini_bsz=${TRAIN_PROMPT_MINI_BSZ:-32}

RAY_ADDRESS=${RAY_ADDRESS:-"http://localhost:8265"}
WORKING_DIR=${WORKING_DIR:-"${PWD}"}
RUNTIME_ENV=${RUNTIME_ENV:-"${WORKING_DIR}/verl/trainer/runtime_env.yaml"}
NNODES=${NNODES:-1}
echo "NNODES: ${NNODES}"

# Paths
RAY_DATA_HOME=${RAY_DATA_HOME:-"${HOME_DIR}"}
MODEL_PATH="/apps/models/Qwen3-8B-Base"
# MODEL_PATH="/apps/models/Qwen3-30B-A3B-Base"
CKPTS_DIR=${CKPTS_DIR:-"${RAY_DATA_HOME}/ckpts/${project_name}/${exp_name}"}
TRAIN_FILE=${TRAIN_FILE:-"${RAY_DATA_HOME}/data/dapo-math-17k-one.parquet"}
TEST_FILE=${TEST_FILE:-"${RAY_DATA_HOME}/data/aime-2024.parquet"}

# Algorithm
temperature=1.0
top_p=1.0
top_k=-1 # 0 for HF rollout, -1 for vLLM rollout
val_top_p=${VAL_TOP_P:-1.0}

# Performance Related Parameter
use_dynamic_bsz=True
actor_ppo_max_token_len=$((max_prompt_length + max_response_length))
infer_ppo_max_token_len=$((max_prompt_length + max_response_length))
offload=True
gen_tp=1
actor_lr_warmup_steps=${ACTOR_LR_WARMUP_STEPS:-10}
rollout_max_num_batched_tokens=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-$((1024 * 16))}
rollout_enforce_eager=${ROLLOUT_ENFORCE_EAGER:-True}
trainer_logger=${TRAINER_LOGGER:-'["console","wandb"]'}
trainer_val_before_train=${TRAINER_VAL_BEFORE_TRAIN:-False}
trainer_test_freq=${TRAINER_TEST_FREQ:-10}
trainer_save_freq=${TRAINER_SAVE_FREQ:-5}
trainer_total_epochs=${TRAINER_TOTAL_EPOCHS:-10}
trainer_total_training_steps=${TRAINER_TOTAL_TRAINING_STEPS:-}
trainer_resume_mode=${TRAINER_RESUME_MODE:-auto}

export VERL_LOGGING_LEVEL=INFO
export NVTE_FP8_BLOCK_SCALING_FP32_SCALES=1
export TORCHDYNAMO_DISABLE=1
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export VERL_SET_TRITON_TORCH_ALLOCATOR=1
export WANDB_API_KEY=${WANDB_API_KEY:?WANDB_API_KEY must be set}

################################################### start of config ###################################################

DATA=(
    data.train_files="${TRAIN_FILE}"
    data.val_files="${TEST_FILE}"
    data.prompt_key=prompt
    data.truncation='left'
    data.return_raw_chat=$return_raw_chat
    data.filter_overlong_prompts=True
    data.max_prompt_length=${max_prompt_length}
    data.max_response_length=${max_response_length}
    data.gen_batch_size=${gen_prompt_bsz}
    data.train_batch_size=${train_prompt_bsz}
)

ALGORITHM=(
    algorithm.adv_estimator=${adv_estimator}
    algorithm.use_kl_in_reward=${use_kl_in_reward}
    algorithm.kl_ctrl.kl_coef=${kl_coef}
    algorithm.filter_groups.enable=${enable_filter_groups}
    algorithm.filter_groups.max_num_gen_batches=${max_num_gen_batches}
    algorithm.filter_groups.metric=${filter_groups_metric}
    algorithm.rollout_correction.rollout_is=${rollout_is}
    algorithm.rollout_correction.rollout_is_threshold=${rollout_is_threshold}
)

PERF_OPT=(
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    actor_rollout_ref.model.use_remove_padding=True

    +actor_rollout_ref.actor.optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=True
    +actor_rollout_ref.actor.optim.override_optimizer_config.use_precision_aware_optimizer=True
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=False
)

MEGATRON_BF16_TRAINING=(
    +actor_rollout_ref.actor.megatron.override_ddp_config.overlap_grad_reduce=True
    +actor_rollout_ref.actor.megatron.override_ddp_config.overlap_param_gather=True
    +actor_rollout_ref.actor.optim.override_optimizer_config.overlap_param_gather=True
    actor_rollout_ref.actor.megatron.use_distributed_optimizer=True
)

ACTOR=(
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss}
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef}
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low}
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high}
    actor_rollout_ref.actor.clip_ratio_c=10.0
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz}
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len}
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.actor.optim.lr_warmup_steps=${actor_lr_warmup_steps}
    actor_rollout_ref.actor.optim.weight_decay=0.1
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz}
    actor_rollout_ref.actor.megatron.param_offload=${offload}
    actor_rollout_ref.actor.megatron.optimizer_offload=${offload}
    actor_rollout_ref.actor.megatron.grad_offload=${offload}
    actor_rollout_ref.actor.megatron.use_mbridge=True
    # actor_rollout_ref.actor.megatron.vanilla_mbridge=False
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.optim.clip_grad=1.0
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_permute_fusion=False
    actor_rollout_ref.actor.megatron.override_transformer_config.attention_backend='fused'
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode}
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=4
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=1
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=1
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=1
    actor_rollout_ref.actor.megatron.context_parallel_size=1
)

ROLLOUT=(
    actor_rollout_ref.rollout.n=${n_resp_per_prompt}
    +actor_rollout_ref.rollout.engine_kwargs.sglang.fp8_gemm_runner_backend=triton
    +actor_rollout_ref.rollout.engine_kwargs.sglang.moe_runner_backend=flashinfer_trtllm_routed
    actor_rollout_ref.rollout.calculate_log_probs=True
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp}
    actor_rollout_ref.rollout.enable_chunked_prefill=True
    actor_rollout_ref.rollout.max_num_batched_tokens=${rollout_max_num_batched_tokens}
    actor_rollout_ref.rollout.temperature=${temperature}
    actor_rollout_ref.rollout.top_p=${top_p}
    actor_rollout_ref.rollout.top_k=${top_k}
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature}
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p}
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k}
    actor_rollout_ref.rollout.val_kwargs.do_sample=True
    actor_rollout_ref.rollout.val_kwargs.n=1
    actor_rollout_ref.rollout.name=${rollout_name}
    actor_rollout_ref.rollout.enforce_eager=${rollout_enforce_eager}
)

FORWARD_ONLY_SETS=(
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz}
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz}
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len}
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len}
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=4
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=1
    actor_rollout_ref.ref.megatron.expert_model_parallel_size=1
    actor_rollout_ref.ref.megatron.expert_tensor_parallel_size=1
    actor_rollout_ref.ref.megatron.context_parallel_size=1
)

MODEL=(
    actor_rollout_ref.model.path="${MODEL_PATH}"
)

REWARD_MODEL=(
    reward_model.reward_manager=dapo
    +reward_model.overlong_buffer.enable=${enable_overlong_buffer}
    +reward_model.overlong_buffer.len=${overlong_buffer_len}
    +reward_model.overlong_buffer.penalty_factor=${overlong_penalty_factor}
    +reward_model.reward_kwargs.max_resp_len=${max_response_length}
)

TRAINER=(
    trainer.logger="${trainer_logger}"
    trainer.project_name="${project_name}"
    trainer.experiment_name="${exp_name}"
    trainer.n_gpus_per_node=8
    trainer.nnodes="${NNODES}"
    trainer.val_before_train=${trainer_val_before_train}
    trainer.test_freq=${trainer_test_freq}
    trainer.save_freq=${trainer_save_freq}
    trainer.max_actor_ckpt_to_keep=5
    trainer.total_epochs=${trainer_total_epochs}
    trainer.default_local_dir="${CKPTS_DIR}"
    trainer.resume_mode=${trainer_resume_mode}
    +trainer.use_legacy_worker_impl="disable"
)

if [ -n "${trainer_total_training_steps}" ]; then
    TRAINER+=(trainer.total_training_steps=${trainer_total_training_steps})
fi

################################################### start script ###################################################

# WANDB_API_KEY must be provided via the environment.
# export ROCR_VISIBLE_DEVICES=
# ray start --head --node-ip-address 0.0.0.0 --num-gpus 8

# actor_rollout_ref.rollout.quantization=nvfp4_qat
# +actor_rollout_ref.rollout.prequant_model_path="${QUANT_MODEL_PATH}"
python3 -m recipe.dapo.main_dapo \
    --config-path=config \
    --config-name='dapo_megatron_trainer.yaml' \
    "${DATA[@]}" \
    "${ALGORITHM[@]}" \
    "${PERF_OPT[@]}" \
    "${MEGATRON_BF16_TRAINING[@]}" \
    "${MODEL[@]}" \
    "${ROLLOUT[@]}" \
    "${ACTOR[@]}" \
    "${FORWARD_ONLY_SETS[@]}" \
    "${REWARD_MODEL[@]}" \
    "${TRAINER[@]}"
