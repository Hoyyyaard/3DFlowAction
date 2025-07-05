

export PYTHONPATH=$PYTHONPATH:<YOUR PROJECT PATH>
export DEV_PATH=<YOUR PROJECT PATH>
export HYDRA_FULL_ERROR=1
accelerate launch \
    --mixed_precision fp16 \
    --main_process_port 29510 \
    --num_processes 1 \
    scripts/flow_generation/train_flow_generation_libero.py \
    --config-name preprocess_bridge_dataset \
