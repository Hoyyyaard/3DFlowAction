

export PYTHONPATH=$PYTHONPATH:<YOUR PROJECT PATH>
export DEV_PATH=<YOUR PROJECT PATH>
export WANDB_MODE=offline



accelerate launch \
    --main_process_ip $CHIEF_IP   --main_process_port 29504  --machine_rank $INDEX \
    scripts/flow_generation/train_flow_generation_libero.py \
    --config-name train_flow_generation_bridge_wovae \
