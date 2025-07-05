export PYTHONPATH=$PYTHONPATH:/<YOUR PROJECT PATH>
export DEV_PATH=<YOUR PROJECT PATH>
accelerate launch \
    --mixed_precision fp16 \
    --num_processes 1 \
    scripts/flow_generation/inference_flow_generation_libero.py \
    --config-name inference.yaml \