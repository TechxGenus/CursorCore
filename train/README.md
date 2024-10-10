# Training

This directory contains programs related to training the CursorCore series. Some other training engines that support custom templates should also work.

## Prepare conversation

Run the following program to inject the conversation template into the model:

```bash
python train/prepare_conversation.py --model_name_or_path deepseek-ai/deepseek-coder-1.3b-base --save_path train/formatted-deepseek-coder-1.3b-base
```

## Prepare data

Run the following program to format training data:

```bash
# WF Format
python train/prepare_data.py --input_path data/data.json --output_path data/train_data.json

# LC Format
python train/prepare_data.py --input_path data/data.json --output_path data/train_data_lc.json --format_type lc

# SR Format
python train/prepare_data.py --input_path data/data.json --output_path data/train_data_sr.json --format_type sr
```

## Training script

Our script can be run with common distributed launchers such as deepspeed and accelerate. Here is an example script for training models using deepspeed:

```bash
MODEL=$1
DATA=$2
OUTPUT=$3
mkdir -p $OUTPUT

deepspeed --master_port='10086' train/training.py \
    --model_name_or_path $MODEL \
    --data_path $DATA \
    --model_max_length 16384 \
    --batch_max_length 50000 \
    --learning_rate 5e-5 \
    --weight_decay 0 \
    --num_train_epochs 2 \
    --gradient_accumulation_steps 1 \
    --gradient_checkpointing \
    --lr_scheduler_type cosine \
    --warmup_steps 15 \
    --seed 10086 \
    --output_dir $OUTPUT \
    --bf16 True \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --load_best_model_at_end False \
    --save_total_limit 1000 \
    --logging_steps 20 \
    --tf32 True \
    --optim adafactor \
    --use_liger_kernel True \
    --deepspeed train/ds_config.json
```
