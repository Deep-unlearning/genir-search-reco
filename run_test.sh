
DATASET=All_Beauty

CKPT_PATH=./ckpt/tiger/All_Beauty/T5-base-nosk-nosk-5e-4/checkpoint-1786
RESULTS_FILE=$CKPT_PATH/TIGER-$DATASET.json

python test.py \
    --gpu_id 9 \
    --ckpt_path $CKPT_PATH \
    --results_file $RESULTS_FILE \
    --test_batch_size 10 \
    --num_beams 20 \
    --index_file .index.json \
    --filter_items
