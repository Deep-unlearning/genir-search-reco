# export CUDA_VISIBLE_DEVICES=0


# DATASET=Video_Games
# MODEL=google-t5/t5-small

# CKPT_PATH=./ckpt/tiger/$DATASET/$MODEL-4x256-sentencet5-space-prefix/checkpoint-57500
# RESULTS_FILE=./results/t5-small-4x256-sentencet5-space-prefix-57500.json


# torchrun --nproc_per_node=1 --master_port=9324 test_ddp.py \
#     --ckpt_path $CKPT_PATH \
#     --dataset $DATASET \
#     --results_file $RESULTS_FILE \
#     --test_batch_size 8 \
#     --num_beams 50 \
#     --index_file .sentence_t5_4x256.index.json \
#     --filter_items

export CUDA_VISIBLE_DEVICES=0,1


DATASET=Video_Games
MODEL=google-t5/t5-small
CKPT_PATH=./ckpt/tiger/Video_Games/google-t5/t5-small-4x256-productsearch-item2index/checkpoint-212500
RESULTS_FILE=./results/t5-small-4x256-seqrec-productsearch-212500-search.json


torchrun --nproc_per_node=2 --master_port=9324 test_ddp.py \
    --ckpt_path $CKPT_PATH \
    --dataset $DATASET \
    --results_file $RESULTS_FILE \
    --test_batch_size 8 \
    --num_beams 50 \
    --index_file .sentence_t5_4x256.index.json \
    --filter_items


# DATASET=All_Beauty

# CKPT_PATH=./ckpt/tiger/All_Beauty/T5X-nosk-nosk-5e-4/checkpoint-1927
# RESULTS_FILE=TIGER-$DATASET.json


# torchrun --nproc_per_node=2 --master_port=9324 test_ddp.py \
#     --ckpt_path $CKPT_PATH \
#     --dataset $DATASET \
#     --results_file $RESULTS_FILE \
#     --test_batch_size 8 \
#     --num_beams 50 \
#     --index_file  .index.json \
#     --filter_items