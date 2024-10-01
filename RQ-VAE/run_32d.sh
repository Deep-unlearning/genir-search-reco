DATASET=Video_Games
DATA_PATH=./ID_generation/preprocessing/processed/$DATASET/$DATASET.embeddings.npy
CKPT_PATH=./results/$DATASET/3x256-0.0-0.0-0.0

python -u main.py \
    --num_emb_list 256 256 256 \
    --sk_epsilons 0.0 0.0 0.0\
    --device cuda:0 \
    --data_path $DATA_PATH \
    --batch_size 1024 \
    --ckpt_dir $CKPT_PATH \

