import argparse
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
import sys
from typing import List
os.environ["WANDB_PROJECT"] = "TIGER AMAZON 2023 Video_Games"
import torch
import transformers
import wandb

from transformers import T5Tokenizer, T5Config, T5ForConditionalGeneration, AutoModelForSeq2SeqLM

from utils import *
from collator import Collator

def train(args):


    set_seed(args.seed)
    ensure_dir(args.output_dir)

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    local_rank = int(os.environ.get("LOCAL_RANK") or 0)
    if local_rank == 0:
        print(vars(args))
        run = wandb.init(
            # set the wandb project where this run will be logged
            project="TIGER AMAZON 2023 Video_Games",
            # track hyperparameters and run metadata
            config={
            "learning_rate": args.learning_rate,
            "architecture": args.base_model,
            "batch_size": args.per_device_batch_size,
            "epochs": args.epochs,
            }
        )
        wandb.log({"accuracy": 0.9})
        print(f"WandB run URL: {run.get_url()}")

    if ddp:
        device_map = {"": local_rank}
    device = torch.device("cuda", local_rank)


    config = T5Config.from_pretrained(args.base_model)
    tokenizer = T5Tokenizer.from_pretrained(
        args.base_model,
        model_max_length=args.model_max_length,
    )
    args.deepspeed = None
    gradient_checkpointing= False


    train_data, valid_data = load_datasets(args)
    # print(len(valid_data))
    # train_data = SeqRecDataset(args, mode="processed", mode_dataset=".train")#, sample_num=args.train_data_sample_num)
    # train_data = ConcatDataset([train_data])
    # valid_data = SeqRecDataset(args, mode="processed", mode_dataset=".valid")
    
    add_num = tokenizer.add_tokens(train_data.datasets[0].get_new_tokens())
    config.vocab_size = len(tokenizer)
    if local_rank == 0:
        print("add {} new token.".format(add_num))
        print("data num:", len(train_data))
        print("data num valid:", len(valid_data))
        tokenizer.save_pretrained(args.output_dir)
        config.save_pretrained(args.output_dir)
        print(train_data[100])
        print(valid_data[100])

    collator = Collator(args, tokenizer)

    # model = T5ForConditionalGeneration(config)
    if "t5" in args.base_model:
        model = AutoModelForSeq2SeqLM.from_pretrained(args.base_model)
        print("load from pretrained")
    else:
        model = T5ForConditionalGeneration(config)
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)
    if local_rank == 0:
        print(model)


    if not ddp and torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True


    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=valid_data,
        args=transformers.TrainingArguments(
            seed=args.seed,
            per_device_train_batch_size=args.per_device_batch_size,
            per_device_eval_batch_size=args.per_device_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            warmup_ratio=args.warmup_ratio,
            num_train_epochs=args.epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            logging_dir=args.output_dir,
            lr_scheduler_type=args.lr_scheduler_type,
            fp16=args.fp16,
            bf16=args.bf16,
            logging_steps=args.logging_step,
            optim=args.optim,
            eval_strategy=args.save_and_eval_strategy,
            save_strategy=args.save_and_eval_strategy,
            eval_steps=args.save_and_eval_steps,
            save_steps=args.save_and_eval_steps,
            output_dir=args.output_dir,
            save_total_limit=5,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False, 
            ddp_find_unused_parameters=False if ddp else None,
            # report_to="wandb",
            eval_delay= 1 if args.save_and_eval_strategy=="epoch" else 5000,
        ),
        tokenizer=tokenizer,
        data_collator=collator,
    )
    model.config.use_cache = False


    trainer.train(
        resume_from_checkpoint=args.resume_from_checkpoint,
    )

    trainer.save_state()
    trainer.save_model(output_dir=args.output_dir)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='TIGER')
    parser = parse_global_args(parser)
    parser = parse_train_args(parser)
    parser = parse_dataset_args(parser)

    args = parser.parse_args()

    train(args)