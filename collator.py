import torch
import copy
import argparse
from dataclasses import dataclass

import transformers
import math
from torch.utils.data import Sampler
import torch.distributed as dist
from transformers import LlamaForCausalLM, LlamaTokenizer, LlamaConfig, T5Tokenizer, T5Config, T5ForConditionalGeneration, DataCollatorForLanguageModeling


class Collator(object):

    def __init__(self, args, tokenizer):
        self.args = args
        self.only_train_response = args.only_train_response
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = 0
        # print(self.tokenizer.model_max_length)

    def __call__(self, batch):

        input_texts = [d["input_ids"] for d in batch]
        label_texts = [d["labels"] for d in batch]

        inputs = self.tokenizer(input_texts,
                                return_tensors="pt",
                                padding="longest",
                                max_length=self.tokenizer.model_max_length,
                                truncation=True,
                                return_attention_mask=True)

        labels = self.tokenizer(label_texts,
                                return_tensors="pt",
                                padding="longest",
                                max_length=self.tokenizer.model_max_length,
                                truncation=True,
                                return_attention_mask=True)
        inputs['labels'] = labels['input_ids']
        inputs['labels'][inputs['labels'] == self.tokenizer.pad_token_id] = -100

        # print(inputs.input_ids[0])
        # print(inputs.labels[0])

        # print the shape of the input and labels
        # print(f"Input shape: {inputs['input_ids'].shape}, Labels shape: {inputs['labels'].shape}")


        return inputs

class GPT2Collator(object):
    def __init__(self, args, tokenizer):
        self.args = args
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id  # GPT-2 uses the EOS token as the pad token

    def __call__(self, batch):
        input_texts = [d["input_ids"] for d in batch]
        label_texts = [d["labels"] for d in batch]  # Each element in label_texts contains only the next item

        # Tokenize input sequences
        input_encodings = self.tokenizer(
            input_texts,
            padding="longest",
            truncation=False,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt"
        )

        # Tokenize label sequences
        label_encodings = self.tokenizer(
            label_texts,
            padding="longest",
            truncation=False,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt"
        )

        # Initialize labels tensor with -100 (which will ignore these positions in loss computation)
        labels = torch.full_like(input_encodings['input_ids'], fill_value=-100)

        print("Input shape:", input_encodings['input_ids'].shape)
        print("Label shape:", label_encodings['input_ids'].shape)
        print("Labels:", label_encodings['input_ids'])

        # Identify the boundary where the label part starts
        for i, d in enumerate(batch):
            input_len = len(self.tokenizer.encode(d["input_ids"], add_special_tokens=False))
            labels[i, :input_len] = label_encodings['input_ids'][i, :input_len]
            # shift the labels to the right and replace by the label_encodings
            labels[i, input_len:] = label_encodings['input_ids'][i, :-input_len]


        input_encodings['labels'] = labels

        print("Input shape:", input_encodings['input_ids'].shape)
        print("Label shape:", input_encodings['labels'].shape)

        # Decode only the valid tokens from input_encodings
        valid_input_ids = input_encodings['input_ids'][0].tolist()
        valid_labels = [id_ for id_ in input_encodings['labels'][0].tolist() if id_ != -100]

        print("Input:", self.tokenizer.decode(valid_input_ids))
        print("Label:", self.tokenizer.decode(valid_labels))

        return input_encodings



class TestCollator(object):

    def __init__(self, args, tokenizer):
        self.args = args
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = 0

    def __call__(self, batch):

        input_texts = [d["input_ids"] for d in batch]
        targets = [d["labels"] for d in batch]

        inputs = self.tokenizer(
            text=input_texts,
            return_tensors="pt",
            padding="longest",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_attention_mask=True,
        )

        return (inputs, targets)

