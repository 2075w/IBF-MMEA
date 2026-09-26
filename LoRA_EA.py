import json
import re

import numpy as np
import pandas as pd
from collections import defaultdict
import torch
from PIL import Image
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM, Qwen2VLForConditionalGeneration, \
    AutoConfig
from transformers import TrainingArguments, Trainer, DataCollatorForSeq2Seq, DataCollatorForLanguageModeling, \
    AutoProcessor

# System prompt for knowledge graph alignment task
prompt1 = "You're an expert in knowledge graph alignment."


def save_peft_model(trainer, tokenizer, lora_path):
    """
    Save the trained PEFT model and tokenizer.

    Args:
        trainer: HuggingFace Trainer instance
        tokenizer: Tokenizer instance
        lora_path: Path to save the PEFT model
    """
    peft_model_id = lora_path
    trainer.model.save_pretrained(peft_model_id)
    tokenizer.save_pretrained(peft_model_id)
    print("save model finished!")


def get_acc(predicts):
    """
    Calculate accuracy metrics for entity alignment predictions.

    Args:
        predicts: List of prediction dictionaries

    Prints:
        - Small model accuracy
        - Alignment accuracy
        - Average inference rounds
        - Number of correctly aligned entities
    """
    acc = 0
    all = 0
    noin = 0
    rs = 0
    for predict in predicts:
        tar, align, pre, ca, r = predict.values()
        rs += r
        all += 1
        if str(align) in ca:
            if pre == str(align) and str(align) != "999999":
                acc += 1
            if str(align) == "999999":
                noin += 1
    print('Small model accuracy:', 1 - noin / all)
    print("accuracy of align:", acc / all)
    print("Average round:", rs / all)
    print("Number of aligned entities:", acc)


def model_param(model):
    """
    Print model parameter names for debugging.

    Args:
        model: PyTorch model instance
    """
    names = []
    for name, param in model.named_parameters():
        if name not in names:
            names.append(name)
        else:
            print(name)
        # if param.requires_grad:
        #     print(f"{name}: requires_grad={param.requires_grad}")


def load_and_merge_json_files(file_paths):
    """
    Load multiple JSON files and group them by target field.

    Args:
        file_paths: List of JSON file paths

    Returns:
        List of data grouped by target field, sorted by target
    """
    all_data = []

    # 1. Load all data from files
    for path in file_paths:
        with open(path, 'r') as f:
            data = json.load(f)
            all_data.extend(data)  # Merge data

    # 2. Group by target
    target_groups = defaultdict(list)
    for item in all_data:
        target_groups[item["target"]].append(item)

    # 3. Return sorted by target
    redata = [group for target, group in sorted(target_groups.items())]
    return [group for target, group in sorted(target_groups.items())]


class GLM_Lora:
    """
    GLM model with LoRA fine-tuning for entity alignment.

    Attributes:
        model_path: Path to pre-trained model
        tokenizer: Model tokenizer
        unaligned: Identifier for unaligned entities
        device: Computation device
        prompt: System prompt
        output: Output token identifier
    """

    def __init__(self, model_path):
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, use_fast=True, trust_remote_code=True)
        self.unaligned = '999999'
        self.train = True
        self.device = "cuda:1"
        self.prompt = prompt1
        self.output = "Output:"

    def process_func(self, example):
        """
        Preprocess data examples for GLM model.

        Args:
            example: Dictionary containing instruction, input, and output

        Returns:
            Dictionary with input_ids, attention_mask, and labels
        """
        MAX_LENGTH = 2048
        input_ids, attention_mask, labels = [], [], []
        instruction = self.tokenizer((f"[gMASK]<sop><|system|>\n{self.prompt}<|user|>\n"
                                      f"{example['instruction'] + example['input']}<|assistant|>\n"
                                      ).strip(),
                                     add_special_tokens=False)

        response = self.tokenizer(f"{example['output']}", add_special_tokens=False)
        input_ids = instruction["input_ids"] + response["input_ids"] + [self.tokenizer.pad_token_id]
        attention_mask = instruction["attention_mask"] + response["attention_mask"] + [
            1]  # EOS token also needs attention
        labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [self.tokenizer.pad_token_id]

        # Truncate if exceeds maximum length
        if len(input_ids) > MAX_LENGTH:
            input_ids = input_ids[:MAX_LENGTH]
            attention_mask = attention_mask[:MAX_LENGTH]
            labels = labels[:MAX_LENGTH]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

    def model_lora(self, data_paths, lora_path, arg, accelerator=None):
        """
        Fine-tune GLM model with LoRA.

        Args:
            data_paths: List of training data file paths
            lora_path: Path to save LoRA model
            arg: Training arguments
            accelerator: HuggingFace Accelerator instance (optional)
        """
        dfs = []
        for path in data_paths:
            df = pd.read_json(path)
            dfs.append(df)
        df = pd.concat(dfs, axis=0, ignore_index=True)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle data
        ds = Dataset.from_pandas(df)
        torch.cuda.empty_cache()

        self.tokenizer.pad_token = self.tokenizer.eos_token
        tokenized_id = ds.map(self.process_func, remove_columns=ds.column_names)

        model = AutoModelForCausalLM.from_pretrained(self.model_path, torch_dtype=torch.bfloat16,
                                                     trust_remote_code=True)
        model_param(model)

        # LoRA configuration
        config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=["query_key_value"],  # Target attention modules
            inference_mode=False,  # Training mode
            r=4,  # LoRA rank
            lora_alpha=32,  # LoRA alpha
            lora_dropout=arg.dropout  # Dropout ratio
        )

        # Training arguments
        args = TrainingArguments(
            output_dir="./output/GLM4",
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            logging_steps=100,
            num_train_epochs=arg.epoch,
            save_strategy="epoch",
            learning_rate=1e-5,
            save_on_each_node=True,
            gradient_checkpointing=True,
            ddp_find_unused_parameters=False,  # Accelerate training
        )

        model = get_peft_model(model, config)

        model.enable_input_require_grads()
        model.config.use_cache = False

        model.print_trainable_parameters()

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=tokenized_id,
            data_collator=DataCollatorForSeq2Seq(tokenizer=self.tokenizer),
        )

        if accelerator:
            # Prepare model and trainer with Accelerate
            model, trainer = accelerator.prepare(model, trainer)

        trainer.train()
        save_peft_model(trainer, self.tokenizer, lora_path + '_' + str(arg.epoch))

    def model_test(self, datapaths, lora_path, save_path, accelerator=None):
        """
        Test GLM model with LoRA fine-tuning.

        Args:
            datapaths: List of test data file paths
            lora_path: Path to LoRA model
            save_path: Path to save test results
            accelerator: HuggingFace Accelerator instance (optional)
        """
        self.train = False
        torch.cuda.set_device(self.device)

        print("lora path:{0}\ndata path：{1}".format(lora_path, datapaths))
        candi_ids_part = r'\d+(?=\:)'
        dfs = load_and_merge_json_files(datapaths)

        # Load base model
        model = AutoModelForCausalLM.from_pretrained(self.model_path, torch_dtype=torch.bfloat16,
                                                     trust_remote_code=True)

        # Load LoRA weights
        model = PeftModel.from_pretrained(model, model_id=lora_path).to(self.device).eval()

        acc = 0
        MAX_LENGTH = 4000
        tooLong = 0
        isNone = 0
        no_in_candi = 0
        acc_emb = 0
        savedata = []
        predicts = []

        for i in tqdm(range(len(dfs))):
            rounds = 0
            for j in range(len(dfs[0])):
                ins, _, label, target = list(dfs[i][j].values())
                label = str(label)
                acc_emb += label != self.unaligned

                # Prepare model inputs
                inputs = self.tokenizer.apply_chat_template(
                    [
                        {"role": "assistant", "content": self.prompt},
                        {"role": "user", "content": ins.strip()}
                    ],
                    add_generation_prompt=True,
                    tokenize=True,
                    return_tensors="pt",
                    return_dict=True
                ).to(self.device)

                # Extract candidate entity IDs
                candis = list(
                    set(re.findall(candi_ids_part, ins.split("candidate entities:")[-1].split('relationships')[0])))
                candis.append(self.unaligned)

                # Handle long inputs
                if inputs['input_ids'].shape[1] > MAX_LENGTH:
                    tooLong += 1
                    if inputs['input_ids'].shape[1] > 4060:
                        if j == len(dfs) - 1:
                            savedata.append(
                                {"target": target, "label": label, "predict": self.unaligned, "candidates": candis,
                                 'round': rounds})
                        rounds = 0
                        continue

                    inputs['input_ids'] = inputs['input_ids'][:MAX_LENGTH]

                # Generate prediction
                gen_kwargs = {"max_new_tokens": 100, "do_sample": False, "top_k": 1}
                with torch.no_grad():
                    outputs = model.generate(**inputs, **gen_kwargs)
                    outputs = outputs[:, inputs['input_ids'].shape[1]:]
                    predict = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

                if self.output in predict:
                    predict = predict.split("Output:")[-1]

                # Extract numeric prediction
                try:
                    predict = re.findall(r'\d+', predict)[0]
                except:
                    isNone += 1
                    predict = self.unaligned
                predicts.append(predict)

                if predict not in candis:
                    isNone += 1
                if predict == self.unaligned:
                    if j == len(dfs[0]) - 1:
                        no_in_candi += 1
                    else:
                        continue
                rounds += j + 1
                savedata.append(
                    {"target": target, "label": label, "predict": predict, "candidates": candis, 'round': rounds})
                if predict == label and label != self.unaligned:
                    acc += 1
                break

        get_acc(savedata)
        print("Too long inputs:{0}, Small model actual accuracy：{1}".format(tooLong, acc_emb / len(dfs)))
        with open(save_path, 'w') as f:
            json.dump(savedata, f)
        print("Path of save result:", save_path)
