from trl import DPOTrainer, KTOTrainer, ORPOTrainer, KTOConfig, ORPOConfig
import os, re, time
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence
import numpy as np
from copy import deepcopy
import transformers
from config import PROMPT_FORMAT_PREF_DATASET, DELIMITERS, DEFAULT_TOKENS, SPECIAL_DELM_TOKENS, TEST_INJECTED_PROMPT
from test import load_data
from struq import jload, jdump, format_with_other_delimiters, jlload
from train import ModelArguments, DataArguments, AttackArguments, TrainingArguments, smart_tokenizer_and_embedding_resize
from utils import process_experiments_folder, process_multiple_jsonl
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
import torch
import random

# def set_global_seed(seed):
#     # torch.manual_seed(seed)
#     np.random.seed(seed)
#     random.seed(seed)

#     # if torch.cuda.is_available():
#     #     torch.cuda.manual_seed(seed)
#     #     torch.cuda.manual_seed_all(seed)
#     #     torch.backends.cudnn.deterministic = True
#     #     torch.backends.cudnn.benchmark = False
    
#     # transformers.set_seed(seed)
#     # torch.use_deterministic_algorithms(True, warn_only=True)


def load_model_for_training(model_args, training_args, continue_training=False):
    for alignment in ["dpo", "kto", "orpo"]:
        base_model_index = model_args.model_name_or_path.find(alignment) - 1
        if base_model_index > 0:
            break
        else:
            base_model_index = False

    base_model_path = (
        model_args.model_name_or_path[:base_model_index]
        if base_model_index
        else model_args.model_name_or_path.split("_")[0]
    )

    if continue_training:
        # continue training from a secalign model
        merged_save_path = model_args.model_name_or_path + "_merged" # path to the merged model (base + lora)
        if os.path.exists(merged_save_path):
            model_to_load = merged_save_path
        else:
            model_to_load = base_model_path
    else:
        # start secalign training from base model
        model_to_load = base_model_path

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_to_load,
        cache_dir=training_args.cache_dir,
        low_cpu_mem_usage=True,
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_to_load,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side=model_args.padding_side,
        use_fast=False,
    )

    if (continue_training and not os.path.exists(merged_save_path)) or (not continue_training): 
        special_tokens_dict = dict()
        special_tokens_dict["pad_token"] = DEFAULT_TOKENS["pad_token"]
        special_tokens_dict["eos_token"] = DEFAULT_TOKENS["eos_token"]
        special_tokens_dict["bos_token"] = DEFAULT_TOKENS["bos_token"]
        special_tokens_dict["unk_token"] = DEFAULT_TOKENS["unk_token"]
        special_tokens_dict["additional_special_tokens"] = SPECIAL_DELM_TOKENS
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=special_tokens_dict, tokenizer=tokenizer, model=model
        )

        if continue_training:
            model = PeftModel.from_pretrained(
                model, model_args.model_name_or_path, is_trainable=True
            )
            model = model.merge_and_unload()
            
            model.save_pretrained(merged_save_path)
            tokenizer.save_pretrained(merged_save_path)

    return model, tokenizer

def generate_preference_data(
    clean_data_path, frontend_delimiters, attack, alignment, tokenizer
):
    preference_data_path = (
        clean_data_path.split("/")[0]
        + "/preference_"
        + frontend_delimiters
        + "_"
        + alignment
        + "_"
        + attack
        + "_"
        + clean_data_path.split("/")[2]
    )
    naive_proportion = 0.9
    if not os.path.exists(preference_data_path):
        clean_data = jload(clean_data_path)
        prompt_dict = PROMPT_FORMAT_PREF_DATASET[frontend_delimiters]
        preference_data = []
        if "Completion" in attack:
            ref_inst_resp = {}
            for ref_sample in jload("data/training/alpaca_data.json"):
                ref_inst_resp[
                    ref_sample["instruction"].replace(tokenizer.pad_token, "")
                ] = ref_sample["output"]

        for i in range(len(clean_data)):
            if clean_data[i].get("input", "") == "":
                continue
            current_sample = deepcopy(clean_data[i])
            if current_sample["input"][-1] != ".":
                current_sample["input"] += "."
            injected_sample = np.random.choice(clean_data)
            injected_prompt = (
                (
                    "answer the following question. "
                    + injected_sample["instruction"]
                    + " "
                    + injected_sample["input"]
                )
                if injected_sample["instruction"][-1] == "?"
                else (
                    injected_sample["instruction"][0].lower()
                    + injected_sample["instruction"][1:]
                    + " "
                    + injected_sample["input"]
                )
            )
            if attack == "Naive":
                current_sample["input"] += (
                    " " + injected_prompt[0].upper() + injected_prompt[1:]
                )
            elif attack == "NaiveCompletion":
                if np.random.rand() < naive_proportion:
                    current_sample["input"] += (
                        " " + injected_prompt[0].upper() + injected_prompt[1:]
                    )
                else:
                    current_sample["input"] += (
                        "\n\n"
                        + DELIMITERS["SpclSpclSpcl"][2]
                        + "\n"
                        + ref_inst_resp.get(
                            current_sample["instruction"], current_sample["output"]
                        )
                        + "\n\n"
                        + DELIMITERS["SpclSpclSpcl"][0]
                        + "\n"
                        + injected_prompt.capitalize()
                    )
                    if injected_sample["input"] != "":
                        current_sample["input"] += (
                            "\n\n"
                            + DELIMITERS["SpclSpclSpcl"][1]
                            + "\n"
                            + injected_sample["input"]
                        )
                    current_sample["input"] = format_with_other_delimiters(
                        current_sample["input"], test=False
                    )
            else:
                raise NotImplementedError

            if alignment == "dpo" or alignment == "orpo":
                    preference_data.append(
                        {
                            "prompt": prompt_dict["prompt_input"].format_map(
                                current_sample
                            ),
                            "chosen": current_sample["output"] + tokenizer.eos_token,
                            "rejected": injected_sample["output"] + tokenizer.eos_token,
                        }
                    )
            elif alignment == "kto" or alignment == "bco":
                preference_data.append(
                    {
                        "prompt": prompt_dict["prompt_input"].format_map(
                            current_sample
                        ),
                        "completion": current_sample["output"] + tokenizer.eos_token,
                        "label": True,
                    }
                )
                preference_data.append(
                    {
                        "prompt": prompt_dict["prompt_input"].format_map(
                            current_sample
                        ),
                        "completion": injected_sample["output"] + tokenizer.eos_token,
                        "label": False,
                    }
                )

        jdump(preference_data, preference_data_path)
    time.sleep(10)
    return load_dataset("json", data_files=preference_data_path, split="train")


def generate_adversarial_training_data(
    samples_path, sample_ids, frontend_delimiters, tokenizer, 
    suffix_folder=None, alignment="dpo"
):  
    if suffix_folder is None:
        attack = "None"
    else:
        attack = "GCG" if "direct" in suffix_folder.lower() else "CGCG"
    
    preference_data_path = (
        samples_path.split("/")[0]
        + "/preference_"
        + frontend_delimiters
        + "_"
        + alignment
        + "_"
        + attack
        + ".json"
    )

    samples = load_data(samples_path)

    if attack == "CGCG":
        attack_results_df = process_experiments_folder(
            base_folder=suffix_folder,
            dir_keyword="sample",
            last_entry_only=True,
        )
        sample_key = "sample"
    elif attack == "GCG":
        attack_results_df = process_multiple_jsonl(
            results_dir=suffix_folder, 
            last_entry_only=True,
        )
        sample_key = "sample_id"

    gcg_preference_data = []
    for sample_id in sample_ids:
        d_item = samples[sample_id]
        if (
            d_item["input"][-1] != "."
            and d_item["input"][-1] != "!"
            and d_item["input"][-1] != "?"
        ):
            d_item["input"] += "."

        if attack != "None":
            adv_suffix = attack_results_df[
                attack_results_df[sample_key] == sample_id
            ]["suffix"].tolist()[-1]

            prompt = PROMPT_FORMAT_PREF_DATASET[frontend_delimiters]["prompt_input"].format_map(
                {
                    "instruction": d_item["instruction"],
                    "input": d_item["input"] + " " + TEST_INJECTED_PROMPT + " " + adv_suffix,
                }
            )
        else:
            prompt = PROMPT_FORMAT_PREF_DATASET[frontend_delimiters]["prompt_input"].format_map(
                {
                    "instruction": d_item["instruction"],
                    "input": d_item["input"] + " " + TEST_INJECTED_PROMPT,
                }
            )

        gcg_preference_data.append(
            {
                "prompt": prompt,
                "chosen": d_item["output"] + tokenizer.eos_token,
                "rejected": "Hacked!" + tokenizer.eos_token,
            }
        )
    
    jdump(gcg_preference_data, preference_data_path)
    time.sleep(10)
    return load_dataset(
        "json", data_files=preference_data_path, split="train"
    )


def align():
    # set_global_seed(100)
    gcg_training = False
    continue_training = gcg_training

    sample_ids = [
        170, 15, 7, 55, 92, 134, 125, 124, 158, 184, 
        75, 149, 138, 71, 186, 145, 176, 118, 16, 135, 
        190, 22, 104, 141, 4, 74, 136, 44, 63, 108
    ]

    parser = transformers.HfArgumentParser(
        (ModelArguments, TrainingArguments, DataArguments, AttackArguments)
    )
    model_args, training_args, data_args, attack_args = (
        parser.parse_args_into_dataclasses()
    )
    os.makedirs(training_args.output_dir, exist_ok=True)
    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}
    if "Instruct" in model_args.model_name_or_path:
        frontend_delimiters = model_args.model_name_or_path.split("/")[-1].split("_")[0]
    else:
        _, frontend_delimiters, _, _ = model_args.model_name_or_path.split("/")[
            -1
        ].split("_")

    model, tokenizer = load_model_for_training(
        model_args, training_args, continue_training=continue_training
    )

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=64,
        lora_alpha=8,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"],
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    print(training_args.output_dir, "\n\n\n")
    if model_args.window_size > 0:
        model.config.window = model_args.window_size

    if gcg_training:
        full_dataset = generate_adversarial_training_data(
            samples_path="data/eval/davinci_003_outputs.json",
            sample_ids=sample_ids,
            frontend_delimiters=frontend_delimiters,
            tokenizer=tokenizer,
            suffix_folder=data_args.suffix_folder,
            alignment=attack_args.alignment,
        )

    # Split dataset into train and eval
    split_dataset = full_dataset.train_test_split(test_size=0.1, seed=100)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]

    trainer = {
        "dpo": DPOTrainer,
        "kto": KTOTrainer,
        "orpo": ORPOTrainer,
    }[attack_args.alignment](
        model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    align()
