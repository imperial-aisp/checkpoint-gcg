import os
import subprocess
import json

# Download data dependencies
output_data_dir = 'data'
data_urls = {
    "training": [
        "https://raw.githubusercontent.com/vinid/safety-tuned-llamas/refs/heads/main/data/training/saferpaca_Instructions_2000.json",
        # training data for safety_tuned_llama
        "https://raw.githubusercontent.com/gururise/AlpacaDataCleaned/refs/heads/main/alpaca_data_cleaned.json",
        # training data for secalign and struq
        "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/refs/heads/main/alpaca_data.json",
        # training data for secalign and struq
    ],
    "eval": [
        "https://huggingface.co/datasets/hamishivi/alpaca-farm-davinci-003-2048-token/resolve/main/davinci_003_outputs.json",
        # evaluation data for secalign and struq
        "https://raw.githubusercontent.com/llm-attacks/llm-attacks/refs/heads/main/data/advbench/harmful_behaviors.csv",
        # evaluation data for safety_tuned_llama
        "https://raw.githubusercontent.com/meta-llama/PurpleLlama/refs/heads/main/CybersecurityBenchmarks/datasets/prompt_injection/prompt_injection.json",
        # evaluation of universal attack
        'https://raw.githubusercontent.com/egozverev/Should-It-Be-Executed-Or-Processed/refs/heads/main/datasets/SEP_dataset.json',
        # SEP dataset
    ],
    "configs": [
        "https://raw.githubusercontent.com/vinid/safety-tuned-llamas/refs/heads/main/configs/alpaca.json",
        # training configs for safety_tuned_llama
    ],
}

os.makedirs(output_data_dir, exist_ok=True)
for data_type in data_urls:
    os.makedirs(os.path.join(output_data_dir, data_type), exist_ok=True)

    for data_url in data_urls[data_type]:
        filename = data_url.split("/")[-1]
        filepath = os.path.join(output_data_dir, data_type, filename)

        if not os.path.exists(filepath):
            commands = [
                ["wget", data_url, "-O", filepath],
                ["curl", "-L", data_url, "-o", filepath],
            ]

            for cmd in commands:
                try:
                    subprocess.run(cmd, check=True)
                    print(f"Downloaded {filename} to {filepath}")
                    break
                except FileNotFoundError:
                    # command not installed
                    continue
                except subprocess.CalledProcessError:
                    # command installed but failed
                    continue
            else:
                print(f"Failed to download {filename} from {data_url}")

        else:
            print(f"{filename} already exists at {filepath}")

# Post-process CyberSecEval2 data
with open(os.path.join(output_data_dir, 'eval', 'prompt_injection.json')) as f:
    raw_cyberseceval2 = json.load(f)
processed_cyberseceval2 = [entry for entry in raw_cyberseceval2 if entry['injection_type'] == 'indirect']
processed_cyberseceval2 = [{'instruction': entry['test_case_prompt'],
                            'input': entry['user_input'],
                            'judge_question': entry['judge_question']} for entry in processed_cyberseceval2]
with open(os.path.join(output_data_dir, 'eval', 'CySE_prompt_injections.json'), 'w') as f:
    json.dump(processed_cyberseceval2, f, indent=4, default=str)

# Post-process SEP data
processed_sep = []
with open(os.path.join(output_data_dir, 'eval', 'SEP_dataset.json')) as f:
    raw_sep = json.load(f)
    for entry in raw_sep:
        injection = entry['prompt_instructed'].replace(entry['prompt_clean'], '')
        if injection.startswith(' ') or injection.startswith('\n'): injection = injection[1:]
        processed_sep.append({
            'instruction': entry['system_prompt_clean'],
            'input': entry['prompt_clean'],
            'injection': injection,
        })
with open(os.path.join(output_data_dir, 'eval', 'SEP_dataset_test.json'), 'w') as f:
    json.dump(processed_sep, f, indent=4, default=str)
