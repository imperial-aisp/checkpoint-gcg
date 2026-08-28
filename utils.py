import os
import json
import pandas as pd
import re


def read_jsonl_file(filepath):
    with open(filepath, "r") as f:
        lines = f.readlines()

    # Read the first JSON object (assume it's the multi-line config)
    config_lines = []
    i = 0
    for i, line in enumerate(lines):
        config_lines.append(line)
        if line.strip() == "}":
            break

    config_str = "".join(config_lines)
    config = json.loads(config_str)

    # Read the rest as JSONL
    entries = [json.loads(line) for line in lines[i + 1 :] if line.strip()]
    return config, entries


def extract_last_values(jsonl_filepath, success_keyword="success_begin_with"):
    """
    Extracts the 'step' and 'time_min' values from the last entry in a JSONL file.

    Args:
        jsonl_filepath (str): Path to the JSONL file
        success_keyword (str): Keyword to identify success in the JSONL file

    Returns:
        tuple: (step, time_min) values from the last entry
    """
    last_step = None
    last_time_min = None
    try:
        config, entries = read_jsonl_file(jsonl_filepath)
        if entries:
            last_entry = entries[-1]
            if "step" in last_entry and "time_min" in last_entry:
                last_step = last_entry["step"]
                last_time_min = last_entry["time_min"]
                suffix = last_entry.get("suffix", None)

            # individual sample attack
            if success_keyword in last_entry:
                success = last_entry[success_keyword]
            else:
                success = False

    except Exception as e:
        print(f"Error reading file {jsonl_filepath}: {e}")

    return last_step, last_time_min, suffix, success


def extract_n_from_filename(filename, keyword="sample"):
    """
    Extract the number of samples from the filename.

    Args:
        filename (str): Filename like 'bs512_seed0_l5_t1.0_static_k256_1samples.jsonl'

    Returns:
        int: Number of samples
    """
    if keyword == "sample":
        match = re.search(r"(\d+)samples\.jsonl$", filename)
    elif keyword == "checkpoint":
        match = re.search(r"checkpoint_(\d+)\.jsonl$", filename)
    elif keyword == "random_init":
        match = re.search(r"random_init_(\d+)\.jsonl$", filename)
    if match:
        return int(match.group(1))
    return None


def process_experiments_folder(
    base_folder="experiments", dir_keyword="checkpoint", defense_type="prompt_injection", last_entry_only=True
):
    """
    Process all checkpoints and JSONL files in the experiments folder.

    Args:
        base_folder (str): Path to the experiments folder
        dir_keyword (str): Keyword to identify the directory ('checkpoint' or 'sample')

    Returns:
        pandas.DataFrame: DataFrame containing checkpoint, nsamples, step, and time_min
    """
    if defense_type == "prompt_injection":
        success_keyword = "success_begin_with"
    elif defense_type == "jailbreak":
        success_keyword = "jailbroken"

    results = []

    if not os.path.exists(base_folder):
        print(f"Error: {base_folder} directory not found.")
        return pd.DataFrame()

    if dir_keyword == "checkpoint":
        # Walk through all subdirectories
        for root, dirs, files in os.walk(base_folder):
            # Check if this is a checkpoint directory
            checkpoint_match = re.search(r"checkpoint_(\d+)", root)
            if checkpoint_match:
                checkpoint_num = int(checkpoint_match.group(1))

                # Process all JSONL files in this checkpoint directory
                for file in files:
                    if file.endswith(".jsonl"):
                        nsamples = extract_n_from_filename(file, "sample")
                        if nsamples is not None:
                            jsonl_path = os.path.join(root, file)
                            step, time_min, suffix, num_success = extract_last_values(
                                jsonl_path, success_keyword
                            )

                            if step is not None and time_min is not None:
                                results.append(
                                    {
                                        "checkpoint": checkpoint_num,
                                        "nsamples": nsamples,
                                        "step": step,
                                        "suffix": suffix,
                                        "time_min": time_min,
                                        f"num_{success_keyword}": num_success,
                                    }
                                )
    elif dir_keyword == "sample":
        # Walk through all subdirectories
        for root, dirs, files in os.walk(base_folder):
            # Check if this is a sample directory
            sample_match = re.search(r"sample_(\d+)", root)
            if sample_match:
                sample_num = int(sample_match.group(1))

                # Process all JSONL files in this sample directory
                for file in files:
                    if file.endswith(".jsonl"):
                        checkpoint_num = extract_n_from_filename(file, "checkpoint")
                        if checkpoint_num is not None:
                            jsonl_path = os.path.join(root, file)

                            if last_entry_only:
                                step, time_min, suffix, sucess = extract_last_values(
                                    jsonl_path, success_keyword
                                )

                                results.append(
                                    {
                                        "sample": sample_num,
                                        "checkpoint_num": checkpoint_num,
                                        "step": step,
                                        "suffix": suffix,
                                        "time_min": time_min,
                                        f"{success_keyword}": sucess,
                                    }
                                )
                            else:
                                # Process all entries
                                config, entries = read_jsonl_file(jsonl_path)
                                for entry in entries:
                                    step = entry.get("step", None)
                                    time_min = entry.get("time_min", None)
                                    suffix = entry.get("suffix", None)
                                    sucess = entry.get(success_keyword, False)

                                    results.append(
                                        {
                                            "sample": sample_num,
                                            "checkpoint_num": checkpoint_num,
                                            "step": step,
                                            "suffix": suffix,
                                            "time_min": time_min,
                                            f"{success_keyword}": sucess,
                                        }
                                    )

                                

    df = pd.DataFrame(results)

    if not df.empty:
        if dir_keyword == "checkpoint":
            keys = ["checkpoint", "nsamples"]
        elif dir_keyword == "sample":
            keys = ["sample", "checkpoint_num", "step"]
        df = df.sort_values(keys).reset_index(drop=True)

    if dir_keyword == "sample":
        # Get the max steps for each (sample, checkpoint_num) combination
        max_per_checkpoint = df.groupby(['sample', 'checkpoint_num'])['step'].max()

        # Calculate cumulative sum of max steps per sample, then shift WITHIN each sample group
        cumsum_max = max_per_checkpoint.groupby(level='sample').cumsum()
        cumsum_max = cumsum_max.groupby(level='sample').shift(1, fill_value=0)
        # Map this back to the original dataframe
        df['total_steps'] = df['step'] + df.set_index(['sample', 'checkpoint_num']).index.map(cumsum_max)

        # df["total_steps"] = df.groupby("sample")["step"].cumsum()

    if dir_keyword == "checkpoint":
        df["total_steps"] = df.groupby("checkpoint")["step"].cumsum()

    return df

def extract_data_from_jsonl(
    file_path, last_entry_only=True, defense_type="prompt_injection"
):
    config, entries = read_jsonl_file(file_path)
    data = []
    if last_entry_only:
        entry = entries[-1]
        if defense_type == "prompt_injection":
            data.append(
                {
                    "sample_id": entry["sample_id"],
                    "loss": entry["loss"],
                    "success_begin_with": entry.get("success_begin_with", False),
                    "success_in_response": entry.get("success_in_response", False),
                    "suffix": entry.get("suffix", ""),
                    "step": entry["step"],
                }
            )
        elif defense_type == "jailbreak":
            data.append(
                {
                    "sample_id": entry["sample_id"],
                    "loss": entry["loss"],
                    "jailbroken": entry.get("jailbroken", False),
                    "target_generated": entry.get("target_generated", False),
                    "suffix": entry.get("suffix", ""),
                    "step": entry["step"],
                    "generated_output": entry.get("generated", ""),
                }
            )
    else:
        for entry in entries:
            if defense_type == "prompt_injection":
                data.append(
                    {
                        "sample_id": entry["sample_id"],
                        "loss": entry["loss"],
                        "success_begin_with": entry.get("success_begin_with", False),
                        "success_in_response": entry.get("success_in_response", False),
                        "suffix": entry.get("suffix", ""),
                        "step": entry["step"],
                    }
                )
            elif defense_type == "jailbreak":
                data.append(
                    {
                        "sample_id": entry["sample_id"],
                        "loss": entry["loss"],
                        "jailbroken": entry.get("jailbroken", False),
                        "target_generated": entry.get("target_generated", False),
                        "suffix": entry.get("suffix", ""),
                        "step": entry["step"],
                        "generated_output": entry.get("generated", ""),
                    }
                )

    return data


def process_multiple_jsonl(
    results_dir, last_entry_only=True, defense_type="prompt_injection"
):
    files = [
        os.path.join(results_dir, f)
        for f in os.listdir(results_dir)
        if f.endswith(".jsonl")
    ]
    all_data = []
    for file_path in files:
        all_data.extend(
            extract_data_from_jsonl(
                file_path, last_entry_only=last_entry_only, defense_type=defense_type
            )
        )

    # Convert to DataFrame
    df = pd.DataFrame(all_data)
    df["sample_id"] = df["sample_id"].astype(int)

    # sort by sample_id
    df = df.sort_values(by=["sample_id", "step"]).reset_index(drop=True)
    return df