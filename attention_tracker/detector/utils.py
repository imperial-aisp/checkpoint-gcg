import torch 
import numpy as np

def process_attn(attention, rng, attn_func):
    heatmap_benign_inst = np.zeros((len(attention), attention[0].shape[1]))
    heatmap_inj_inst = np.zeros((len(attention), attention[0].shape[1]))
    heatmap_suffix = np.zeros((len(attention), attention[0].shape[1]))
    heatmap_inj_inst_adv_suffix = np.zeros((len(attention), attention[0].shape[1]))
    for i, attn_layer in enumerate(attention):
        attn_layer = attn_layer.to(torch.float32).numpy()

        if "sum" in attn_func:
            last_token_attn_to_inst = np.sum(attn_layer[0, :, -1, rng[0][0]:rng[0][1]], axis=1)
            benign_inst_attn = last_token_attn_to_inst
            inj_inst_attn = np.sum(attn_layer[0, :, -1, rng[2][0]:rng[2][1]], axis=1)
            suffix_attn = np.sum(attn_layer[0, :, -1, rng[3][0]:rng[3][1]], axis=1)
            inj_inst_adv_suffix_attn = np.sum(attn_layer[0, :, -1, rng[4][0]:rng[4][1]], axis=1)
        
        elif "max" in attn_func:
            last_token_attn_to_inst = np.max(attn_layer[0, :, -1, rng[0][0]:rng[0][1]], axis=1)
            benign_inst_attn = last_token_attn_to_inst
            inj_inst_attn = np.max(attn_layer[0, :, -1, rng[2][0]:rng[2][1]], axis=1)
            suffix_attn = np.max(attn_layer[0, :, -1, rng[3][0]:rng[3][1]], axis=1)
            inj_inst_adv_suffix_attn = np.max(attn_layer[0, :, -1, rng[4][0]:rng[4][1]], axis=1)

        else: raise NotImplementedError
            
        last_token_attn_to_benign_inst_sum = np.sum(attn_layer[0, :, -1, rng[0][0]:rng[0][1]], axis=1)
        last_token_attn_to_full_data_part_sum = np.sum(attn_layer[0, :, -1, rng[1][0]:rng[1][1]], axis=1)

        if "normalize" in attn_func:
            epsilon = 1e-8
            heatmap_benign_inst[i, :] = benign_inst_attn / (last_token_attn_to_benign_inst_sum + last_token_attn_to_full_data_part_sum + epsilon)

            heatmap_inj_inst[i, :] = inj_inst_attn / (last_token_attn_to_benign_inst_sum + last_token_attn_to_full_data_part_sum + epsilon)

            heatmap_suffix[i, :] = suffix_attn / (last_token_attn_to_benign_inst_sum + last_token_attn_to_full_data_part_sum + epsilon)

            heatmap_inj_inst_adv_suffix[i, :] = inj_inst_adv_suffix_attn / (last_token_attn_to_benign_inst_sum + last_token_attn_to_full_data_part_sum + epsilon)
        else:
            heatmap_benign_inst[i, :] = benign_inst_attn
            heatmap_inj_inst[i, :] = inj_inst_attn
            heatmap_suffix[i, :] = suffix_attn
            heatmap_inj_inst_adv_suffix[i, :] = inj_inst_adv_suffix_attn
    heatmap_benign_inst = np.nan_to_num(heatmap_benign_inst, nan=0.0)
    heatmap_inj_inst = np.nan_to_num(heatmap_inj_inst, nan=0.0)
    heatmap_suffix = np.nan_to_num(heatmap_suffix, nan=0.0)
    heatmap_inj_inst_adv_suffix = np.nan_to_num(heatmap_inj_inst_adv_suffix, nan=0.0)

    return heatmap_benign_inst, heatmap_inj_inst, heatmap_suffix, heatmap_inj_inst_adv_suffix

def calc_attn_score(heatmap, heads):
    score = np.mean([heatmap[l, h] for l, h in heads], axis=0)
    return score

