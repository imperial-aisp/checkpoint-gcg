import numpy as np
from tqdm import tqdm

from .utils import process_attn, calc_attn_score


class AttentionDetector():
    def __init__(self, model, pos_examples=None, neg_examples=None, use_token="first", instruction="Say xxxxxx", threshold=0.5):
        self.name = "attention"
        self.attn_func = "normalize_sum"
        self.model = model
        self.important_heads = model.important_heads
        self.instruction = instruction
        self.use_token = use_token
        self.threshold = threshold
        if pos_examples and neg_examples:
            pos_scores, neg_scores = [], []
            for prompt in tqdm(pos_examples, desc="pos_examples"):
                _, _, attention_maps, _, input_range, generated_probs = self.model.inference(
                    instruction=self.instruction, data=prompt, max_output_tokens=1
                )
                pos_scores.append(self.attn2score(attention_maps, input_range)[0])

            for prompt in tqdm(neg_examples, desc="neg_examples"):
                _, _, attention_maps, _, input_range, generated_probs = self.model.inference(
                    instruction=self.instruction, data=prompt, max_output_tokens=1
                )
                neg_scores.append(self.attn2score(attention_maps, input_range)[0])

            self.threshold = (np.mean(pos_scores) + np.mean(neg_scores)) / 2

        if pos_examples and not neg_examples:
            pos_scores = []
            for prompt in tqdm(pos_examples, desc="pos_examples"):
                _, _, attention_maps, _, input_range, generated_probs = self.model.inference(
                    instruction=self.instruction, data=prompt, max_output_tokens=1
                )
                pos_scores.append(self.attn2score(attention_maps, input_range)[0])

            self.threshold = np.mean(pos_scores) - 4 * np.std(pos_scores)

    def attn2score(self, attention_maps, input_range):
        """
        input_range: list of tuples indicating the ranges of different parts of the input
            e.g., [(0, inst_end), (inst_end, data_end), (data_end, inj_inst_end), (inj_inst_end, total_end)]
         0: benign instruction
         1: full data part (benign data + injected instruction + suffix)
         2: injected instruction
         3: suffix
        """
        if self.use_token == "first":
            attention_maps = [attention_maps[0]]

        # scores = []
        # for attention_map in attention_maps:
        #     heatmap = process_attn(
        #         attention_map, input_range, self.attn_func)
        #     score = calc_attn_score(heatmap, self.important_heads)
        #     scores.append(score)

        # return sum(scores) if len(scores) > 0 else 0
        
        attn_scores_benign_inst = []
        attn_scores_inj_inst = []
        attn_scores_suffix = []
        attn_scores_inj_inst_adv_suffix = []
        for attention_map in attention_maps[:1]:
            heatmap_benign_inst, heatmap_inj_inst, heatmap_suffix, heatmap_inj_inst_adv_suffix = process_attn(
                attention_map, input_range, self.attn_func)
            benign_inst_score = calc_attn_score(heatmap_benign_inst, self.important_heads)
            inj_inst_score = calc_attn_score(heatmap_inj_inst, self.important_heads)
            suffix_score = calc_attn_score(heatmap_suffix, self.important_heads)
            inj_inst_adv_suffix_score = calc_attn_score(heatmap_inj_inst_adv_suffix, self.important_heads)
            attn_scores_benign_inst.append(benign_inst_score)
            attn_scores_inj_inst.append(inj_inst_score)
            attn_scores_suffix.append(suffix_score)
            attn_scores_inj_inst_adv_suffix.append(inj_inst_adv_suffix_score)

        return sum(attn_scores_benign_inst) if len(attn_scores_benign_inst) > 0 else 0, sum(attn_scores_inj_inst) if len(attn_scores_inj_inst) > 0 else 0, sum(attn_scores_suffix) if len(attn_scores_suffix) > 0 else 0, sum(attn_scores_inj_inst_adv_suffix) if len(attn_scores_inj_inst_adv_suffix) > 0 else 0

    # def detect(self, data_prompt):
    #     _, _, attention_maps, _, input_range, _ = self.model.inference(
    #         self.instruction, data_prompt, max_output_tokens=1)

    #     focus_score = self.attn2score(attention_maps, input_range)
    #     return bool(focus_score <= self.threshold), {"focus_score": focus_score}

    def detect(self, instruction, data, suffix=None):
        _, _, attention_maps, _, input_range, _ = self.model.inference(
            instruction, data, suffix, max_output_tokens=1)

        focus_score = self.attn2score(attention_maps, input_range)[0]
        return bool(focus_score <= self.threshold), {"focus_score": focus_score}
    
    def get_attn_scores(self, text=None, instruction=None, data=None, suffix=None, inj_prompt=False):
        """Get the attention scores for the benign instruction, injected instruction, and suffix.
        """
        if text:
            generated_text, output_tokens, attention_maps, input_tokens, input_range, generated_probs = self.model.inference(
                text=text, max_output_tokens=4
            )
        else:
            generated_text, output_tokens, attention_maps, input_tokens, input_range, generated_probs = self.model.inference(
                instruction=instruction, data=data, suffix=suffix, inj_prompt=inj_prompt, max_output_tokens=4
            )
        attn_score_benign_inst, attn_score_inj_inst, attn_score_suffix, attn_score_inj_inst_adv_suffix = self.attn2score(attention_maps, input_range)
        return attn_score_benign_inst, attn_score_inj_inst, attn_score_suffix, attn_score_inj_inst_adv_suffix, generated_text