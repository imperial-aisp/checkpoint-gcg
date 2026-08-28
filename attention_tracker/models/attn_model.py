import torch
from .model import Model
from .utils import sample_token, get_last_attn
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn.functional as F
from test import load_secalign_model
from config import PROMPT_FORMAT, DELIMITERS, TEST_INJECTED_PROMPT

# device = '0' if torch.cuda.is_available() else 'cpu'

class AttentionModel(Model):
    def __init__(self, config, device='0'):
        super().__init__(config)
        self.name = config["model_info"]["name"]
        self.max_output_tokens = int(config["params"]["max_output_tokens"])

        checkpoint_dir = config["model_info"]["checkpoint_dir"]
        model_name_or_path = config["model_info"]["model_name_or_path"]
        # model_id = config["model_info"]["model_id"]
        # self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        # self.model = AutoModelForCausalLM.from_pretrained(
        #     model_id,
        #     torch_dtype=torch.bfloat16,
        #     device_map=device,
        #     trust_remote_code=True,
        #     attn_implementation="eager",
        # ).eval()

        self.model, self.tokenizer, _, _ = load_secalign_model(
            checkpoint_dir, model_name_or_path, device=device, checkpoint=-1
        )

        self.top_k = None # 50
        self.top_p = None

        if config["params"]["important_heads"] == "all":
            attn_size = self.get_map_dim()
            self.important_heads = [[i, j] for i in range(
                attn_size[0]) for j in range(attn_size[1])]
        else:
            self.important_heads = config["params"]["important_heads"]


    def get_map_dim(self):
        _, _, attention_maps, _, _, _ = self.inference(instruction="print hi", data="")
        attention_map = attention_maps[0]
        return len(attention_map), attention_map[0].shape[1]
    

    def get_input_ids(self, instruction, data, suffix=None, inj_prompt=False):
        delm = DELIMITERS["Meta-Llama-3-8B-Instruct"]
        sep_tokens = self.tokenizer("\n\n", add_special_tokens=False).input_ids
        num_tok_sep = len(sep_tokens)

        if suffix is not None:
            static_str = (
                f"{delm[0]}\n{instruction}\n\n{delm[1]}\n{data} {TEST_INJECTED_PROMPT}"
            )
            static_toks = self.tokenizer(static_str).input_ids  # BOS auto-added

            # Cumulative-prefix tokenization to derive slice boundaries.
            p_role1 = f"{delm[0]}\n"
            p_inst  = p_role1 + instruction
            p_role2 = p_inst + f"\n\n{delm[1]}\n"
            p_data  = p_role2 + data
            n_role1 = len(self.tokenizer(p_role1).input_ids)
            n_inst  = len(self.tokenizer(p_inst).input_ids)
            n_role2 = len(self.tokenizer(p_role2).input_ids)
            n_data  = len(self.tokenizer(p_data).input_ids)
            n_static = len(static_toks)

            benign_inst_role_slice = slice(None, n_role1)
            benign_inst_slice      = slice(n_role1, n_inst)
            data_role_slice        = slice(n_inst, n_role2)
            data_slice             = slice(n_role2, n_data)
            inj_inst_slice         = slice(n_data, n_static)

            # Suffix part: matches GCG exactly — " " + suffix tokenized
            # separately, appended after the static prompt tokens.
            space_toks   = self.tokenizer(" ", add_special_tokens=False).input_ids
            suffix_toks  = self.tokenizer(suffix, add_special_tokens=False).input_ids
            resp_toks    = self.tokenizer(delm[2], add_special_tokens=False).input_ids
            nl_toks      = self.tokenizer("\n", add_special_tokens=False).input_ids

            toks = static_toks + space_toks + suffix_toks + sep_tokens + resp_toks + nl_toks

            adv_suffix_start = n_static
            adv_suffix_stop  = n_static + len(space_toks) + len(suffix_toks)
            adv_suffix_slice = slice(adv_suffix_start, adv_suffix_stop)
            inj_inst_adv_suffix_slice = slice(inj_inst_slice.start, adv_suffix_stop)

            assistant_role_start = adv_suffix_stop + num_tok_sep
            assistant_role_slice = slice(assistant_role_start, len(toks))
            full_data_slice = slice(data_slice.start, adv_suffix_stop)

        elif inj_prompt:
            static_str = (
                f"{delm[0]}\n{instruction}\n\n{delm[1]}\n{data} {TEST_INJECTED_PROMPT}"
            )
            static_toks = self.tokenizer(static_str).input_ids  # BOS auto-added

            p_role1 = f"{delm[0]}\n"
            p_inst  = p_role1 + instruction
            p_role2 = p_inst + f"\n\n{delm[1]}\n"
            p_data  = p_role2 + data
            n_role1 = len(self.tokenizer(p_role1).input_ids)
            n_inst  = len(self.tokenizer(p_inst).input_ids)
            n_role2 = len(self.tokenizer(p_role2).input_ids)
            n_data  = len(self.tokenizer(p_data).input_ids)
            n_static = len(static_toks)

            benign_inst_role_slice = slice(None, n_role1)
            benign_inst_slice      = slice(n_role1, n_inst)
            data_role_slice        = slice(n_inst, n_role2)
            data_slice             = slice(n_role2, n_data)
            inj_inst_slice         = slice(n_data, n_static)

            resp_toks = self.tokenizer(delm[2], add_special_tokens=False).input_ids
            nl_toks   = self.tokenizer("\n", add_special_tokens=False).input_ids
            toks = static_toks + sep_tokens + resp_toks + nl_toks

            assistant_role_slice = slice(n_static + num_tok_sep, len(toks))
            full_data_slice = slice(data_slice.start, inj_inst_slice.stop)
            adv_suffix_slice = None
            inj_inst_adv_suffix_slice = slice(inj_inst_slice.start, inj_inst_slice.stop)

        else:
            static_str = (
                f"{delm[0]}\n{instruction}\n\n{delm[1]}\n{data}"
            )
            static_toks = self.tokenizer(static_str).input_ids  # BOS auto-added

            p_role1 = f"{delm[0]}\n"
            p_inst  = p_role1 + instruction
            p_role2 = p_inst + f"\n\n{delm[1]}\n"
            n_role1 = len(self.tokenizer(p_role1).input_ids)
            n_inst  = len(self.tokenizer(p_inst).input_ids)
            n_role2 = len(self.tokenizer(p_role2).input_ids)
            n_static = len(static_toks)

            benign_inst_role_slice = slice(None, n_role1)
            benign_inst_slice      = slice(n_role1, n_inst)
            data_role_slice        = slice(n_inst, n_role2)
            data_slice             = slice(n_role2, n_static)

            resp_toks = self.tokenizer(delm[2], add_special_tokens=False).input_ids
            nl_toks   = self.tokenizer("\n", add_special_tokens=False).input_ids
            toks = static_toks + sep_tokens + resp_toks + nl_toks

            assistant_role_slice = slice(n_static + num_tok_sep, len(toks))
            full_data_slice = slice(data_slice.start, data_slice.stop)

            inj_inst_slice = None
            adv_suffix_slice = None
            inj_inst_adv_suffix_slice = None


        return toks, (benign_inst_slice, full_data_slice, inj_inst_slice, adv_suffix_slice, inj_inst_adv_suffix_slice)


    def inference(self, text=None, instruction=None, data=None, suffix=None, inj_prompt=False, max_output_tokens=None):

        if (instruction is not None) and (data is not None) and (suffix is not None):
            frontend_delimiters = "Meta-Llama-3-8B-Instruct"
            prompt_template = PROMPT_FORMAT[frontend_delimiters]["prompt_input"]

            text = prompt_template.format_map(
                {"instruction": instruction, 
                 "input": data + " " + TEST_INJECTED_PROMPT + " " + suffix}
            )
            input_ids, (benign_inst_slice, full_data_slice, inj_inst_slice, adv_suffix_slice, inj_inst_adv_suffix_slice) = self.get_input_ids(
                instruction, data, suffix
            )
            input_ids = torch.tensor([input_ids]).to(self.model.device)

            input_tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
            attention_mask = torch.ones_like(input_ids).to(self.model.device)

            # convert benign_inst_slice, full_data_slice, inj_inst_slice, adv_suffix_slice to tuple of positions
            data_range = (
                (benign_inst_slice.start, benign_inst_slice.stop),
                (full_data_slice.start, full_data_slice.stop),
                (inj_inst_slice.start, inj_inst_slice.stop),
                (adv_suffix_slice.start, adv_suffix_slice.stop),
                (inj_inst_adv_suffix_slice.start, inj_inst_adv_suffix_slice.stop),
            )

        elif (instruction is not None) and (data is not None) and inj_prompt:
            frontend_delimiters = "Meta-Llama-3-8B-Instruct"
            prompt_template = PROMPT_FORMAT[frontend_delimiters]["prompt_input"]

            text = prompt_template.format_map(
                {"instruction": instruction,
                 "input": data + " " + TEST_INJECTED_PROMPT}
            )
            input_ids, (benign_inst_slice, full_data_slice, inj_inst_slice, adv_suffix_slice, inj_inst_adv_suffix_slice) = self.get_input_ids(
                instruction, data, inj_prompt=True
            )
            input_ids = torch.tensor([input_ids]).to(self.model.device)

            input_tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
            attention_mask = torch.ones_like(input_ids).to(self.model.device)

            # Injected prompt present but no adversarial suffix — zero out the
            # suffix slice so process_attn returns zero for suffix components.
            data_range = (
                (benign_inst_slice.start, benign_inst_slice.stop),
                (full_data_slice.start, full_data_slice.stop),
                (inj_inst_slice.start, inj_inst_slice.stop),
                (0, 0),
                (inj_inst_adv_suffix_slice.start, inj_inst_adv_suffix_slice.stop),
            )

        elif (instruction is not None) and (data is not None):
            frontend_delimiters = "Meta-Llama-3-8B-Instruct"
            prompt_template = PROMPT_FORMAT[frontend_delimiters]["prompt_input"]

            text = prompt_template.format_map(
                {"instruction": instruction,
                 "input": data}
            )
            input_ids, (benign_inst_slice, full_data_slice, inj_inst_slice, adv_suffix_slice, inj_inst_adv_suffix_slice) = self.get_input_ids(
                instruction, data
            )
            input_ids = torch.tensor([input_ids]).to(self.model.device)

            input_tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
            attention_mask = torch.ones_like(input_ids).to(self.model.device)

            # No injected instruction / suffix in this prompt — pad with empty
            # slices so downstream process_attn (which always indexes rng[2..4])
            # produces zeros for those components.
            data_range = (
                (benign_inst_slice.start, benign_inst_slice.stop),
                (full_data_slice.start, full_data_slice.stop),
                (0, 0),
                (0, 0),
                (0, 0),
            )

        # elif instruction is not None and data is not None:
        #     messages = [
        #         {"role": "system", "content": instruction},
        #         {"role": "user", "content": "Data: " + data}
        #     ]

        #     # Use tokenization with minimal overhead
        #     text = self.tokenizer.apply_chat_template(
        #         messages,
        #         tokenize=False,
        #         add_generation_prompt=True
        #     )
        # elif text is not None:
        #     frontend_delimiters = "Meta-Llama-3-8B-Instruct"
        #     data_delm = DELIMITERS[frontend_delimiters][1]

        #     instruction, data = text.split(f"\n\n{data_delm}\n")

        #     instruction_len = len(self.tokenizer.encode(instruction))
        #     data_len = len(self.tokenizer.encode(data))

        #     model_inputs = self.tokenizer(
        #         [text], return_tensors="pt", padding="longest").to(self.model.device)
        #     input_tokens = self.tokenizer.convert_ids_to_tokens(
        #         model_inputs['input_ids'][0])
            
        #     data_range = (
        #         (5, 5+instruction_len),
        #         (-5-data_len, -5),
        #         (-5-20-8, -5-20),
        #         (-5-20, -5),
        #         (-5-20-8, -5),
        #     )

        #     input_ids = model_inputs.input_ids
        #     attention_mask = model_inputs.attention_mask

        # # find the data token positions
        # if "qwen" in self.name:
        #     data_range = ((3, 3+instruction_len), (-5-data_len, -5))
        # elif "phi3" in self.name:
        #     data_range = ((1, 1+instruction_len), (-2-data_len, -2))
        # elif "llama3-8b" in self.name:
        #     data_range = ((5, 5+instruction_len), (-5-data_len, -5))
        # elif "mistral-7b" in self.name:
        #     data_range = ((3, 3+instruction_len), (-1-data_len, -1))
        # elif "granite3-8b" in self.name:
        #     data_range = ((3, 3+instruction_len), (-5-data_len, -5))
        # else:
        #     raise NotImplementedError

        generated_tokens = []
        generated_probs = []

        attention_maps = []

        if max_output_tokens != None:
            n_tokens = max_output_tokens
        else:
            n_tokens = self.max_output_tokens

        with torch.no_grad():
            for i in range(n_tokens):
                output = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_attentions=True
                )

                logits = output.logits[:, -1, :]
                probs = F.softmax(logits, dim=-1)
                next_token_id = logits.argmax(dim=-1).squeeze()
                # next_token_id = sample_token(
                #     logits[0], top_k=self.top_k, top_p=self.top_p, temperature=1.0)[0]

                generated_probs.append(probs[0, next_token_id.item()].item())
                generated_tokens.append(next_token_id.item())

                if next_token_id.item() == self.tokenizer.eos_token_id:
                    break

                input_ids = torch.cat(
                    (input_ids, next_token_id.unsqueeze(0).unsqueeze(0)), dim=-1)
                attention_mask = torch.cat(
                    (attention_mask, torch.tensor([[1]], device=input_ids.device)), dim=-1)

                attention_map = [attention.detach().cpu().half()
                                 for attention in output['attentions']]
                attention_map = [torch.nan_to_num(
                    attention, nan=0.0) for attention in attention_map]
                attention_map = get_last_attn(attention_map)
                attention_maps.append(attention_map)

        output_tokens = [self.tokenizer.decode(
            token, skip_special_tokens=True) for token in generated_tokens]
        generated_text = self.tokenizer.decode(
            generated_tokens, skip_special_tokens=True)

        return generated_text, output_tokens, attention_maps, input_tokens, data_range, generated_probs
