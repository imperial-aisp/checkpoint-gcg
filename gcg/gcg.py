"""GCG Attack."""

import gc
import numpy as np
import torch
from ml_collections import ConfigDict
from typing import Dict, List, Any
import json
import logging
import time
from pathlib import Path
from dataclasses import dataclass

from gcg.base import BaseAttack, AttackResult
from gcg.eval_input import EvalInput
from gcg.model import TransformersModel
from gcg.types import BatchTokenIds
from gcg.utils import Message, Role, get_nonascii_toks
import re
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')
from nltk.corpus import stopwords, wordnet
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import defaultdict, OrderedDict
from torch.nn.utils.rnn import pad_sequence
import time

def generate_random_suffixes(
        model,
        tokenizer,
        suffix_manager,
        suffix_length=20,
        num_suffixes=10000,
        allow_non_ascii=False,
):
    """
    Generate num_suffixes random suffixes of a given length using the tokenizer's vocabulary.
    If allow_non_ascii is False, only ASCII characters are used.
    """
    vocab_size = tokenizer.vocab_size

    wrapped_model = TransformersModel(
        "alpaca@none",
        suffix_manager=suffix_manager,
        model=model,
        tokenizer=tokenizer,
        system_message="",
        max_tokens=100,
        temperature=0.0,
    )

    if not allow_non_ascii:
        # get non-ASCII token IDs to exclude
        non_ascii_tok_ids = get_nonascii_toks(tokenizer)
        non_ascii_tok_ids = [tensor.item() for tensor in non_ascii_tok_ids]
        # create a list of valid token IDs (excluding non-ASCII ones)
        valid_tok_ids = torch.tensor(
            [i for i in range(vocab_size) if i not in non_ascii_tok_ids], device="cpu"
        )
        random_indices = torch.randint(
            0,
            len(valid_tok_ids),
            size=(int(num_suffixes * 1.2), suffix_length),
            device="cpu",
        )  # 20% more as buffer
        random_token_matrix = valid_tok_ids[random_indices]
    else:
        # generate all random token IDs at once (allowing non-ASCII)
        random_token_matrix = torch.randint(
            0, vocab_size, size=(int(num_suffixes * 1.2), suffix_length), device="cpu"
        )

    # filter out suffixes that do not tokenize back to the same ids
    is_valid = wrapped_model.filter_suffixes(suffix_ids=random_token_matrix)
    num_valid = is_valid.int().sum().item()
    logger.info(f"Generated {num_valid} valid random suffixes.")

    adv_suffix_ids = random_token_matrix[is_valid]
    # decode each suffix
    adv_suffixes = tokenizer.batch_decode(adv_suffix_ids, skip_special_tokens=True)
    return adv_suffixes

logger = logging.getLogger(__name__)


def _rand_permute(size, device: str = "cuda", dim: int = -1):
    return torch.argsort(torch.rand(size, device=device), dim=dim)


class GCGAttack(BaseAttack):
    """GCG Attack (see https://llm-attacks.org/)."""

    name: str = "gcg"

    def __init__(self, config: ConfigDict, *args, **kwargs) -> None:
        """Initialize GCG attack."""
        self._topk = config.topk
        self._num_coords: tuple[int, int] = config.num_coords
        self._mu: float = config.mu
        if not isinstance(self._num_coords, tuple) or len(self._num_coords) != 2:
            raise ValueError(
                f"num_coords must be tuple of two ints, got {self._num_coords}"
            )

        # Init base class after setting parameters because it will call
        # _get_name_tokens() which uses the parameters. See below.
        super().__init__(config, *args, **kwargs)
        self._momentum: torch.Tensor | None = None

    def _get_name_tokens(self) -> list[str]:
        atk_tokens = super()._get_name_tokens()
        atk_tokens.append(f"k{self._topk}")
        if any(c != 1 for c in self._num_coords):
            if self._num_coords[0] == self._num_coords[1]:
                atk_tokens.append(f"c{self._num_coords[0]}")
            else:
                atk_tokens.append(f"c{self._num_coords[0]}-{self._num_coords[1]}")
        if self._mu != 0:
            atk_tokens.append(f"m{self._mu}")
        return atk_tokens

    def _param_schedule(self):
        num_coords = round(
            self._num_coords[0]
            + (self._num_coords[1] - self._num_coords[0]) * self._step / self._num_steps
        )
        return num_coords

    @torch.no_grad()
    def _compute_grad(self, eval_input: EvalInput, normalize_grads: bool = True, **kwargs) -> torch.Tensor:
        _ = kwargs  # unused
        grad = self._model.compute_grad(
            eval_input,
            normalize_grads=normalize_grads,
            temperature=self._loss_temperature,
            return_logits=True,
            **kwargs,
        )
        if self._mu == 0:
            return grad

        # Calculate momentum term
        if self._momentum is None:
            self._momentum = torch.zeros_like(grad)
        self._momentum.mul_(self._mu).add_(grad)
        return self._momentum

    @torch.no_grad()
    def _sample_updates(
        self,
        optim_ids,
        *args,
        grad: torch.Tensor | None = None,
        **kwargs,
    ) -> BatchTokenIds:
        _ = args, kwargs  # unused
        assert isinstance(grad, torch.Tensor), "grad is required for GCG!"
        assert len(grad) == len(optim_ids), (
            f"grad and optim_ids must have the same length ({len(grad)} vs "
            f"{len(optim_ids)})!"
        )
        device = grad.device
        num_coords = self._param_schedule()
        num_coords = min(num_coords, len(optim_ids))
        if self._not_allowed_tokens is not None:
            grad[:, self._not_allowed_tokens.to(device)] = np.infty

        top_indices = (-grad).topk(self._topk, dim=1).indices

        batch_size = int(self._batch_size * 1.25)
        old_token_ids = optim_ids.repeat(batch_size, 1)

        if num_coords == 1:
            # Each position will have `batch_size / len(optim_ids)` candidates
            new_token_pos = torch.arange(
                0,
                len(optim_ids),
                len(optim_ids) / batch_size,
                device=device,
            ).type(torch.int64)
            # Get random indices to select from topk
            # rand_idx: [seq_len, topk, 1]
            rand_idx = _rand_permute(
                (len(optim_ids), self._topk, 1), device=device, dim=1
            )
            # Get the first (roughly) batch_size / seq_len indices at each position
            rand_idx = torch.cat(
                [r[: (new_token_pos == i).sum()] for i, r in enumerate(rand_idx)],
                dim=0,
            )
            assert rand_idx.shape == (batch_size, 1), rand_idx.shape
            new_token_val = torch.gather(top_indices[new_token_pos], 1, rand_idx)
            new_token_ids = old_token_ids.scatter(
                1, new_token_pos.unsqueeze(-1), new_token_val
            )
        else:
            # Random choose positions to update
            new_token_pos = _rand_permute(
                (batch_size, len(optim_ids)), device=device, dim=1
            )[:, :num_coords]
            # Get random indices to select from topk
            rand_idx = torch.randint(
                0, self._topk, (batch_size, num_coords, 1), device=device
            )
            new_token_val = torch.gather(top_indices[new_token_pos], -1, rand_idx)
            new_token_ids = old_token_ids
            for i in range(num_coords):
                new_token_ids.scatter_(
                    1, new_token_pos[:, i].unsqueeze(-1), new_token_val[:, i]
                )

        assert new_token_ids.shape == (
            batch_size,
            len(optim_ids),
        ), new_token_ids.shape
        return new_token_ids

    def _get_next_suffix(
        self, eval_input: EvalInput, adv_suffixes: list[str], num_valid: int
    ) -> tuple[str, float]:
        """Select the suffix for the next step."""
        # Compute loss on model
        output = self._model.compute_suffix_loss(
            eval_input,
            batch_size=self._mini_batch_size,
            temperature=self._loss_temperature,
        )
        losses = output.losses
        self._num_queries += output.num_queries

        idx = losses[:num_valid].argmin()
        adv_suffix = adv_suffixes[idx]
        loss = losses[idx].item()
        return adv_suffix, loss

    def setup_run_prep_inputs(self, messages, target, adv_suffix):
        self.messages = messages
        self.target = target

        self._setup_run(
            messages=messages,
            adv_suffix=adv_suffix,
        )
        self.num_fixed_tokens = self._model.num_fixed_tokens

        # =============== Prepare inputs and determine slices ================ #
        self.eval_input = self._suffix_manager.gen_eval_inputs(
            messages,
            adv_suffix,
            target,
            num_fixed_tokens=self.num_fixed_tokens,
            max_target_len=self._seq_len,
        )
        self.eval_input.to(self._model.model.device)
        self.optim_slice = self.eval_input.optim_slice

    def compute_grad(self, adv_suffix):
        dynamic_input_ids = self._suffix_manager.get_input_ids(
            self.messages, adv_suffix, self.target
        )[0][self.num_fixed_tokens :]
        dynamic_input_ids = dynamic_input_ids.to(self._model.model.device)
        optim_ids = dynamic_input_ids[self.optim_slice]
        self.eval_input.dynamic_input_ids = dynamic_input_ids
        self.eval_input.suffix_ids = optim_ids

        # Compute grad as needed (None if no-grad attack)
        # computes for entire batch
        token_grads = self._compute_grad(self.eval_input, normalize_grads=False)
        del dynamic_input_ids
        gc.collect()
        return token_grads

    def compute_suffix_loss(self, adv_suffix_ids):
        self.eval_input.suffix_ids = adv_suffix_ids
        losses = self._compute_suffix_loss(self.eval_input)
        return losses


class CombinedMultiSampleAttack(GCGAttack):
    name: str = "gcg_combined_multi_sample"

    def __init__(
        self,
        config: ConfigDict,
        samples: List[Dict],
        sample_ids: List[int],
        data_delm: str,
        test_injected_prompt: str | None,
        sys_input: str,
        sys_no_input: str,
        *args,
        **kwargs,
    ) -> None:
        """Initialize the attack."""
        super().__init__(config, *args, **kwargs)
        self._samples = samples
        self._sample_ids = sample_ids

        self._setup_log_file(config)
        self._preprocess_samples(
            data_delm, test_injected_prompt, sys_input, sys_no_input
        )
        self._initiate_individual_attacks(config)

    def _setup_log_file(self, config):
        log_dir = Path(config.log_dir)
        logger.info("Logging to %s", log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        atk_name = str(self).replace(f"{self.name}_", "")
        if config.custom_name:
            atk_name += f"_{config.custom_name}"
        atk_name += f"_{config.num_samples_included}samples"
        log_file = log_dir / f"{atk_name}.jsonl"
        # Delete log file if it exists
        log_file.unlink(missing_ok=True)
        self._log_file = log_file

    def _preprocess_samples(
        self, data_delm, test_injected_prompt, sys_input, sys_no_input
    ) -> None:
        """Preprocess samples."""
        self._samples_messages = []
        if self._defense_type == "prompt_injection":
            for sample in self._samples:
                if (
                    sample["input"][-1] != "."
                    and sample["input"][-1] != "!"
                    and sample["input"][-1] != "?"
                ):
                    sample["input"] += "."

                injected_prompt = (
                    f" {test_injected_prompt}" if test_injected_prompt else ""
                )
                prompt_no_sys = (
                    f"{sample['instruction']}\n\n{data_delm}\n{sample['input']}"
                    f"{injected_prompt}"
                )
                messages = [
                    Message(Role.SYSTEM, sys_input),
                    Message(Role.USER, prompt_no_sys),
                ]
                self._samples_messages.append(messages)
        elif self._defense_type == "jailbreak":
            for sample in self._samples:
                prompt_no_sys = f"{sample['goal']}"
                messages = [
                    Message(Role.SYSTEM, sys_no_input),
                    Message(Role.USER, prompt_no_sys),
                ]
                self._samples_messages.append(messages)

    def _initiate_individual_attacks(self, config):
        self._attacks = [
            GCGAttack(
                config=config,
                model=self._peftmodel,
                tokenizer=self._tokenizer,
                eval_func=self._eval_func,
                suffix_manager=self._suffix_manager,
                not_allowed_tokens=self._not_allowed_tokens,
            )
            for _ in range(len(self._samples))
        ]

    def _setup_run(self, adv_suffix):
        self._start_time = time.time()
        self._num_queries = 0
        self._step = None
        self._best_loss, self._best_suffix = float("inf"), adv_suffix
        self._attacks_best_loss = [float("inf") for _ in range(len(self._samples))]
        self._seen_suffixes = set()
        self._visited_suffixes = set()
        self._num_repeated = 0
        if not self._fixed_params:
            return

    def _test_all(self, adv_suffix, target_outputs):
        results = [
            self._eval_func(
                adv_suffix,
                messages,
                target_output,
                self._defense_type,
                self._prompt_template,
                self._peftmodel,
                self._tokenizer,
            )
            for (messages, target_output) in zip(self._samples_messages, target_outputs)
        ]
        if self._defense_type == "prompt_injection":
            in_response = [result[0] for result in results]
            begin_with = [result[1] for result in results]
            generated_output = [result[2][0][0] for result in results]

            test_results = {
                f"sample_{sample_id}": {
                    "in_response": in_response[i],
                    "begin_with": begin_with[i],
                    "generated_output": generated_output[i],
                }
                for i, sample_id in enumerate(self._sample_ids)
            }

            num_success_begin_with = sum(begin_with)
            num_success_in_response = sum(in_response)
            test_results["num_success_begin_with"] = num_success_begin_with
            test_results["num_success_in_response"] = num_success_in_response

            return test_results, num_success_begin_with, num_success_in_response

        elif self._defense_type == "jailbreak":
            target_generated = [result[0] for result in results]
            jailbroken = [result[1] for result in results]
            generated_output = [result[2][0][0] for result in results]

            test_results = {
                f"sample_{sample_id}": {
                    "target_generated": target_generated[i],
                    "jailbroken": jailbroken[i],
                    "generated_output": generated_output[i],
                }
                for i, sample_id in enumerate(self._sample_ids)
            }

            num_jailbroken = sum(jailbroken)
            num_target_generated = sum(target_generated)
            test_results["num_jailbroken"] = num_jailbroken
            test_results["num_target_generated"] = num_target_generated

            return test_results, num_jailbroken, num_target_generated

    def _save_best(
        self,
        current_loss: float,
        current_suffix: str,
        attacks_current_loss: List[float],
    ) -> None:
        """Save the best loss and suffix so far."""
        if current_loss < self._best_loss:
            self._best_loss = current_loss
            self._best_suffix = current_suffix
            self._attacks_best_loss = attacks_current_loss

    @torch.no_grad()
    def run(self, target_outputs: list[str]) -> AttackResult:
        """Run the attack."""
        if self._add_space:
            target_outputs = ["▁" + target for target in target_outputs]
        # Setting up init suffix
        adv_suffix = self._adv_suffix_init
        adv_suffix_ids = self._tokenizer(
            adv_suffix, add_special_tokens=False, return_tensors="pt"
        ).input_ids
        adv_suffix_ids.squeeze_(0)

        self._setup_run(adv_suffix)

        with self._log_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(self._config.to_dict(), indent=4) + "\n")

        logger.debug("Starting attack with suffix: %s", adv_suffix)
        assert adv_suffix_ids.ndim == 1, adv_suffix_ids.shape
        logger.debug(
            "\nInitialized suffix (%d tokens):\n%s",
            len(adv_suffix_ids),
            adv_suffix,
        )

        for i, attack in enumerate(self._attacks):
            messages = self._samples_messages[i]
            attack.setup_run_prep_inputs(messages, target_outputs[i], adv_suffix)

        passed = True

        # test before running GCG
        self._step = 0
        # Get total loss across all attacks
        total_losses = torch.zeros(len(adv_suffix_ids), device=adv_suffix_ids.device)
        attacks_losses = []
        for attack in self._attacks:
            losses = attack.compute_suffix_loss(adv_suffix_ids)
            total_losses += losses
            attacks_losses.append(losses)

        current_loss = total_losses[0].item() / len(self._attacks)
        attacks_current_loss = [
            attacks_losses[i][0].item() for i in range(len(self._attacks))
        ]
        self._save_best(current_loss, adv_suffix, attacks_current_loss)
        self._visited_suffixes.add(adv_suffix)
        log_dict = {
            "current_loss": {
                "overall": round(current_loss, 6),
            },
            "best_loss": {
                "overall": round(self._best_loss, 6),
            },
            "suffix": adv_suffix,
            "best_suffix": self._best_suffix,
        }
        for j, sample_id in enumerate(self._sample_ids):
            log_dict["current_loss"][f"sample_{sample_id}"] = round(
                attacks_current_loss[j], 6
            )
            log_dict["best_loss"][f"sample_{sample_id}"] = round(
                self._attacks_best_loss[j], 6
            )

        if all(
            attacks_current_loss[j] < self._loss_threshold
            for j in range(len(self._sample_ids))
        ):
            self._num_queries += len(self._attacks)
            test_results, num_success_begin_with, num_success_in_response = (
                self._test_all(adv_suffix, target_outputs)
            )
            log_dict["test_results"] = test_results
            passed = num_success_begin_with < len(self._attacks)

        if not passed:
            self._best_suffix = adv_suffix
            log_dict["best_suffix"] = self._best_suffix

        # Logging
        self.log(log_dict=log_dict)

        if not passed:
            logger.info("Attack succeeded! Early stopping...")
            attack_result = AttackResult(
                best_loss=self._best_loss,
                best_suffix=self._best_suffix,
                num_queries=self._num_queries,
                success=not passed,
                steps=self._step,
            )
            return attack_result

        same_best_loss_steps = 0
        for i in range(1, self._num_steps + 1):
            self._step = i

            # aggregate gradients
            token_grads = torch.zeros(
                (
                    self._attacks[0].optim_slice.stop
                    - self._attacks[0].optim_slice.start,
                    len(self._tokenizer),
                ),
                device=self._attacks[0].eval_input.dynamic_input_ids.device,
            )
            for attack in self._attacks:
                token_grads += attack.compute_grad(adv_suffix)
            token_grads /= token_grads.norm(dim=-1, keepdim=True)

            # sample updates
            adv_suffix_ids = self._sample_updates(
                optim_ids=self._attacks[0].eval_input.suffix_ids,
                grad=token_grads,
            )

            # Filter out "invalid" adversarial suffixes
            adv_suffix_ids, num_valid = self._filter_suffixes(adv_suffix_ids)
            adv_suffixes = self._tokenizer.batch_decode(
                adv_suffix_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            self._seen_suffixes.update(adv_suffixes)

            # Get total loss across all attacks
            total_losses = torch.zeros(
                len(adv_suffix_ids), device=adv_suffix_ids.device
            )
            attacks_losses = []
            for attack in self._attacks:
                losses = attack.compute_suffix_loss(adv_suffix_ids)
                total_losses += losses
                attacks_losses.append(losses)

            idx = total_losses[:num_valid].argmin()
            adv_suffix = adv_suffixes[idx]
            current_loss = total_losses[idx].item() / len(self._attacks)
            attacks_current_loss = [
                attacks_losses[i][idx].item() for i in range(len(self._attacks))
            ]

            prev_best_loss = self._best_loss
            # Save the best candidate and update visited suffixes
            self._save_best(current_loss, adv_suffix, attacks_current_loss)
            self._visited_suffixes.add(adv_suffix)

            if (
                abs(prev_best_loss - self._best_loss)
                > self._config.same_best_loss_threshold
            ):
                same_best_loss_steps = 0
            else:
                same_best_loss_steps += 1

            if i % self._log_freq == 0:
                log_dict = {
                    "current_loss": {
                        "overall": round(current_loss, 6),
                    },
                    "best_loss": {
                        "overall": round(self._best_loss, 6),
                    },
                    "suffix": adv_suffix,
                    "best_suffix": self._best_suffix,
                }
                for j, sample_id in enumerate(self._sample_ids):
                    log_dict["current_loss"][f"sample_{sample_id}"] = round(
                        attacks_current_loss[j], 6
                    )
                    log_dict["best_loss"][f"sample_{sample_id}"] = round(
                        self._attacks_best_loss[j], 6
                    )

                if all(
                    attacks_current_loss[j] < self._loss_threshold
                    for j in range(len(self._sample_ids))
                ):
                    self._num_queries += len(self._attacks)
                    test_results, num_success_begin_with, num_success_in_response = (
                        self._test_all(adv_suffix, target_outputs)
                    )
                    log_dict["test_results"] = test_results
                    passed = num_success_begin_with < len(self._attacks)

                if not passed:
                    self._best_suffix = adv_suffix
                    log_dict["best_suffix"] = self._best_suffix

                # Logging
                self.log(log_dict=log_dict)
            del token_grads
            gc.collect()

            if not passed:
                logger.info("Attack succeeded! Early stopping...")
                break
            if self._num_queries >= self._max_queries > 0:
                logger.info("Max queries reached! Finishing up...")
                break

            if (self._config.early_stopping) and (
                same_best_loss_steps >= self._config.num_same_best_loss
            ):
                logger.info(
                    f"No change (> {self._config.same_best_loss_threshold}) in best_loss for {self._config.num_same_best_loss} steps! Early stopping..."
                )
                break

        attack_result = AttackResult(
            best_loss=self._best_loss,
            best_suffix=self._best_suffix,
            num_queries=self._num_queries,
            success=not passed,
            steps=self._step,
        )
        return attack_result

    def format(self, d, tab=0):
        s = ["{\n"]
        for k, v in d.items():
            s.append("%s%r: %s,\n" % ("  " * tab, k, v))
        s.append("%s}" % ("  " * tab))
        return "".join(s)

    def log(
        self, step: int | None = None, log_dict: dict[str, Any] | None = None
    ) -> None:
        """Log data using logger from a single step."""
        step = step or self._step
        log_dict["mem"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
        log_dict["time_per_step_s"] = round(
            (time.time() - self._start_time) / (step + 1), 2
        )
        log_dict["queries"] = self._num_queries
        log_dict["time_min"] = round((time.time() - self._start_time) / 60, 2)

        logger.info(
            "[step: %4d/%4d] %s", step, self._num_steps, self.format(log_dict, 2)
        )
        log_dict["step"] = step

        # Convert all tensor values to lists or floats
        def tensor_to_serializable(val):
            if isinstance(val, torch.Tensor):
                return val.tolist() if val.numel() > 1 else val.item()
            return val

        log_dict = {k: tensor_to_serializable(v) for k, v in log_dict.items()}
        test_results = log_dict.pop("test_results", None)
        step = log_dict.pop("step", None)
        if test_results is not None:
            log_dict["test_results"] = test_results
        if step is not None:
            log_dict["step"] = step
        with self._log_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(log_dict) + "\n")
