"""
BEAST attack adapted for checkpoint-assisted attacks on SecAlign models.

Based on "Fast Adversarial Attacks on Language Models In One GPU Minute"
(https://arxiv.org/abs/2402.15570), but adapted to a fixed-length beam search
so it slots into the checkpoint framework in test.py:

  * Beam width k1 of length-L suffixes (token IDs).
  * At each step, for every candidate, generate k2 neighbors by picking a random
    position and resampling that token from the sampler model conditioned on
    [static_prefix, candidate[:position]] (top-p / temperature).
  * Score all k1*k2 neighbors by CE loss of the target response on the target 
    model with the static prefix kv-cached.
  * Keep the top-k1 neighbors; repeat for config.num_steps iterations.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from ml_collections import ConfigDict

from gcg.base import AttackResult
from gcg.utils import Message, SuffixManager, get_prefix_cache


logger = logging.getLogger(__name__)


def _sample_top_p(logits: torch.Tensor, top_p: float, num_samples: int = 1) -> torch.Tensor:
    """Returning token IDs of shape (batch, num_samples).

    Matches the behavior of BEAST's reference implementation.
    """
    probs = F.softmax(logits, dim=-1)
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    cum = torch.cumsum(probs_sort, dim=-1)
    mask = cum - probs_sort > top_p
    probs_sort = probs_sort.masked_fill(mask, 0.0)
    probs_sort = probs_sort / probs_sort.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    sampled = torch.multinomial(probs_sort, num_samples=max(1, num_samples))
    return probs_idx.gather(-1, sampled)


class BEASTAttack:

    name: str = "beast"

    def __init__(
        self,
        config: ConfigDict,
        model,
        tokenizer,
        suffix_manager: SuffixManager,
        eval_func,
        sampler_model,
        sampler_device: torch.device | str,
        initial_beam_suffixes: list[str] | None = None,
        not_allowed_tokens: torch.Tensor | None = None,
        **kwargs,
    ) -> None:
        _ = kwargs, not_allowed_tokens
        self._config = config
        self._model = model
        self._tokenizer = tokenizer
        self._suffix_manager = suffix_manager
        self._eval_func = eval_func
        self._sampler_model = sampler_model
        self._sampler_device = torch.device(sampler_device)

        self._defense_type = config.defense_type
        self._prompt_template = config.prompt_template
        self._adv_suffix_init: str = config.adv_suffix_init
        self._initial_beam_suffixes: list[str] = list(initial_beam_suffixes or [])

        # BEAST hyperparameters
        self._L: int = config.beast_suffix_length
        self._k1: int = config.beast_k1
        self._k2: int = config.beast_k2
        self._top_p: float = config.beast_top_p
        self._temperature: float = config.beast_temperature
        self._score_batch_size: int = config.beast_score_batch_size

        # Budget / logging (mirrors GCG conventions)
        self._num_steps: int = config.num_steps
        self._seq_len: int = config.seq_len
        self._loss_threshold: float = config.loss_threshold_for_output_gen
        self._log_freq: int = config.log_freq
        self._early_stopping: bool = config.early_stopping
        self._num_same_best_loss: int = config.beast_num_same_best_loss
        self._same_best_loss_threshold: float = config.same_best_loss_threshold

        self._peftmodel = model  # alias used by _test_suffix
        self._device = next(model.parameters()).device

        self._start_time: float | None = None
        self._step: int = 0
        self._best_loss: float = float("inf")
        self._best_suffix: str = self._adv_suffix_init
        self._num_queries: int = 0

    def _setup_log_file(self, config: ConfigDict) -> None:
        """Mirror GCG _setup_log_file so test.py's reader picks up the same filenames."""
        log_dir = Path(config.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        if hasattr(config, "lora_scale") and config.lora_scale is not None:
            log_file = log_dir / f"scale_{config.lora_scale}.jsonl"
        elif config.checkpoint == -1:
            log_file = log_dir / f"{config.sample_id}.jsonl"
        else:
            log_file = log_dir / f"checkpoint_{config.checkpoint}.jsonl"
        log_file.unlink(missing_ok=True)
        self._log_file = log_file

    @torch.no_grad()
    def _build_layout(self, messages: list[Message], target: str):
        """Resolve input_ids layout (static_prefix + L-token suffix + post_suffix).

        We call `SuffixManager.get_input_ids` once with a placeholder suffix
        that is guaranteed to tokenize to exactly L tokens. The returned slices
        then stay valid as long as we substitute any L-token suffix into
        optim_slice.

        Returns:
            full_input_ids_template: 1D tensor — input_ids with placeholder
                suffix, target tokens included up to seq_len.
            static_input_ids: 1D tensor — the suffix_manager's static prefix.
            optim_slice, target_slice, loss_slice: slices into full template.
        """
        placeholder_str, placeholder_ids = self._find_placeholder_string()
        out = self._suffix_manager.get_input_ids(messages, placeholder_str, target)
        full_input_ids, optim_slice, target_slice, loss_slice = out
        actual_len = optim_slice.stop - optim_slice.start
        if actual_len != self._L:
            raise RuntimeError(
                f"BEAST placeholder tokenized to {actual_len} tokens but L={self._L}. "
                f"Placeholder text was: {placeholder_str!r}. This can happen when "
                f"BPE merges adjacent atoms; try a different `--beast_suffix_length`."
            )
        full_input_ids = full_input_ids.clone()
        full_input_ids[optim_slice] = torch.tensor(
            placeholder_ids, dtype=full_input_ids.dtype
        )

        # Truncate target to seq_len — matches GCG
        if (target_slice.stop - target_slice.start) > self._seq_len:
            target_slice = slice(
                target_slice.start, target_slice.start + self._seq_len
            )
            loss_slice = slice(
                loss_slice.start, loss_slice.start + self._seq_len
            )
            full_input_ids = full_input_ids[: target_slice.stop]

        static_input_ids = self._suffix_manager.get_input_ids(
            messages, static_only=True
        )
        return (
            full_input_ids.to(self._device),
            static_input_ids.to(self._device),
            optim_slice,
            target_slice,
            loss_slice,
        )

    def _find_placeholder_string(self) -> tuple[str, list[int]]:
        """Build a placeholder suffix string that tokenizes to exactly L tokens.

        Strategy: sample L random token IDs from the vocab, decode them, and
        verify the round-trip re-tokenizes to exactly L tokens. BPE can merge
        adjacent tokens on decode→retokenize, so we retry until we find a
        sample that round-trips cleanly. The placeholder only needs to set up
        the slice layout — the actual suffix tokens are substituted later.
        """
        vocab_size = len(self._tokenizer)
        special_ids = set(self._tokenizer.all_special_ids or [])
        max_attempts = 200
        for _ in range(max_attempts):
            ids = torch.randint(0, vocab_size, (self._L,)).tolist()
            if any(i in special_ids for i in ids):
                continue
            s = self._tokenizer.decode(ids, skip_special_tokens=True)
            retok = self._tokenizer(s, add_special_tokens=False).input_ids
            if len(retok) == self._L:
                return s, list(retok)
        raise RuntimeError(
            f"BEAST: could not find a random placeholder of length L={self._L} "
            f"that round-trips cleanly after {max_attempts} attempts. Try a "
            f"different `--beast_suffix_length`."
        )

    @torch.no_grad()
    def _sampler_prefix_cache(self, static_input_ids: torch.Tensor):
        """KV-cache the static prefix on the sampler model."""
        inp = static_input_ids.to(self._sampler_device).unsqueeze(0)
        out = self._sampler_model(input_ids=inp, use_cache=True)
        # Detach and unwrap to legacy tuple-of-tuples so each forward gets a
        # fresh DynamicCache
        past = [(k.detach(), v.detach()) for k, v in out.past_key_values]
        last_logits = out.logits[:, -1, :]
        return past, last_logits

    @torch.no_grad()
    def _sample_at_position(
        self,
        candidate_ids: torch.Tensor,
        position: int,
        sampler_cache,
        sampler_last_logits: torch.Tensor,
    ) -> int:
        """Sample one replacement token at `position` in `candidate_ids`.

        Uses [static_prefix, candidate[:position]] as conditioning, via the
        cached sampler prefix + a short dynamic forward.
        """
        if position == 0:
            # First-position replacement: condition only on the static prefix,
            # i.e., reuse the last logits from the cache-building forward
            logits = sampler_last_logits / max(self._temperature, 1e-6)
            token = _sample_top_p(logits, self._top_p, num_samples=1)
            return int(token.item())

        context = candidate_ids[:position].to(self._sampler_device).unsqueeze(0)
        out = self._sampler_model(
            input_ids=context,
            past_key_values=sampler_cache,
            use_cache=True,
        )
        logits = out.logits[:, -1, :] / max(self._temperature, 1e-6)
        token = _sample_top_p(logits, self._top_p, num_samples=1)
        return int(token.item())

    @torch.no_grad()
    def _sample_autoregressive(
        self,
        prefix_suffix_ids: torch.Tensor,
        num_tokens: int,
        sampler_cache,
        sampler_last_logits: torch.Tensor,
    ) -> list[int]:
        """Autoregressively sample `num_tokens` tokens after [static, prefix_suffix]."""
        generated: list[int] = []
        if num_tokens <= 0:
            return generated

        if prefix_suffix_ids.numel() == 0:
            # Kick off from the static prefix's last logits
            logits = sampler_last_logits / max(self._temperature, 1e-6)
            first = _sample_top_p(logits, self._top_p, num_samples=1)
            first_id = int(first.item())
            generated.append(first_id)
            running = torch.tensor([first_id], device=self._sampler_device)
            past = sampler_cache
        else:
            running = prefix_suffix_ids.to(self._sampler_device)
            past = sampler_cache

        # Walk forward token-by-token. We pass the whole `running` tail on the
        # first forward (to consume the provided prefix) and then single tokens.
        remaining = num_tokens - len(generated)
        # First forward consumes either the provided prefix, or the one seed
        # token we just sampled.
        first_forward_input = running.unsqueeze(0)
        out = self._sampler_model(
            input_ids=first_forward_input,
            past_key_values=past,
            use_cache=True,
        )
        past = out.past_key_values
        for _ in range(remaining):
            logits = out.logits[:, -1, :] / max(self._temperature, 1e-6)
            nxt = _sample_top_p(logits, self._top_p, num_samples=1)
            nxt_id = int(nxt.item())
            generated.append(nxt_id)
            out = self._sampler_model(
                input_ids=nxt.to(self._sampler_device),
                past_key_values=past,
                use_cache=True,
            )
            past = out.past_key_values
        return generated

    @torch.no_grad()
    def _seed_initial_beam(
        self,
        sampler_cache,
        sampler_last_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Return a (k1, L) tensor of initial suffix token IDs.

        Strategy: take the last k1 suffix strings from the previous checkpoint's
        log (passed in via `initial_beam_suffixes`). Each is (re)tokenized with
        the target tokenizer and either truncated to the last L tokens or
        padded with sampler-generated continuations. If fewer than k1 prior
        suffixes are available, fill the remainder with fresh autoregressive
        samples from the sampler.
        """
        beam: list[list[int]] = []

        for s in self._initial_beam_suffixes[: self._k1]:
            ids = self._tokenizer(s, add_special_tokens=False).input_ids
            if len(ids) >= self._L:
                ids = ids[-self._L :]
            else:
                need = self._L - len(ids)
                pad = self._sample_autoregressive(
                    prefix_suffix_ids=torch.tensor(ids, dtype=torch.long),
                    num_tokens=need,
                    sampler_cache=sampler_cache,
                    sampler_last_logits=sampler_last_logits,
                )
                ids = list(ids) + list(pad)
            beam.append(list(ids))

        while len(beam) < self._k1:
            ids = self._sample_autoregressive(
                prefix_suffix_ids=torch.tensor([], dtype=torch.long),
                num_tokens=self._L,
                sampler_cache=sampler_cache,
                sampler_last_logits=sampler_last_logits,
            )
            beam.append(list(ids))

        return torch.tensor(beam, dtype=torch.long, device=self._device)

    @torch.no_grad()
    def _score_suffixes(
        self,
        suffix_ids_batch: torch.Tensor,  # (N, L)
        full_input_ids_template: torch.Tensor,  # (T,)
        optim_slice: slice,
        target_slice: slice,
        loss_slice: slice,
        prefix_cache,
        num_fixed_tokens: int,
    ) -> torch.Tensor:
        """Return per-candidate CE losses (shape (N,))."""
        N = suffix_ids_batch.shape[0]
        T = full_input_ids_template.shape[0]
        losses = torch.empty(N, device=self._device)

        target_ids_template = full_input_ids_template[target_slice].to(self._device)
        optim_slice_dyn_start = optim_slice.start - num_fixed_tokens
        optim_slice_dyn_stop = optim_slice.stop - num_fixed_tokens
        target_slice_dyn = slice(
            target_slice.start - num_fixed_tokens,
            target_slice.stop - num_fixed_tokens,
        )
        loss_slice_dyn = slice(
            loss_slice.start - num_fixed_tokens,
            loss_slice.stop - num_fixed_tokens,
        )

        dynamic_template = full_input_ids_template[num_fixed_tokens:]  # (T - num_fixed,)

        for start in range(0, N, self._score_batch_size):
            end = min(start + self._score_batch_size, N)
            bs = end - start

            inp = dynamic_template.unsqueeze(0).repeat(bs, 1).clone()
            inp[:, optim_slice_dyn_start:optim_slice_dyn_stop] = suffix_ids_batch[start:end]

            batch_cache = [
                (k.expand(bs, -1, -1, -1),
                 v.expand(bs, -1, -1, -1))
                for k, v in prefix_cache
            ]
            out = self._model(
                input_ids=inp,
                past_key_values=batch_cache,
                use_cache=True,
            )
            logits = out.logits  # (bs, dyn_len, V)

            batch_target_ids = target_ids_template.unsqueeze(0).expand(bs, -1)
            loss_logits = logits[:, loss_slice_dyn, :]
            loss = F.cross_entropy(
                loss_logits.reshape(-1, loss_logits.size(-1)),
                batch_target_ids.reshape(-1),
                reduction="none",
            )
            loss = loss.view(bs, -1).mean(dim=1)
            losses[start:end] = loss
            self._num_queries += bs

        return losses

    @torch.no_grad()
    def _generate_neighbors(
        self,
        beam: torch.Tensor,  # (k1, L)
        sampler_cache,
        sampler_last_logits: torch.Tensor,
    ) -> torch.Tensor:
        """For each of k1 candidates, spin off k2 neighbors (one random-position resample each).

        Returns a (k1*k2, L) tensor, ordered so that neighbors[c*k2 : (c+1)*k2]
        are the children of candidate c.
        """
        k1, L = beam.shape
        neighbors = beam.unsqueeze(1).repeat(1, self._k2, 1).clone()  # (k1, k2, L)
        for c in range(k1):
            candidate = beam[c]
            for j in range(self._k2):
                i = int(torch.randint(0, L, (1,)).item())
                new_tok = self._sample_at_position(
                    candidate, i, sampler_cache, sampler_last_logits
                )
                neighbors[c, j, i] = new_tok
        return neighbors.view(k1 * self._k2, L)

    def _test_suffix(
        self,
        messages: list[Message],
        target_output: str,
        adv_suffix: str,
        log_dict: dict,
    ):
        self._num_queries += 1
        result = self._eval_func(
            adv_suffix,
            messages,
            target_output,
            self._defense_type,
            self._prompt_template,
            self._peftmodel,
            self._tokenizer,
        )
        passed = result[1] == 0
        if self._defense_type == "prompt_injection":
            log_dict["success_begin_with"] = result[1] == 1
            log_dict["success_in_response"] = result[0] == 1
            log_dict["generated"] = result[2][0][0]
        elif self._defense_type == "jailbreak":
            log_dict["jailbroken"] = result[1] == 1
            log_dict["target_generated"] = result[0] == 1
            log_dict["generated"] = result[2][0][0]
        return passed, log_dict

    @torch.no_grad()
    def run(self, messages: list[Message], target: str) -> AttackResult:
        self._start_time = time.time()
        self._num_queries = 0
        self._best_loss = float("inf")
        self._best_suffix = self._adv_suffix_init

        with self._log_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(self._config.to_dict(), indent=4) + "\n")

        # Layout + static-prefix kv cache on the target model
        full_input_ids_template, static_input_ids, optim_slice, target_slice, loss_slice = (
            self._build_layout(messages, target)
        )
        prefix_cache, num_fixed_tokens = get_prefix_cache(
            self._suffix_manager, self._model, self._tokenizer, messages
        )
        prefix_cache = [(k.detach(), v.detach()) for k, v in prefix_cache]
        # Sanity: _build_layout's static_input_ids length should match what
        # get_prefix_cache cached. If it doesn't, downstream slice arithmetic
        # silently goes wrong
        if num_fixed_tokens != static_input_ids.shape[0]:
            logger.warning(
                "BEAST: num_fixed_tokens (%d) != static_input_ids length (%d); "
                "using num_fixed_tokens for slice offsets.",
                num_fixed_tokens,
                static_input_ids.shape[0],
            )

        # Static-prefix kv cache on the sampler model
        sampler_cache, sampler_last_logits = self._sampler_prefix_cache(static_input_ids)

        # Seed the beam
        beam = self._seed_initial_beam(sampler_cache, sampler_last_logits)

        # Initial scoring + step-0 logging (mirrors GCG)
        beam_losses = self._score_suffixes(
            beam,
            full_input_ids_template,
            optim_slice,
            target_slice,
            loss_slice,
            prefix_cache,
            num_fixed_tokens,
        )

        best_idx = int(torch.argmin(beam_losses).item())
        best_suffix_ids = beam[best_idx]
        best_suffix_str = self._tokenizer.decode(
            best_suffix_ids.tolist(), skip_special_tokens=True
        )
        self._best_loss = float(beam_losses[best_idx].item())
        self._best_suffix = best_suffix_str

        self._step = 0
        passed = True
        log_dict = {
            "loss": self._best_loss,
            "best_loss": self._best_loss,
            "suffix": best_suffix_str,
            "best_suffix": best_suffix_str,
            "beam": self._decode_beam(beam),
            "beam_losses": beam_losses.detach().cpu().tolist(),
        }
        if self._best_loss < self._loss_threshold:
            passed, log_dict = self._test_suffix(
                messages, target, best_suffix_str, log_dict
            )
        self.log(log_dict=log_dict, config=self._config)
        if not passed:
            logger.info("BEAST attack succeeded at init! Early stopping...")
            return AttackResult(
                best_loss=self._best_loss,
                best_suffix=self._best_suffix,
                num_queries=self._num_queries,
                success=not passed,
                steps=self._step,
            )

        same_best_loss_steps = 0

        for iteration in range(1, self._num_steps + 1):
            self._step = iteration

            # 1. Generate k1 * k2 neighbors
            neighbors = self._generate_neighbors(
                beam, sampler_cache, sampler_last_logits
            )

            # 2. Score them on the target model
            neighbor_losses = self._score_suffixes(
                neighbors,
                full_input_ids_template,
                optim_slice,
                target_slice,
                loss_slice,
                prefix_cache,
                num_fixed_tokens,
            )

            # 3. Top-k1 selection
            topk_idx = torch.topk(neighbor_losses, k=self._k1, largest=False).indices
            beam = neighbors[topk_idx].clone()
            beam_losses = neighbor_losses[topk_idx].clone()

            # 4. Track best-of-beam and log
            best_idx = int(torch.argmin(beam_losses).item())
            best_suffix_ids = beam[best_idx]
            best_suffix_str = self._tokenizer.decode(
                best_suffix_ids.tolist(), skip_special_tokens=True
            )
            current_loss = float(beam_losses[best_idx].item())

            prev_best_loss = self._best_loss
            if current_loss < self._best_loss:
                self._best_loss = current_loss
                self._best_suffix = best_suffix_str
            if prev_best_loss - self._best_loss > self._same_best_loss_threshold:
                same_best_loss_steps = 0
            else:
                same_best_loss_steps += 1

            log_dict = {
                "loss": current_loss,
                "best_loss": self._best_loss,
                "suffix": best_suffix_str,
                "best_suffix": self._best_suffix,
                "beam": self._decode_beam(beam),
                "beam_losses": beam_losses.detach().cpu().tolist(),
            }
            if current_loss < self._loss_threshold:
                passed, log_dict = self._test_suffix(
                    messages, target, best_suffix_str, log_dict
                )
            self.log(log_dict=log_dict, config=self._config)

            if not passed:
                logger.info("BEAST attack succeeded! Early stopping...")
                break

            if (
                self._early_stopping
                and same_best_loss_steps >= self._num_same_best_loss
            ):
                logger.info(
                    "No change (> %g) in best_loss for %d iterations! Early stopping...",
                    self._same_best_loss_threshold,
                    self._num_same_best_loss,
                )
                break

        return AttackResult(
            best_loss=self._best_loss,
            best_suffix=self._best_suffix,
            num_queries=self._num_queries,
            success=not passed,
            steps=self._step,
        )

    def _decode_beam(self, beam: torch.Tensor) -> list[str]:
        return self._tokenizer.batch_decode(
            beam.tolist(), skip_special_tokens=True
        )

    def log(
        self,
        step: int | None = None,
        log_dict: dict[str, Any] | None = None,
        config: ConfigDict | None = None,
    ) -> None:
        step = step if step is not None else self._step
        log_dict["mem"] = torch.cuda.max_memory_allocated() / 1e9
        log_dict["time_per_step_s"] = (time.time() - self._start_time) / max(step, 1)
        log_dict["queries"] = self._num_queries
        log_dict["time_min"] = (time.time() - self._start_time) / 60
        log_dict["sample_id"] = config.sample_id
        log_dict["step"] = step

        def _serialize(val):
            if isinstance(val, torch.Tensor):
                return val.tolist() if val.numel() > 1 else val.item()
            return val

        log_dict = {k: _serialize(v) for k, v in log_dict.items()}
        logger.info(
            "[beast iter: %4d/%4d] loss=%.4f best=%.4f",
            step,
            self._num_steps,
            log_dict.get("loss", float("nan")),
            log_dict.get("best_loss", float("nan")),
        )
        with self._log_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(log_dict) + "\n")
