"""Score the five preference labels by one forward pass, for HF models.

Why this exists. llama.cpp 9307 cannot load every architecture we want to judge
with (Laguna S 2.1 fails with "unknown model architecture"), so the cross-vendor
replication needs a transformers path. Generating with constrained decoding under
transformers would be slow and would need a second grammar implementation.

Why it is comparable to the llama.cpp runs. We established empirically that a judge
asked for a bare digit disagrees with the same judge asked for a JSON object on
53% of items -- the scaffolding before the answer changes the answer. So this
scorer does not ask for a bare digit. It prefills the assistant turn with the exact
JSON prefix the constrained decoder would have emitted, and reads the next-token
distribution restricted to the five digits. That is the same conditional
distribution the constrained decoder samples from at that position.

That argument is a hypothesis, not a licence: `c3_prefill_equivalence.py` checks it
against recorded llama.cpp labels before any cross-vendor number is reported.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# The exact string llama-server's schema-constrained decode produced, minus the digit.
JSON_PREFIX = '{\n  "p": "'


@dataclass
class ScoredItem:
    comparison_id: str
    order: str
    code: str
    probs: dict[str, float]


class HFPreferenceScorer:
    def __init__(self, model_path: str, *, dtype: torch.dtype = torch.bfloat16,
                 device: str = "cuda", codes: tuple[str, ...] = ("1", "2", "3", "4", "5")):
        self.tok = AutoTokenizer.from_pretrained(model_path)
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "left"  # last position must be the real final token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=dtype, device_map=device,
        ).eval()
        self.codes = codes
        self.code_ids = self._resolve_code_ids()

    def _resolve_code_ids(self) -> list[int]:
        """Token id for each digit as it appears immediately after the JSON prefix.

        Tokenisers differ on whether '1' after '"' is its own token, so resolve by
        encoding prefix+digit and taking the first token that diverges from prefix.
        """
        base = self.tok.encode(JSON_PREFIX, add_special_tokens=False)
        ids = []
        for c in self.codes:
            full = self.tok.encode(JSON_PREFIX + c, add_special_tokens=False)
            if len(full) <= len(base) or full[: len(base)] != base:
                # Tokeniser merged across the boundary; fall back to the bare digit.
                cand = self.tok.encode(c, add_special_tokens=False)
                if len(cand) != 1:
                    raise ValueError(f"cannot resolve a single token for code {c!r}")
                ids.append(cand[0])
            else:
                ids.append(full[len(base)])
        if len(set(ids)) != len(ids):
            raise ValueError(f"code tokens are not distinct: {ids}")
        return ids

    def _render(self, messages: list[dict]) -> str:
        text = self.tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            **({"enable_thinking": False} if "enable_thinking"
               in (self.tok.chat_template or "") else {}),
        )
        return text + JSON_PREFIX

    @torch.no_grad()
    def score_batch(self, batch: list[tuple[str, str, list[dict]]],
                    *, max_len: int = 4096) -> list[ScoredItem]:
        texts = [self._render(m) for _, _, m in batch]
        enc = self.tok(texts, return_tensors="pt", padding=True, truncation=True,
                       max_length=max_len, add_special_tokens=False).to(self.model.device)
        logits = self.model(**enc).logits[:, -1, :].float()
        sel = logits[:, self.code_ids]
        probs = torch.softmax(sel, dim=-1)
        best = probs.argmax(dim=-1)
        out = []
        for i, (cid, order, _) in enumerate(batch):
            out.append(ScoredItem(
                comparison_id=cid, order=order, code=self.codes[best[i].item()],
                probs={c: float(probs[i, j]) for j, c in enumerate(self.codes)},
            ))
        return out
