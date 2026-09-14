"""BPE Tokenizer engine with Kimi-K3 tokens and Indonesian-optimized regex splitter."""

from typing import Dict, List, Optional, Set, Union
import tiktoken

# Indonesian & multilingual frontier regex pattern for BPE pre-tokenization
# Supports Indonesian hyphenated compounds/reduplications, clitics (-nya, -ku, -mu, 'kan),
# grouped digits (1-3), punctuation/symbols, and code/newline indentations.
INDONESIAN_BPE_PAT = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d|'kan|'ku|'mu|'nya)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}{1,3}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""

# Canonical special tokens used across OutMind (chat, media, mode, and control)
SPECIAL_TOKEN_NAMES: List[str] = [
    "[BOS]",
    "[EOS]",
    "<|end_of_msg|>",
    "<|open|>",
    "<|close|>",
    "<|sep|>",
    "[start_header_id]",
    "[end_header_id]",
    "[EOT]",
    "<|media_begin|>",
    "<|media_content|>",
    "<|media_end|>",
    "<|media_pad|>",
    "<osagent_mode>",
    "[UNK]",
    "[PAD]",
]

# Legacy 163K special tokens dictionary for Moonshot Kimi-K3 compatibility
KIMI_SPECIAL_TOKENS: Dict[str, int] = {
    name: 163584 + i if i < 9 else (163602 + i - 9 if i < 13 else (163649 if i == 13 else 163838 + i - 14))
    for i, name in enumerate(SPECIAL_TOKEN_NAMES)
}


def _load_tiktoken_model(vocab_file: str) -> Dict[bytes, int]:
    """Loads BPE mergeable ranks directly from file without stale temp caching."""
    import base64
    ranks: Dict[bytes, int] = {}
    with open(vocab_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            token, rank = line.split()
            ranks[base64.b64decode(token)] = int(rank)
    return ranks


class OutMindTokenizer:
    def __init__(self, vocab_file: Optional[str] = None):
        self.bos_token = "[BOS]"
        self.eos_token = "[EOS]"
        self.pad_token = "[PAD]"
        self.eot_token = "[EOT]"
        self.start_header_token = "[start_header_id]"
        self.end_header_token = "[end_header_id]"

        if vocab_file:
            import os
            mergeable_ranks = _load_tiktoken_model(vocab_file)
            # Assign special tokens contiguously at the end of mergeable ranks
            n_merges = len(mergeable_ranks)
            self.special_tokens = {name: n_merges + i for i, name in enumerate(SPECIAL_TOKEN_NAMES)}
            self._encoding = tiktoken.Encoding(
                name=f"outmind_custom_{os.path.basename(vocab_file)}",
                pat_str=INDONESIAN_BPE_PAT,
                mergeable_ranks=mergeable_ranks,
                special_tokens=self.special_tokens,
            )
        else:
            # Fallback to o200k_base with contiguous special tokens appended
            base = tiktoken.get_encoding("o200k_base")
            specials = dict(base._special_tokens)
            next_id = (max(specials.values()) + 1) if specials else len(base._mergeable_ranks)
            for name in SPECIAL_TOKEN_NAMES:
                if name not in specials:
                    specials[name] = next_id
                    next_id += 1
            self.special_tokens = specials
            self._encoding = tiktoken.Encoding(
                name="outmind_frontier_base",
                pat_str=INDONESIAN_BPE_PAT,
                mergeable_ranks=base._mergeable_ranks,
                special_tokens=self.special_tokens,
            )

        self.bos_id = self._encoding.encode(self.bos_token, allowed_special="all")[0]
        self.eos_id = self._encoding.encode(self.eos_token, allowed_special="all")[0]
        self.pad_id = self._encoding.encode(self.pad_token, allowed_special="all")[0]
        self.eot_id = self._encoding.encode(self.eot_token, allowed_special="all")[0]

    @property
    def vocab_size(self) -> int:
        return self._encoding.n_vocab

    def encode(self, text: str, allowed_special: Union[str, Set[str]] = "all") -> List[int]:
        return self._encoding.encode(text, allowed_special=allowed_special)

    def decode(self, token_ids: List[int], errors: str = "replace") -> str:
        try:
            return self._encoding.decode(token_ids, errors=errors)
        except Exception:
            chunks = []
            for t in token_ids:
                try:
                    chunks.append(self._encoding.decode_single_token_bytes(t).decode("utf-8", errors=errors))
                except Exception:
                    chunks.append(f"<|token_{t}|>")
            return "".join(chunks)

    def apply_chat_template(
        self,
        messages: List[Dict[str, str]],
        add_generation_prompt: bool = True,
    ) -> str:
        """Renders multi-turn conversations into Kimi-K3/LLaMA-3 chat format."""
        rendered = f"{self.bos_token}"
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            rendered += f"{self.start_header_token}{role}{self.end_header_token}\n{content}{self.eot_token}\n"

        if add_generation_prompt:
            rendered += f"{self.start_header_token}assistant{self.end_header_token}\n"

        return rendered
