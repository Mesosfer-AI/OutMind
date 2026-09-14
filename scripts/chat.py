"""Interactive terminal chat assistant with streaming token generation."""

import argparse
import sys
import torch

from outmind.config import OutMindConfig
from outmind.generate import generate_stream
from outmind.model import OutMindForCausalLM
from outmind.tokenizer import OutMindTokenizer


def main():
    parser = argparse.ArgumentParser(description="OutMind Interactive Streaming Terminal Chat")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to trained model checkpoint (.pt)")
    parser.add_argument("--preset", type=str, default="nano", choices=["nano", "small", "moe"])
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--test-prompt", type=str, default=None, help="Direct prompt for non-interactive test")
    args = parser.parse_args()

    tokenizer = OutMindTokenizer()
    cfg = getattr(OutMindConfig, args.preset)(vocab_size=tokenizer.vocab_size)
    model = OutMindForCausalLM(cfg)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(state.get("model_state", state))
        print(f"[OutMind] Loaded weights from {args.checkpoint}")

    model.to(device)
    model.eval()

    system_prompt = "Kamu adalah OutMind, asisten AI cerdas dan ramah buatan Mesosfer-AI."

    if args.test_prompt:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": args.test_prompt},
        ]
        formatted = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        print(f"User: {args.test_prompt}\nOutMind: ", end="", flush=True)
        for token_chunk in generate_stream(
            model,
            tokenizer,
            formatted,
            max_new_tokens=32,
            temperature=args.temperature,
            top_p=args.top_p,
        ):
            print(token_chunk, end="", flush=True)
        print()
        return

    print("=" * 60)
    print("[OutMind] Interactive Chat (Type 'exit' or 'quit' to stop)")
    print(f"Preset: {args.preset.upper()} | Device: {device.upper()}")
    print("=" * 60)

    history = [{"role": "system", "content": system_prompt}]

    while True:
        try:
            user_input = input("\nUser > ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit"):
                print("Goodbye!")
                break

            history.append({"role": "user", "content": user_input})
            formatted_prompt = tokenizer.apply_chat_template(history, add_generation_prompt=True)

            print("OutMind > ", end="", flush=True)
            response_chunks = []

            for chunk in generate_stream(
                model,
                tokenizer,
                formatted_prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            ):
                print(chunk, end="", flush=True)
                response_chunks.append(chunk)

            print()
            history.append({"role": "assistant", "content": "".join(response_chunks)})

        except (KeyboardInterrupt, EOFError):
            print("\nExiting chat.")
            break


if __name__ == "__main__":
    main()
