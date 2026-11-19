"""LoRA / QLoRA adaptation of a small open-weight model.

Trains the behaviour the RAG pipeline needs: cite the ``[DOCn]`` labels you
used, emit exactly ``INSUFFICIENT_EVIDENCE`` when the context does not support
an answer, and ignore forged instructions inside retrieved documents.

The training loop is written out by hand rather than delegated to
``Trainer``. It is ~60 lines, every hyper-parameter is visible, and loss is
masked to the completion so the model is not rewarded for reproducing the
prompt -- all of which matters more than convenience when the code has to be
explained live.

Hardware check first, always::

    python finetuning/train_lora.py --check-only

Train (CPU is the expected case on this machine)::

    python finetuning/train_lora.py --epochs 3 --output-dir finetuning/adapters/run1

If a CUDA GPU and ``bitsandbytes`` are present the base model is loaded in
4-bit (QLoRA); otherwise it is loaded in fp32 and plain LoRA is used. Which
path ran is recorded in ``training_log.json`` -- do not report QLoRA unless
the log says ``quantization: "4bit"``.
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
SEED = 20261119


# --------------------------------------------------------------------------
# Hardware / dependency inspection
# --------------------------------------------------------------------------
@dataclass
class Environment:
    ok: bool
    device: str
    quantization: str
    details: dict[str, Any] = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)


def inspect_environment() -> Environment:
    """Report what is actually available. Never assume a GPU exists."""
    details: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
    }
    missing: list[str] = []

    try:
        import torch
    except ImportError:
        missing.append("torch")
        return Environment(False, "none", "none", details, missing)

    details["torch"] = torch.__version__
    details["torch_threads"] = torch.get_num_threads()

    for package in ("transformers", "peft"):
        try:
            module = __import__(package)
            details[package] = getattr(module, "__version__", "unknown")
        except ImportError:
            missing.append(package)

    cuda = torch.cuda.is_available()
    details["cuda_available"] = cuda
    if cuda:
        details["cuda_device"] = torch.cuda.get_device_name(0)
        free, total = torch.cuda.mem_get_info()
        details["cuda_total_gb"] = round(total / 1024**3, 2)
        details["cuda_free_gb"] = round(free / 1024**3, 2)
        details["bf16_supported"] = torch.cuda.is_bf16_supported()

    quantization = "none"
    if cuda:
        try:
            import bitsandbytes  # noqa: F401

            details["bitsandbytes"] = "available"
            quantization = "4bit"
        except ImportError:
            details["bitsandbytes"] = "not installed (QLoRA unavailable, using fp16 LoRA)"
            quantization = "fp16"
    else:
        quantization = "fp32-cpu"

    try:
        import psutil

        details["system_ram_gb"] = round(psutil.virtual_memory().total / 1024**3, 1)
    except ImportError:
        details["system_ram_gb"] = "unknown (psutil not installed)"

    return Environment(
        ok=not missing,
        device="cuda" if cuda else "cpu",
        quantization=quantization,
        details=details,
        missing=missing,
    )


def print_environment(env: Environment) -> None:
    print("=" * 68)
    print("HARDWARE / DEPENDENCY CHECK")
    print("=" * 68)
    for key, value in env.details.items():
        print(f"  {key:>20}: {value}")
    print(f"  {'device':>20}: {env.device}")
    print(f"  {'quantization':>20}: {env.quantization}")
    if env.missing:
        print(f"\n  MISSING: {', '.join(env.missing)}")
        print("  Install with: pip install -r requirements-finetuning.txt")
    if env.device == "cpu":
        print(
            "\n  No CUDA GPU detected. Training will run on CPU: this works for a\n"
            "  0.5B base model and a corpus this size, but expect minutes, not\n"
            "  seconds, and do not describe the run as QLoRA."
        )
    print("=" * 68)


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def encode_example(tokenizer, example: dict, max_len: int):
    """Tokenise one example, masking the prompt so loss covers only the answer."""
    import torch

    prompt = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": example["system"]},
            {"role": "user", "content": example["user"]},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(
        example["target"] + tokenizer.eos_token, add_special_tokens=False
    )["input_ids"]

    input_ids = (prompt_ids + answer_ids)[:max_len]
    labels = ([-100] * len(prompt_ids) + answer_ids)[:max_len]
    return torch.tensor(input_ids), torch.tensor(labels)


def collate(batch, pad_id: int):
    import torch

    width = max(len(ids) for ids, _ in batch)
    input_ids, labels, attention = [], [], []
    for ids, lbl in batch:
        pad = width - len(ids)
        input_ids.append(torch.cat([ids, torch.full((pad,), pad_id, dtype=ids.dtype)]))
        labels.append(torch.cat([lbl, torch.full((pad,), -100, dtype=lbl.dtype)]))
        attention.append(torch.cat([torch.ones(len(ids)), torch.zeros(pad)]).long())
    return torch.stack(input_ids), torch.stack(labels), torch.stack(attention)


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------
def train(args: argparse.Namespace, env: Environment) -> dict:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if env.device == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    print(f"\nLoading base model: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs: dict[str, Any] = {}
    if env.quantization == "4bit":
        from transformers import BitsAndBytesConfig

        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    elif env.quantization == "fp16":
        load_kwargs["torch_dtype"] = torch.float16

    model = AutoModelForCausalLM.from_pretrained(args.base_model, **load_kwargs)
    if env.device == "cuda":
        model = model.to("cuda")

    lora = LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=args.dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable:,} / {total:,} ({100 * trainable / total:.3f}%)")

    rows = load_jsonl(args.train_file)
    encoded = [encode_example(tokenizer, row, args.max_seq_len) for row in rows]
    print(f"Training examples: {len(encoded)} from {args.train_file}")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr
    )
    model.train()

    indices = list(range(len(encoded)))
    history: list[dict] = []
    step = 0
    started = time.perf_counter()

    for epoch in range(args.epochs):
        random.Random(args.seed + epoch).shuffle(indices)
        epoch_loss, batches = 0.0, 0
        for start in range(0, len(indices), args.batch_size):
            batch = [encoded[i] for i in indices[start : start + args.batch_size]]
            input_ids, labels, attention = collate(batch, tokenizer.pad_token_id)
            if env.device == "cuda":
                input_ids, labels, attention = (
                    input_ids.cuda(),
                    labels.cuda(),
                    attention.cuda(),
                )

            out = model(input_ids=input_ids, attention_mask=attention, labels=labels)
            loss = out.loss / args.grad_accum
            loss.backward()

            if (batches + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0
                )
                optimizer.step()
                optimizer.zero_grad()

            epoch_loss += out.loss.item()
            batches += 1
            step += 1
            if step % args.log_every == 0:
                print(f"  epoch {epoch + 1} step {step:4d}  loss {out.loss.item():.4f}")
                history.append({"step": step, "epoch": epoch + 1, "loss": out.loss.item()})

            if args.max_steps and step >= args.max_steps:
                break

        mean = epoch_loss / max(batches, 1)
        print(f"epoch {epoch + 1}/{args.epochs} mean loss {mean:.4f}")
        history.append({"step": step, "epoch": epoch + 1, "epoch_mean_loss": mean})
        if args.max_steps and step >= args.max_steps:
            print(f"stopping early at --max-steps {args.max_steps}")
            break

    elapsed = time.perf_counter() - started
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    print(f"\nAdapter saved to: {args.output_dir}")

    log = {
        "status": "COMPLETED",
        "base_model": args.base_model,
        "seed": args.seed,
        "device": env.device,
        "quantization": env.quantization,
        "method": "QLoRA (4-bit)" if env.quantization == "4bit" else "LoRA",
        "epochs_requested": args.epochs,
        "steps_completed": step,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "learning_rate": args.lr,
        "lora": {"r": args.rank, "alpha": args.alpha, "dropout": args.dropout},
        "max_seq_len": args.max_seq_len,
        "train_examples": len(encoded),
        "trainable_parameters": trainable,
        "total_parameters": total,
        "wall_clock_seconds": round(elapsed, 1),
        "environment": env.details,
        "history": history,
    }
    (args.output_dir / "training_log.json").write_text(
        json.dumps(log, indent=2), encoding="utf-8"
    )
    print(f"Training log: {args.output_dir / 'training_log.json'}")
    print(f"Wall clock: {elapsed / 60:.1f} min")
    return log


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--train-file", type=Path, default=Path("finetuning/synthetic_train.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("finetuning/adapters/run1"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--max-seq-len", type=int, default=768)
    parser.add_argument("--max-steps", type=int, default=0, help="0 = no limit")
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--check-only", action="store_true", help="inspect hardware and exit")
    args = parser.parse_args()

    env = inspect_environment()
    print_environment(env)

    if args.check_only:
        return
    if not env.ok:
        raise SystemExit(
            f"\nCannot train: missing {', '.join(env.missing)}.\n"
            "Install with: pip install -r requirements-finetuning.txt\n"
            "The experiment stays UNEXECUTED / NOT MEASURED until it actually runs."
        )

    train(args, env)


if __name__ == "__main__":
    main()
