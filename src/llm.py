#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

# The PMC-LLaMA path was crashing in native tokenizer/model init on this macOS setup.
# Keeping tokenizer parallelism off avoids that fragile multiprocessing path.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

DEFAULT_QUESTION = "How important is APOE4 status on Alzheimer's disease diagnosis?"
DEFAULT_PMC_LLAMA_MODEL_ID = "axiong/PMC_LLaMA_13B"

@dataclass
class ModelResult:
    """Normalized result payload returned for each queried model."""

    model_key: str
    importance_probability: Optional[float]
    num_sources: int
    reasoning: str
    raw_response: str
    error: Optional[str] = None


def build_prompt(question: str) -> str:
    """Build a single instruction-style prompt that asks for JSON-only output."""

    return (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n"
        "You are a biomedical assistant. Answer the question and return JSON only with keys "
        "`importance_probability`, `num_sources`, and `reasoning`. The probability must be between 0 and 1, "
        "`num_sources` must be an integer count of supporting sources/documents, and the reasoning must be exactly "
        "one sentence.\n\n"
        f"### Input:\n{question}\n\n"
        "### Response:"
    )


def first_sentence(text: str) -> str:
    """Trim free-form text down to one sentence for the reasoning field."""

    text = " ".join(text.strip().split())
    if not text:
        return ""
    return re.split(r"(?<=[.!?])\s+", text, maxsplit=1)[0].strip()


def strip_response_prefix(text: str) -> str:
    """Drop the echoed prompt if the model repeats the instruction template."""

    marker = "### Response:"
    if marker in text:
        return text.split(marker, 1)[1].strip()
    return text.strip()


def normalize_probability(value) -> Optional[float]:
    """Parse a numeric score and clamp it to the [0, 1] probability range."""

    try:
        prob = float(str(value).replace("%", "").strip())
    except (TypeError, ValueError):
        return None
    if prob > 1.0:
        prob /= 100.0
    return max(0.0, min(1.0, prob))


def normalize_int(value) -> Optional[int]:
    """Parse an integer field and reject non-integer values."""

    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def parse_llm_json(text: str) -> tuple[Optional[float], int, str]:
    """Extract a probability, source count, and one-sentence rationale from model output."""

    clean_text = strip_response_prefix(text)
    match = re.search(r"(\{.*?\})", clean_text, flags=re.DOTALL)
    if match:
        try:
            payload = json.loads(match.group(1))
            return (
                normalize_probability(payload.get("importance_probability")),
                normalize_int(payload.get("num_sources")) or 0,
                first_sentence(str(payload.get("reasoning", ""))),
            )
        except json.JSONDecodeError:
            pass

    # If the model did not actually return structured JSON, do not invent a score by
    # scraping the first number from the prompt echo. Keep the reasoning text only.
    return None, 0, first_sentence(clean_text)


def text_to_tensor(question: str, seq_len: int = 4096, vocab_size: int = 20000):
    """Convert the text question into a fixed-size integer tensor for MedPalm."""

    import torch

    encoded = [ord(ch) % vocab_size for ch in question][:seq_len]
    if len(encoded) < seq_len:
        encoded.extend([0] * (seq_len - len(encoded)))
    return torch.tensor([encoded], dtype=torch.long)


def run_medpalm(question: str) -> ModelResult:
    """Run the pip-installed MedPalm forward pass and turn its tensor output into a score."""

    try:
        import torch
        from medpalm.model import MedPalm
    except ImportError as exc:
        return ModelResult(
            model_key="medpalm",
            importance_probability=None,
            num_sources=0,
            reasoning="",
            raw_response="",
            error=f"missing package: {exc}",
        )

    try:
        img = torch.randn(1, 3, 256, 256)
        text = text_to_tensor(question)
        model = MedPalm()
        output = model(img, text)
        # MedPalm returns a tuple here: the token-level logits tensor and an auxiliary scalar.
        # We score from the logits because there is no text-generation API in this package path.
        if isinstance(output, tuple):
            logits = output[0]
            aux_value = output[1]
        else:
            logits = output
            aux_value = None

        score = float(torch.sigmoid(logits.float().mean()).item())
        reasoning = (
            "This score is derived from the MedPalm tensor output because the package example exposes a forward pass, "
            "not a text-generation interface."
        )
        raw = f"logits_shape={tuple(logits.shape)} logits_mean={logits.float().mean().item():.6f}"
        if aux_value is not None and hasattr(aux_value, "item"):
            raw += f" aux_value={float(aux_value.item()):.6f}"
        return ModelResult(
            model_key="medpalm",
            importance_probability=score,
            num_sources=0,
            reasoning=reasoning,
            raw_response=raw,
        )
    except Exception as exc:
        return ModelResult(
            model_key="medpalm",
            importance_probability=None,
            num_sources=0,
            reasoning="",
            raw_response="",
            error=str(exc),
        )


def run_pmc_llama(question: str, model_id: str, max_new_tokens: int) -> ModelResult:
    """Load PMC-LLaMA from Hugging Face cache, generate an answer, and parse it."""

    try:
        import torch
        import transformers
    except ImportError as exc:
        return ModelResult(
            model_key="pmc_llama",
            importance_probability=None,
            num_sources=0,
            reasoning="",
            raw_response="",
            error=f"missing package: {exc}",
        )

    try:
        # `use_fast=False` is intentional here; the fast/native path was unstable in this environment.
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, use_fast=False)
        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token

        device = "cuda" if torch.cuda.is_available() else "cpu"
        # This reduces peak RAM during initialization, which matters for large 13B checkpoints.
        model_kwargs = {"low_cpu_mem_usage": True}
        if device == "cuda":
            model_kwargs["torch_dtype"] = torch.float16

        model = transformers.AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
        model = model.to(device)
        model.eval()

        prompt = build_prompt(question)
        model_inputs = tokenizer([prompt], return_tensors="pt", padding=True)
        model_inputs = {k: v.to(device) for k, v in model_inputs.items()}

        # Inference mode avoids autograd overhead and keeps generation lighter-weight.
        with torch.inference_mode():
            generated = model.generate(
                model_inputs["input_ids"],
                attention_mask=model_inputs.get("attention_mask"),
                max_new_tokens=max_new_tokens,
                top_k=50,
                pad_token_id=tokenizer.pad_token_id,
            )
        prompt_len = model_inputs["input_ids"].shape[1]
        generated_only = generated[:, prompt_len:]
        output_text = tokenizer.batch_decode(generated_only, skip_special_tokens=True)[0]
        score, num_sources, reasoning = parse_llm_json(output_text)
        return ModelResult(
            model_key="pmc_llama",
            importance_probability=score,
            num_sources=num_sources,
            reasoning=reasoning or "No reasoning returned.",
            raw_response=output_text,
        )
    except Exception as exc:
        return ModelResult(
            model_key="pmc_llama",
            importance_probability=None,
            num_sources=0,
            reasoning="",
            raw_response="",
            error=str(exc),
        )


def print_results(results: List[ModelResult]) -> None:
    """Print a compact CLI summary for each model result."""

    for result in results:
        print(f"model_key={result.model_key}")
        if result.error:
            print(f"error={result.error}")
        else:
            score = "null" if result.importance_probability is None else f"{result.importance_probability:.4f}"
            print(f"importance_probability={score}")
            print(f"num_sources={result.num_sources}")
            print(f"reasoning={result.reasoning}")
        print("")


def save_results(results: List[ModelResult], output_json: Path) -> None:
    """Persist results to JSON so later runs can be compared programmatically."""

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps([asdict(x) for x in results], indent=2) + "\n")


def build_argparser() -> argparse.ArgumentParser:
    """Define the small CLI surface for selecting models and output settings."""

    parser = argparse.ArgumentParser()
    # `question` is the biomedical prompt shared across all requested models.
    parser.add_argument("--question", type=str, default=DEFAULT_QUESTION)
    # `models` stays comma-separated so more backends can be added without changing CLI shape.
    parser.add_argument("--models", type=str, default="medpalm,pmc_llama")
    # `pmc-llama-model-id` points Hugging Face loading at a specific checkpoint repo.
    parser.add_argument("--pmc-llama-model-id", type=str, default=DEFAULT_PMC_LLAMA_MODEL_ID)
    # `max-new-tokens` limits generation length to keep inference bounded.
    parser.add_argument("--max-new-tokens", type=int, default=256)
    # `output-json` stores the normalized per-model results from the current run.
    parser.add_argument("--output-json", type=str, default="results/llm_apoe4.json")
    return parser


def main() -> None:
    """Parse CLI args, run the requested models, then print and save results."""

    args = build_argparser().parse_args()
    requested = [x.strip() for x in args.models.split(",") if x.strip()]
    results: List[ModelResult] = []

    for model_key in requested:
        if model_key == "medpalm":
            results.append(run_medpalm(args.question))
        elif model_key == "pmc_llama":
            results.append(run_pmc_llama(args.question, args.pmc_llama_model_id, args.max_new_tokens))
        else:
            results.append(
                ModelResult(
                    model_key=model_key,
                    importance_probability=None,
                    num_sources=0,
                    reasoning="",
                    raw_response="",
                    error=f"unsupported model key: {model_key}",
                )
            )

    print_results(results)
    save_results(results, Path(args.output_json))


if __name__ == "__main__":
    main()
