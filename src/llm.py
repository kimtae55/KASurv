#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

# Disable tokenizer parallel worker setup because that path was unstable in this environment.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# Keep Hugging Face downloads on the simpler default code path while debugging large-model loads.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

DEFAULT_QUESTION = "How important is APOE4 status on Alzheimer's disease diagnosis?"
DEFAULT_PMC_LLAMA_MODEL_ID = "axiong/PMC_LLaMA_13B"


@dataclass(frozen=True)
class ModelSpec:
    """Configuration for one supported LLM backend."""

    key: str
    model_id: str
    runner: str


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
    """Build an instruction prompt that requests JSON-only output."""

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

    return None, 0, first_sentence(clean_text)


def run_hf_causal_lm(question: str, model_key: str, model_id: str, max_new_tokens: int) -> ModelResult:
    """Run a Hugging Face causal LM and normalize the generated answer."""

    try:
        import torch
        import transformers
    except ImportError as exc:
        return ModelResult(
            model_key=model_key,
            importance_probability=None,
            num_sources=0,
            reasoning="",
            raw_response="",
            error=f"missing package: {exc}",
        )

    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, use_fast=False)
        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_kwargs = {"low_cpu_mem_usage": True}
        if device == "cuda":
            model_kwargs["torch_dtype"] = torch.float16

        model = transformers.AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
        model = model.to(device)
        model.eval()

        prompt = build_prompt(question)
        model_inputs = tokenizer([prompt], return_tensors="pt", padding=True)
        model_inputs = {k: v.to(device) for k, v in model_inputs.items()}

        with torch.inference_mode():
            generated = model.generate(
                model_inputs["input_ids"],
                attention_mask=model_inputs.get("attention_mask"),
                max_new_tokens=max_new_tokens,
                top_k=50,
                pad_token_id=tokenizer.pad_token_id,
            )

        # Decoder-only generation returns prompt + continuation, so strip the prompt tokens first.
        prompt_len = model_inputs["input_ids"].shape[1]
        generated_only = generated[:, prompt_len:]
        output_text = tokenizer.batch_decode(generated_only, skip_special_tokens=True)[0]
        score, num_sources, reasoning = parse_llm_json(output_text)
        return ModelResult(
            model_key=model_key,
            importance_probability=score,
            num_sources=num_sources,
            reasoning=reasoning or "No reasoning returned.",
            raw_response=output_text,
        )
    except Exception as exc:
        return ModelResult(
            model_key=model_key,
            importance_probability=None,
            num_sources=0,
            reasoning="",
            raw_response="",
            error=str(exc),
        )


def default_model_specs(pmc_llama_model_id: str) -> Dict[str, ModelSpec]:
    """Return the supported model registry for this run."""

    return {
        "pmc_llama": ModelSpec(
            key="pmc_llama",
            model_id=pmc_llama_model_id,
            runner="hf_causal_lm",
        ),
    }


def available_runners() -> Dict[str, Callable[[str, str, str, int], ModelResult]]:
    """Map runner names to implementation functions."""

    return {
        "hf_causal_lm": run_hf_causal_lm,
    }


def run_models(question: str, requested: List[str], model_specs: Dict[str, ModelSpec], max_new_tokens: int) -> List[ModelResult]:
    """Run all requested models using the configured registry."""

    runners = available_runners()
    results: List[ModelResult] = []

    for model_key in requested:
        spec = model_specs.get(model_key)
        if spec is None:
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
            continue

        runner = runners[spec.runner]
        results.append(runner(question, spec.key, spec.model_id, max_new_tokens))

    return results


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
    """Define the CLI for question, model selection, and output settings."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--question", type=str, default=DEFAULT_QUESTION)
    parser.add_argument("--models", type=str, default="pmc_llama")
    parser.add_argument("--pmc-llama-model-id", type=str, default=DEFAULT_PMC_LLAMA_MODEL_ID)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--output-json", type=str, default="results/llm_apoe4.json")
    parser.add_argument("--list-models", action="store_true")
    return parser


def main() -> None:
    """Parse CLI args, run the requested models, then print and save results."""

    args = build_argparser().parse_args()
    model_specs = default_model_specs(args.pmc_llama_model_id)

    if args.list_models:
        for key, spec in model_specs.items():
            print(f"{key}: runner={spec.runner} model_id={spec.model_id}")
        return

    requested = [x.strip() for x in args.models.split(",") if x.strip()]
    results = run_models(args.question, requested, model_specs, args.max_new_tokens)
    print_results(results)
    save_results(results, Path(args.output_json))


if __name__ == "__main__":
    main()
