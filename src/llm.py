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

DEFAULT_CONFIG_PATH = Path("config/llm_config.json")


@dataclass(frozen=True)
class ModelSpec:
    """Configuration for one supported LLM backend."""

    key: str
    model_id: str
    runner: str


@dataclass
class LoadedModel:
    """Loaded model assets reused across multiple feature queries."""

    model_key: str
    runner: str
    model_id: str
    tokenizer: object
    model: object
    device: str


@dataclass
class ModelResult:
    """Normalized result payload returned for each queried model."""

    dataset: str
    model_key: str
    feature: str
    outcome: str
    confidence: Optional[float]
    answer: str
    raw_response: str
    error: Optional[str] = None


@dataclass(frozen=True)
class RunConfig:
    """Runtime configuration loaded from JSON."""

    dataset: str
    features: List[str]
    outcome: str
    models: List[str]
    model_ids: Dict[str, str]
    max_new_tokens: int
    output_json: str


def build_question(feature: str, outcome: str) -> str:
    """Build the natural-language claim that the model should assess."""

    return (
        f"In a survival analysis setting, is {feature} associated with increased risk, earlier onset, faster progression, "
        f"or poorer prognosis for {outcome}?"
    )


def build_prompt(feature: str, outcome: str) -> str:
    """Build an instruction prompt that requests JSON-only output."""

    question = build_question(feature, outcome)

    return (
        "You are a biomedical expert.\n"
        "Task:\n"
        "Answer the biomedical question below and return the answer as JSON.\n"
        "You must answer the question itself, not describe the prompt.\n"
        "\n"
        "Question:\n"
        f"{question}\n"
        "\n"
        "Current inputs:\n"
        f'- feature: "{feature}"\n'
        f'- outcome: "{outcome}"\n'
        "\n"
        "Output requirements:\n"
        "1. Return exactly one valid JSON object.\n"
        '2. The JSON object must contain exactly these keys: "answer" and "confidence".\n'
        '3. First determine whether the answer is yes, no, or unknown, then set "confidence" so it is consistent with that answer.\n'
        f'4. "answer" must be 1 to 2 sentences and must follow this style: "Yes, {feature} is known to predict {outcome}, because ...", "No, {feature} is not known to predict {outcome}, because ...", or "Unknown, current evidence is insufficient to determine whether {feature} predicts {outcome}, because ...". Do not repeat the question verbatim.\n'
        '5. "confidence" must be a single number in [-1, 1] representing the answer on a signed confidence scale. Use values near 1 if the answer is yes, values near -1 if the answer is no, and values near 0 if the answer is unknown or the evidence is unclear.\n'
        "6. Answer and confidence must be correctly reflect true facts from biomedical findings, clinical relevance, or prognostic/diagnostic relevance.\n"
        "7. Do not justify importance by saying the feature is common in articles, frequently mentioned, or widely studied.\n"
        '8. Double check the alignment between "answer" and "confidence".'
        "\n"
        "JSON Output:"
    )


def normalize_answer(text: str) -> str:
    """Normalize whitespace in the answer field without rewriting content."""

    text = " ".join(text.strip().split())
    return text


def strip_response_prefix(text: str) -> str:
    """Drop the echoed prompt if the model repeats the instruction template."""

    marker = "### Response:"
    if marker in text:
        return text.split(marker, 1)[1].strip()
    return text.strip()


def normalize_confidence(value) -> Optional[float]:
    """Parse a numeric confidence score and clamp it to the [-1, 1] range."""

    try:
        score = float(str(value).replace("%", "").strip())
    except (TypeError, ValueError):
        return None
    if abs(score) > 1.0:
        score /= 100.0
    return max(-1.0, min(1.0, score))


def parse_llm_json(text: str) -> tuple[Optional[float], str]:
    """Extract a confidence score and answer text from model output."""

    clean_text = strip_response_prefix(text)
    match = re.search(r"(\{.*?\})", clean_text, flags=re.DOTALL)
    if match:
        try:
            payload = json.loads(match.group(1))
            return (
                normalize_confidence(payload.get("confidence")),
                normalize_answer(str(payload.get("answer", ""))),
            )
        except json.JSONDecodeError:
            pass

    score_match = re.search(
        r'"confidence"\s*:\s*(-?[0-9]*\.?[0-9]+)|\bconfidence\b\s*[:=]?\s*(-?[0-9]*\.?[0-9]+)',
        clean_text,
        flags=re.IGNORECASE,
    )
    score = None
    if score_match:
        score = normalize_confidence(score_match.group(1) or score_match.group(2))
    return score, normalize_answer(clean_text)


def load_config(config_path: Path) -> RunConfig:
    """Load and validate the JSON config used for the current run."""

    payload = json.loads(config_path.read_text())
    return RunConfig(
        dataset=str(payload["dataset"]).strip(),
        features=[str(x).strip() for x in payload["features"] if str(x).strip()],
        outcome=str(payload["outcome"]).strip(),
        models=[str(x).strip() for x in payload["models"] if str(x).strip()],
        model_ids={str(k).strip(): str(v).strip() for k, v in payload["model_ids"].items()},
        max_new_tokens=int(payload.get("max_new_tokens", 256)),
        output_json=str(payload.get("output_json", "results/llm_apoe4.json")).strip(),
    )


def load_hf_causal_lm(model_key: str, model_id: str) -> LoadedModel:
    """Load a Hugging Face causal LM once so it can be reused across features."""

    import torch
    import transformers

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, use_fast=False)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_kwargs = {"low_cpu_mem_usage": True}
    if device == "cuda":
        model_kwargs["dtype"] = torch.float16

    model = transformers.AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    model = model.to(device)
    model.eval()
    return LoadedModel(
        model_key=model_key,
        runner="hf_causal_lm",
        model_id=model_id,
        tokenizer=tokenizer,
        model=model,
        device=device,
    )


def infer_hf_causal_lm(loaded_model: LoadedModel, dataset: str, feature: str, outcome: str, max_new_tokens: int) -> ModelResult:
    """Run one feature/outcome query against a preloaded Hugging Face causal LM."""

    import torch

    prompt = build_prompt(feature, outcome)
    model_inputs = loaded_model.tokenizer([prompt], return_tensors="pt", padding=True)
    model_inputs = {k: v.to(loaded_model.device) for k, v in model_inputs.items()}

    with torch.inference_mode():
        generated = loaded_model.model.generate(
            model_inputs["input_ids"],
            attention_mask=model_inputs.get("attention_mask"),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=loaded_model.tokenizer.pad_token_id,
        )

    prompt_len = model_inputs["input_ids"].shape[1]
    generated_only = generated[:, prompt_len:]
    output_text = loaded_model.tokenizer.batch_decode(generated_only, skip_special_tokens=True)[0]
    score, answer = parse_llm_json(output_text)
    return ModelResult(
        dataset=dataset,
        model_key=loaded_model.model_key,
        feature=feature,
        outcome=outcome,
        confidence=score,
        answer=answer or "No answer returned.",
        raw_response=output_text,
    )


def build_model_specs(config: RunConfig) -> Dict[str, ModelSpec]:
    """Return the supported model registry for this run."""

    return {
        "pmc_llama": ModelSpec(
            key="pmc_llama",
            model_id=config.model_ids["pmc_llama"],
            runner="hf_causal_lm",
        ),
    }


def available_loaders() -> Dict[str, Callable[[str, str], LoadedModel]]:
    """Map runner names to model-loading functions."""

    return {
        "hf_causal_lm": load_hf_causal_lm,
    }


def available_inferencers() -> Dict[str, Callable[[LoadedModel, str, str, str, int], ModelResult]]:
    """Map runner names to inference functions over preloaded models."""

    return {
        "hf_causal_lm": infer_hf_causal_lm,
    }


def load_models(config: RunConfig, model_specs: Dict[str, ModelSpec]) -> tuple[Dict[str, LoadedModel], List[ModelResult]]:
    """Load each requested model once and collect load-time errors as results."""

    loaders = available_loaders()
    loaded_models: Dict[str, LoadedModel] = {}
    load_errors: List[ModelResult] = []

    for model_key in config.models:
        spec = model_specs.get(model_key)
        if spec is None:
            load_errors.append(
                ModelResult(
                    dataset=config.dataset,
                    model_key=model_key,
                    feature="",
                    outcome=config.outcome,
                    confidence=None,
                    answer="",
                    raw_response="",
                    error=f"unsupported model key: {model_key}",
                )
            )
            continue

        try:
            loader = loaders[spec.runner]
            loaded_models[model_key] = loader(spec.key, spec.model_id)
        except Exception as exc:
            load_errors.append(
                ModelResult(
                    dataset=config.dataset,
                    model_key=model_key,
                    feature="",
                    outcome=config.outcome,
                    confidence=None,
                    answer="",
                    raw_response="",
                    error=str(exc),
                )
            )

    return loaded_models, load_errors


def run_models(config: RunConfig, loaded_models: Dict[str, LoadedModel], load_errors: List[ModelResult]) -> List[ModelResult]:
    """Run all requested models for each feature using preloaded model assets."""

    inferencers = available_inferencers()
    results: List[ModelResult] = []
    results.extend(load_errors)

    for feature in config.features:
        for model_key in config.models:
            loaded_model = loaded_models.get(model_key)
            if loaded_model is None:
                continue

            try:
                inferencer = inferencers[loaded_model.runner]
                results.append(inferencer(loaded_model, config.dataset, feature, config.outcome, config.max_new_tokens))
            except Exception as exc:
                results.append(
                    ModelResult(
                        dataset=config.dataset,
                        model_key=model_key,
                        feature=feature,
                        outcome=config.outcome,
                        confidence=None,
                        answer="",
                        raw_response="",
                        error=str(exc),
                    )
                )

    return results


def save_results(results: List[ModelResult], output_json: Path) -> None:
    """Persist results to JSON so later runs can be compared programmatically."""

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps([asdict(x) for x in results], indent=2) + "\n")


def build_argparser() -> argparse.ArgumentParser:
    """Define the CLI for question, model selection, and output settings."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--list-models", action="store_true")
    return parser


def main() -> None:
    """Parse CLI args, run the requested models, then print and save results."""

    args = build_argparser().parse_args() 
    config = load_config(Path(args.config))
    model_specs = build_model_specs(config)

    if args.list_models:
        for key, spec in model_specs.items():
            print(f"{key}: runner={spec.runner} model_id={spec.model_id}")

    loaded_models, load_errors = load_models(config, model_specs)
    results = run_models(config, loaded_models, load_errors)
    save_results(results, Path(config.output_json))


if __name__ == "__main__":
    main()
