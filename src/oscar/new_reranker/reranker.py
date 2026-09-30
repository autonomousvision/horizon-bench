"""
Standalone minimal demo for the two best reranking methods.

Usage:
  python best_rerank/reranker.py --method both
"""

import argparse
import json
from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from peft import PeftModel
except Exception:
    PeftModel = None


def _fmt(instruction: str, query: str, doc: str) -> str:
    return "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}".format(
        instruction=instruction,
        query=query,
        doc=doc,
    )


class Reranker:
    def __init__(
        self,
        method: str,
        method2_model_path: str = "method2_mse",
        method2_load_mode: str = "auto",
        method1_model_name: str = "Qwen/Qwen3-Reranker-8B",
        max_length: int = 512,
        batch_size: int = None,
        device: str = "cuda",
    ):
        if method not in ("method1", "method2"):
            raise ValueError("method must be method1 or method2")
        if method2_load_mode not in ("auto", "lora", "full"):
            raise ValueError("method2_load_mode must be one of: auto, lora, full")

        self.method = method
        self.method2_load_mode = method2_load_mode
        self.max_length = max_length
        self.batch_size = batch_size if batch_size is not None else (2 if method == "method1" else 16)
        self.device = torch.device(device if (device != "cuda" or torch.cuda.is_available()) else "cpu")

        self.prefix = (
            "<|im_start|>system\n"
            "Judge whether the Document meets the requirements based on the Query and the Instruct provided. "
            "Note that the answer can only be \"yes\" or \"no\"."
            "<|im_end|>\n<|im_start|>user\n"
        )
        self.suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

        self.task1 = "Given a research goal, evaluate whether the provided insight extracted from another paper is useful for this goal."
        self.task2 = "Given the research goal, determine if the provided insight is useful for achieving the goal."

        if method == "method1":
            self._load_method1(method1_model_name)
        else:
            self._load_method2(method2_model_path, method2_load_mode)

    def _load_model(self, model_id_or_path: str, **kwargs):
        if torch.cuda.is_available():
            try:
                kwargs["attn_implementation"] = "flash_attention_2"
                return AutoModelForCausalLM.from_pretrained(model_id_or_path, **kwargs)
            except Exception:
                kwargs["attn_implementation"] = "sdpa"
                return AutoModelForCausalLM.from_pretrained(model_id_or_path, **kwargs)
        return AutoModelForCausalLM.from_pretrained(model_id_or_path, **kwargs)

    def _finalize_tokens(self):
        self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
        self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")
        self.prefix_tokens = self.tokenizer.encode(self.prefix, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(self.suffix, add_special_tokens=False)

    def _load_method1(self, model_name: str):
        print("Loading method1 model: {0}".format(model_name))
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        self.model = self._load_model(
            model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )
        self.model = self.model.to(self.device)
        self.model.eval()
        self._finalize_tokens()

    def _load_method2(self, model_path: str, load_mode: str = "auto"):
        print("Loading method2 model: {0}".format(model_path))
        model_path_obj = Path(model_path)
        cfg = {}
        cfg_path = model_path_obj / "training_config.json"
        if cfg_path.exists():
            with open(cfg_path, "r") as f:
                cfg = json.load(f)

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
        self._finalize_tokens()

        adapter_cfg_path = model_path_obj / "adapter_config.json"
        adapter_weights_path = model_path_obj / "adapter_model.safetensors"
        has_lora = adapter_cfg_path.exists() and adapter_weights_path.exists()

        has_full = (
            (model_path_obj / "model.safetensors").exists()
            or (model_path_obj / "model.safetensors.index.json").exists()
            or bool(list(model_path_obj.glob("model-*.safetensors")))
        )

        chosen_mode = load_mode
        if chosen_mode == "auto":
            if has_lora:
                chosen_mode = "lora"
            elif has_full:
                chosen_mode = "full"
            else:
                raise FileNotFoundError(
                    "Could not detect LoRA or full model files in {0}".format(model_path)
                )

        if chosen_mode == "lora":
            if not has_lora:
                raise FileNotFoundError(
                    "LoRA files not found in {0}. Need adapter_config.json + adapter_model.safetensors".format(model_path)
                )
            if PeftModel is None:
                raise ImportError("peft is required for LoRA loading but is not installed.")

            with open(adapter_cfg_path, "r") as f:
                adapter_cfg = json.load(f)

            base_model = (
                adapter_cfg.get("base_model_name_or_path")
                or cfg.get("model_name")
                or "Qwen/Qwen3-Reranker-8B"
            )
            print("Detected mode: LoRA")
            print("Loading base model for LoRA: {0}".format(base_model))

            base = self._load_model(
                base_model,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            )
            self.model = PeftModel.from_pretrained(base, model_path)
            self.model = self.model.merge_and_unload()
        else:
            if not has_full:
                raise FileNotFoundError(
                    "Full model files not found in {0}".format(model_path)
                )
            print("Detected mode: Full (non-LoRA)")
            self.model = self._load_model(
                model_path,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            )

        self.model = self.model.to(self.device)
        self.model.eval()

    def _score(self, goal: str, insights: List[str], instruction: str) -> List[float]:
        texts = [_fmt(instruction, goal, s) for s in insights]
        scores = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            toks = self.tokenizer(
                batch,
                padding=False,
                truncation="longest_first",
                return_attention_mask=False,
                max_length=self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens),
            )

            input_ids = [self.prefix_tokens + x + self.suffix_tokens for x in toks["input_ids"]]
            padded = self.tokenizer.pad(
                {"input_ids": input_ids},
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt",
            )

            input_ids_t = padded["input_ids"].to(self.device)
            attn_t = padded["attention_mask"].to(self.device)

            with torch.no_grad():
                logits = self.model(input_ids=input_ids_t, attention_mask=attn_t).logits[:, -1, :]
                yes_logits = logits[:, self.token_true_id]
                no_logits = logits[:, self.token_false_id]
                pair = torch.stack([no_logits, yes_logits], dim=1)
                probs = F.softmax(pair, dim=1)[:, 1].float().cpu().tolist()
                scores.extend(float(p) for p in probs)

        return scores

    def rerank(self, goal: str, insights: List[str]):
        instruction = self.task1 if self.method == "method1" else self.task2
        scores = self._score(goal, insights, instruction)
        ranking = sorted(range(len(scores)), key=lambda i: (-scores[i], i))

        return [
            {
                "rank": pos,
                "index": idx,
                "insight": insights[idx],
                "predicted_score": float(scores[idx]),
            }
            for pos, idx in enumerate(ranking, start=1)
        ]    


def main():
    parser = argparse.ArgumentParser(description="Standalone minimal demo reranker")
    parser.add_argument("--method", choices=["method1", "method2", "both"], default="both")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument(
        "--method2_model_path",
        type=str,
        default=str(Path(__file__).resolve().parent / "method2_mse"),
    )
    parser.add_argument("--method1_model_name", type=str, default="Qwen/Qwen3-Reranker-8B")
    parser.add_argument(
        "--method2_load_mode",
        choices=["auto", "lora", "full"],
        default="auto",
        help="How to load method2 model path: auto-detect, force lora, or force full checkpoint.",
    )
    args = parser.parse_args()

    # -------------------------------------------------------------------------
    # Minimal hardcoded sample (edit these directly for your own quick demos)
    # -------------------------------------------------------------------------
    goal = "To improve the generalization of hand gesture recognition from wearable electromyography signals by aligning their representations with high-quality, semantically rich data from other structural modalities."

    insights = [
      "Wearable EMG signals often lack the semantic richness and signal-to-noise ratio found in other structural modalities, such as vision or high-fidelity IMU data, limiting their standalone generalization.",
      "Simple late-stage fusion is insufficient; a progressive, hierarchical fusion strategy is needed to capture complex, multi-scale interactions between modalities throughout the feature extraction process.",
      "While sEMG captures muscle activation, it lacks the explicit spatial and structural information present in vision-based modalities like hand pose or video.",
      "Supervising sEMG feature extraction with high-quality data from other modalities through cross-modal knowledge distillation can bridge the semantic gap and enrich the sEMG representations.",
      "A multi-stage training approach is necessary to first stabilize the internal sEMG feature space before attempting to align it with complex external modal data.",
      "Standard neural network architectures are not optimized for the specific sparse and multichannel structure of wearable sEMG, necessitating a custom spatial-temporal encoder like sEMGXCM.",
      "Directly classifying a large set of hand gestures from sEMG signals is computationally intensive and prone to poor generalization due to high signal variability.",
      "Machine vision can act as a high-fidelity reference modality to provide structural and semantic context that sEMG signals lack in isolation.",
      "A hierarchical classification approach, where gestures are first grouped into coarse categories before fine-grained identification, can simplify the learning objective for the sEMG model.",
      "Utilizing a specialized colored glove allows for a reliable vision-based pre-classification step, partitioning ten gestures into five manageable categories.",
      "To achieve high accuracy within these categories, basic time-domain features of sEMG signals may be insufficient to capture the subtle nuances of muscle activation.",
      "Integrating frequency-domain features with time-domain features provides a more robust and comprehensive representation of the electromyography signal, enhancing the discriminatory power of the K-Nearest Neighbor classifier.",
      "A cascaded transformer-based fusion branch can adaptively model the similarities and dependencies between different modality features more effectively than static aggregation methods.",
      "Conducting a comparative analysis between feature sets is essential to validate that multi-domain feature fusion actually improves the alignment and recognition performance over standard methods.",
      "To improve the performance of the unimodal EMG branch, the 'interactive knowledge' generated by the multimodal fusion must be explicitly mapped back to the unimodal representations.",
      "Effective cross-modal alignment requires constraints at different levels of abstraction: the embedding space for feature-level similarity and the probability space for categorical consistency.",
      "Supervised contrastive learning can be adapted to cross-modal contexts to ensure that different representations of the same gesture are clustered together while distinct gestures are separated.",
      "Online distillation allows the model to treat the multimodal fusion branch as a dynamic teacher, supervising the unimodal branches in real-time to internalize multimodal insights.",
      "Practical wearable applications require a flexible architecture where the system remains operational even when certain sensor modalities are unavailable or signals are lost.",
      "Raw sEMG signals are highly susceptible to noise and inter-trial variability, which prevents simple classifiers from generalizing across different users and sessions.",
      "Contrastive learning, specifically through mutual information maximization, can be leveraged to filter out trial-specific noise and extract features that are consistent across different sessions of the same gesture."
    ]

    scores = [
      0.87353515625,
      0.69677734375,
      0.79638671875,
      0.75390625,
      0.80859375,
      0.80029296875,
      0.7939453125,
      0.8447265625,
      0.7900390625,
      0.83251953125,
      0.8232421875,
      0.845703125,
      0.8212890625,
      0.77490234375,
      0.74658203125,
      0.86181640625,
      0.7978515625,
      0.8095703125,
      0.84228515625,
      0.8291015625,
      0.837890625
    ]

    methods = [args.method] if args.method != "both" else ["method1", "method2"]
    for method in methods:
        rr = Reranker(
            method=method,
            method2_model_path=args.method2_model_path,
            method2_load_mode=args.method2_load_mode,
            method1_model_name=args.method1_model_name,
            max_length=args.max_length,
            batch_size=args.batch_size,
            device=args.device,
        )
        out = rr.rerank(goal, insights)

        print("\n" + "=" * 80)
        print("METHOD: {0}".format(method))
        print("=" * 80)
        for row in out:
            gt = scores[row["index"]]
            print("#{0} | pred={1:.4f} | gt={2:.4f}".format(row["rank"], row["predicted_score"], gt))
            print(row["insight"])
            print("-" * 80)


if __name__ == "__main__":
    main()
