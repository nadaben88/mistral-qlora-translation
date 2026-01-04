"""
WMT14 English→French Data Preprocessing Script
===============================================

Causal-LM preprocessing for Mistral-7B + QLoRA fine-tuning.

Implements:
- WMT14 fr-en benchmark dataset
- English → French direction
- Prompt masking with -100 for causal LM loss
- Minimal noise filtering (WMT is pre-cleaned)
- Mandatory debug outputs

"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List

from datasets import load_dataset, Dataset
from transformers import AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class WMT14Preprocessor:  
    def __init__(
        self,
        model_name: str = "mistralai/Mistral-7B-v0.3",
        max_length: int = 512,
        output_dir: str = "./data",
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Loading tokenizer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=True
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            logger.info("Pad token set to EOS token")
        # Force padding on the RIGHT
        self.tokenizer.padding_side = "right"
        logger.info(f"Padding side set to: {self.tokenizer.padding_side}")

    # ------------------------------------------------------------------
    # Dataset loading 
    # ------------------------------------------------------------------
    def load_dataset(self, num_train_samples: int = 100000) -> Dict[str, Dataset]:
        """
        Load WMT14 fr-en dataset.
        
        Args:
            num_train_samples: Number of training samples to use (default 100K)
            
        Returns:
            Dictionary with train/validation/test splits
            
        Note: Full training set is 36M pairs. We subsample for efficiency.
        Validation and test are standard WMT benchmarks 
        
        """
        logger.info("Loading WMT14 fr-en dataset...")
        
        # Load full dataset
        dataset = load_dataset("wmt14", "fr-en")
        
        # Subsample training set 
        logger.info(f"Original train size: {len(dataset['train'])} pairs")
        dataset['train'] = dataset['train'].shuffle(seed=42).select(
            range(min(num_train_samples, len(dataset['train'])))
        )
        logger.info(f"Subsampled train to: {len(dataset['train'])} pairs")
        
        # Keep standard validation/test
        logger.info(f"Validation: {len(dataset['validation'])} pairs (newstest2013)")
        logger.info(f"Test: {len(dataset['test'])} pairs (newstest2014)")
        
        return dataset

    # ------------------------------------------------------------------
    # Cleaning & filtering 
    # ------------------------------------------------------------------
    @staticmethod
    def clean_text(text: str) -> str:
        if not isinstance(text, str):
            return ""
        return " ".join(text.strip().split())

    @staticmethod
    def is_noisy_pair(en: str, fr: str) -> bool:
        """
        Minimal filtering for WMT14 (already clean).
        
        WMT data is pre-filtered, so only catch extreme cases.
        
        Reference: http://www.statmt.org/wmt14/translation-task.html
        """
        # Only filter very short or identical
        if len(en) < 3 or len(fr) < 3:
            return True
        if en == fr:
            return True
        
        return False

    # ------------------------------------------------------------------
    # Prompt 
    # ------------------------------------------------------------------
    @staticmethod
    def format_prompt(en: str) -> str:
        return f"English: {en}\nFrench:"

    # ------------------------------------------------------------------
    # Core preprocessing
    # ------------------------------------------------------------------
    def preprocess_function(self, examples: Dict[str, Any]) -> Dict[str, Any]:
        input_ids_list: List[List[int]] = []
        labels_list: List[List[int]] = []

        filtered = 0
        max_len_seen = 0
        debug_printed = False

        for ex in examples["translation"]:
            en = self.clean_text(ex["en"])
            fr = self.clean_text(ex["fr"])

            if self.is_noisy_pair(en, fr):
                filtered += 1
                continue

            prompt = self.format_prompt(en)
            full_text = f"{prompt} {fr}"

            # Tokenize prompt without special tokens
            prompt_ids = self.tokenizer(
                prompt,
                add_special_tokens=False
            )["input_ids"]

            # Tokenize full text with special tokens
            full_ids = self.tokenizer(
                full_text,
                truncation=True,
                max_length=self.max_length - 1,
                padding="max_length",
                add_special_tokens=False
            )["input_ids"]
            # Manually append EOS before padding
            full_ids.append(self.tokenizer.eos_token_id)
            full_ids = full_ids[:self.max_length]

            # Find actual content length (before padding)
            try:
                # Find first pad token position
                pad_start = full_ids.index(self.tokenizer.pad_token_id)
            except ValueError:
                # No padding (sequence is exactly max_length)
                pad_start = len(full_ids)
            # Ensure EOS before padding starts
            if pad_start > 0 and full_ids[pad_start - 1] != self.tokenizer.eos_token_id:
                full_ids[pad_start - 1] = self.tokenizer.eos_token_id

            # Create labels (mask prompt with -100)
            labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
            labels = labels[:len(full_ids)]
            # Set pad tokens in labels to -100
            labels = [
                -100 if full_ids[i] == self.tokenizer.pad_token_id else labels[i]
                for i in range(len(labels))
            ]


            max_len_seen = max(max_len_seen, len(full_ids))

            # Debug output (once)
            if not debug_printed:
                decoded = self.tokenizer.decode(full_ids, skip_special_tokens=False)
                has_eos = decoded.endswith("</s>")
                
                logger.info("===== DEBUG SAMPLE =====")
                logger.info(f"Raw English: {en}")
                logger.info(f"Raw French: {fr}")
                logger.info(f"EOS present: {has_eos}")
                logger.info(f"Tokenized input_ids: {full_ids[:20]}...")
                logger.info(f"Tokenized labels: {labels[:20]}...")
                logger.info(f"Decoded:\n{decoded}")
                logger.info("========================")
                debug_printed = True

            input_ids_list.append(full_ids)
            labels_list.append(labels)

        logger.info(
            f"Batch stats → filtered: {filtered}, "
            f"kept: {len(input_ids_list)}, "
            f"max_length: {max_len_seen}"
        )

        return {
            "input_ids": input_ids_list,
            "labels": labels_list,
        }

    # ------------------------------------------------------------------
    # Split processing 
    # ------------------------------------------------------------------
    def preprocess_split(
        self,
        dataset: Dataset,
        split_name: str,
        num_samples: int = None
    ) -> Dataset:
        logger.info(f"Preprocessing {split_name} split")

        if num_samples is not None:
            dataset = dataset.select(range(min(num_samples, len(dataset))))
            logger.info(f"{split_name}: limited to {len(dataset)} raw samples")

        processed = dataset.map(
            self.preprocess_function,
            batched=True,
            remove_columns=dataset.column_names,
            desc=f"Tokenizing {split_name}"
        )

        logger.info(
            f"{split_name} final size: {len(processed)} examples"
        )

        return processed

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    def save(self, dataset: Dict[str, Dataset]):
        for split, ds in dataset.items():
            path = self.output_dir / split
            ds.save_to_disk(path)
            logger.info(f"Saved {split} to {path}")
            
            #Save human-readable samples
            human_readable_path = self.output_dir / f"{split}_readable.json"
            samples = []
            # Save first 100 samples (or all if less)
            num_samples = min(100, len(ds))
            for i in range(num_samples):
                decoded_input = self.tokenizer.decode(
                    ds[i]["input_ids"],
                    skip_special_tokens=False
                )
                samples.append({
                    "index": i,
                    "english_source": decoded_input.split("French:")[0].replace("English:", "").strip(),
                    "french_target": decoded_input.split("French:")[1].replace("</s>", "").strip(),
                    "full_prompt": decoded_input,
                })
            with open(human_readable_path, "w", encoding="utf-8") as f:
                json.dump(samples, f, indent=2, ensure_ascii=False)

            logger.info(f"Saved {num_samples} readable samples to {human_readable_path}")

        self.tokenizer.save_pretrained(self.output_dir / "tokenizer")

        with open(self.output_dir / "preprocessing_config.json", "w") as f:
            json.dump(
                {
                    "model": self.model_name,
                    "max_length": self.max_length,
                    "task": "English→French translation",
                    "label_masking": "causal LM (-100 on prompt)",
                    "dataset": "WMT14 fr-en", 
                    "benchmark": "Standard MT evaluation benchmark",
                },
                f,
                indent=2,
            )

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------
    def run(self, num_train_samples: int = 100000, debug_samples: Dict[str, int] = None):
        """
        Run preprocessing pipeline.
        
        Args:
            num_train_samples: Number of training samples 
            debug_samples: Optional dict to limit val/test for debugging
        """
        # Load dataset with subsampling
        dataset = self.load_dataset(num_train_samples)
        processed = {}

        # Process each split
        for split in ["train", "validation", "test"]:
            limit = debug_samples.get(split) if debug_samples else None
            processed[split] = self.preprocess_split(
                dataset[split],
                split,
                num_samples=limit
            )

        self.save(processed)
        logger.info("Preprocessing completed successfully.")


def main():
    preprocessor = WMT14Preprocessor() 
    
    # MODE 1: Debug phase (small subset)
    DEBUG_MODE = False
    
    if DEBUG_MODE:
        logger.info("=== DEBUG MODE: Using small subset ===")
        preprocessor.run(
            num_train_samples=100,  
            debug_samples={
                "validation": 20,
                "test": 20,
            }
        )
    else:
        # MODE 2: Production (500 training samples)
        logger.info("=== PRODUCTION MODE: Using 500 samples ===")
        preprocessor.run(
            num_train_samples=500,  
            debug_samples={
                "validation": 100,
                "test": 100,
            }
        )


if __name__ == "__main__":
    main()
