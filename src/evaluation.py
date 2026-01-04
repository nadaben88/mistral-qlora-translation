"""
Model Evaluation Script: WMT14 Benchmark Compliance
====================================================

Evaluates fine-tuned Mistral-7B on English→French translation using
WMT14 standard evaluation protocol with SacreBLEU.

Modes:
- DEBUG: Evaluates on training data (overfitting verification only)
- TEST: Evaluates on WMT14 newstest2014 (official benchmark)

Official WMT14 Evaluation:
- Metric: SacreBLEU only (standard for MT benchmarks)
- Test set: newstest2014 
- No post-processing filters (detokenization handled by SacreBLEU)

"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm

import torch
from datasets import load_from_disk, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import evaluate

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class WMT14Evaluator:
    """
    WMT14-compliant evaluator for English→French translation.
    
    Implements:
    - Model loading (base + LoRA adapters)
    - Translation generation
    - SacreBLEU computation (official WMT metric)
    - Debug mode (train data) and test mode (newstest2014)
    
   
    """
    
    def __init__(
        self,
        base_model_name: str,
        adapter_path: str,
        data_dir: str,
        output_dir: str = "./evaluation_results",
        max_new_tokens: int = 128,
    ):
        """
        Initialize evaluator.
        
    
        """
        self.base_model_name = base_model_name
        self.adapter_path = Path(adapter_path)
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_new_tokens = max_new_tokens
        
        # Check device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        # Load SacreBLEU metric (official MT metric)
        logger.info("Loading SacreBLEU metric...")
        self.bleu_metric = evaluate.load("sacrebleu")
        logger.info("Metric loaded: SacreBLEU (official WMT metric)")
    
    def load_model_and_tokenizer(self):
        """
        Load base model, LoRA adapters, and tokenizer.
        
        """
        logger.info(f"Loading base model: {self.base_model_name}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_name,
            use_fast=True,
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        # Load base model with 4-bit quantization
        logger.info("Loading base model with 4-bit quantization...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            quantization_config=bnb_config,
            device_map="auto",
        )
        
        # Load LoRA adapters
        logger.info(f"Loading LoRA adapters from: {self.adapter_path}")
        model = PeftModel.from_pretrained(model, str(self.adapter_path))
        
        # Merge adapters for faster inference
        logger.info("Merging LoRA adapters...")
        model = model.merge_and_unload()
        model.eval()
        
        logger.info("Model loaded successfully")
        return model, tokenizer
    
    def load_evaluation_dataset(self, mode: str = "debug"):
        
        logger.info(f"Loading evaluation dataset (mode: {mode})...")
        
        if mode == "debug":
            # Debug: Load training data from disk
            dataset = load_from_disk(str(self.data_dir / "train"))
            logger.info(f"Loaded TRAIN data: {len(dataset)} examples")
            logger.info("Purpose: Overfitting verification (NOT a benchmark)")
            
        elif mode == "test":
            # Test: Load official WMT14 newstest2014
            dataset = load_dataset("wmt14", "fr-en", split="test")
            logger.info(f"Loaded WMT14 newstest2014: {len(dataset)} examples")
            logger.info("Purpose: Official benchmark evaluation")
            
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'debug' or 'test'")
        
        return dataset
    
    def extract_translation(self, generated_text: str) -> str:
        # Split at "French:" and take everything after
        if "French:" in generated_text:
            translation = generated_text.split("French:")[-1].strip()
            
            # Stop at first newline 
            if "\n" in translation:
                translation = translation.split("\n")[0].strip()
            
            return translation
        
        # Fallback: return full text 
        return generated_text.strip()
    
    def generate_translation(self, model, tokenizer, source_text: str) -> str:
       
        # Format as training: "English: X\nFrench:"
        prompt = f"English: {source_text}\nFrench:"
        
        # Tokenize
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(model.device)
        
        # Generate 
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                num_beams=1,           
                do_sample=False,       
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        # Decode
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract translation
        translation = self.extract_translation(generated_text)
        
        return translation
    
    def generate_translations_batch(
        self,
        model,
        tokenizer,
        dataset,
        num_samples: int = None,
        mode: str = "debug",
    ) -> Dict[str, List[str]]:
        """
        Generate translations for entire dataset.
        
        """
        logger.info("Generating translations...")
        
        # Limit samples if specified
        if num_samples is not None:
            dataset = dataset.select(range(min(num_samples, len(dataset))))
            logger.info(f"Evaluating on {len(dataset)} samples")
        
        predictions = []
        references = []
        sources = []
        
        # Generate translations
        for example in tqdm(dataset, desc="Translating"):
            try:
                if mode == "debug":
                    # Debug mode
                    if 'input_ids' in example:
                        full_text = tokenizer.decode(
                            example['input_ids'], 
                            skip_special_tokens=True
                        )
                        
                        # Parse "English: X\nFrench: Y"
                        if "English:" in full_text and "French:" in full_text:
                            parts = full_text.split("French:")
                            source = parts[0].replace("English:", "").strip()
                            reference = parts[1].strip()
                        else:
                            logger.warning(f"Unexpected format: {full_text[:100]}")
                            continue
                    else:
                        continue
                        
                elif mode == "test":
                    # Test mode: Use WMT14 format
                    source = example['translation']['en']
                    reference = example['translation']['fr']
                
                # Generate prediction
                prediction = self.generate_translation(model, tokenizer, source)
                
                # Debug: Print first 3 samples
                if len(predictions) < 3:
                    logger.info(f"\n{'='*60}")
                    logger.info(f"Sample {len(predictions) + 1}")
                    logger.info(f"{'='*60}")
                    logger.info(f"Source (EN): {source[:100]}...")
                    logger.info(f"Prediction (FR): {prediction[:100]}...")
                    logger.info(f"Reference (FR): {reference[:100]}...")
                    logger.info(f"{'='*60}\n")
                
                predictions.append(prediction)
                references.append(reference)
                sources.append(source)
                
            except Exception as e:
                logger.error(f"Error processing example: {e}")
                continue
        
        logger.info(f"Successfully generated {len(predictions)} translations")
        
        return {
            'predictions': predictions,
            'references': references,
            'sources': sources,
        }
    
    def compute_bleu(
        self,
        predictions: List[str],
        references: List[str],
    ) -> Dict[str, float]:
    
        
        logger.info("Computing BLEU score (SacreBLEU)...")
        
        # Format references (SacreBLEU expects list of lists)
        references_formatted = [[ref] for ref in references]
        
        # Compute BLEU
        bleu_results = self.bleu_metric.compute(
            predictions=predictions,
            references=references_formatted,
        )
        
        logger.info(f"BLEU score: {bleu_results['score']:.2f}")
        
        return {
            'bleu': bleu_results['score'],
            'bleu_1': bleu_results['precisions'][0],
            'bleu_2': bleu_results['precisions'][1],
            'bleu_3': bleu_results['precisions'][2],
            'bleu_4': bleu_results['precisions'][3],
        }
    
    def save_results(
        self,
        metrics: Dict[str, float],
        translations: Dict[str, List[str]],
        mode: str,
    ):
        """
        Save evaluation results.
        
        """
        # Create mode-specific directory
        mode_dir = self.output_dir / mode
        mode_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metrics
        metrics_path = mode_dir / "metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump({**metrics, "mode": mode}, f, indent=2)
        logger.info(f"Saved metrics to {metrics_path}")
        
        # Save sample translations (first 50)
        samples_path = mode_dir / "sample_translations.json"
        num_samples = min(50, len(translations['predictions']))
        samples = {
            'num_samples': num_samples,
            'mode': mode,
            'translations': [
                {
                    'source': translations['sources'][i],
                    'prediction': translations['predictions'][i],
                    'reference': translations['references'][i],
                }
                for i in range(num_samples)
            ]
        }
        with open(samples_path, 'w', encoding='utf-8') as f:
            json.dump(samples, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved sample translations to {samples_path}")
        
        # Save all predictions (for analysis)
        predictions_path = mode_dir / "all_predictions.txt"
        with open(predictions_path, 'w', encoding='utf-8') as f:
            for pred in translations['predictions']:
                f.write(pred + '\n')
        logger.info(f"Saved all predictions to {predictions_path}")
    
    def evaluate(self, num_samples: int = None, mode: str = "debug"):
        """
        Run full evaluation pipeline.
        
        Args:
            num_samples: Number of samples to evaluate (None for all)
            mode: "debug" (train data) or "test" (WMT14 newstest2014)
        """
        logger.info("\n" + "="*70)
        logger.info(f"WMT14 Evaluation Pipeline - Mode: {mode.upper()}")
        logger.info("="*70)
        
        # Load model and tokenizer
        model, tokenizer = self.load_model_and_tokenizer()
        
        # Load evaluation dataset
        eval_dataset = self.load_evaluation_dataset(mode=mode)
        
        # Generate translations
        translations = self.generate_translations_batch(
            model, tokenizer, eval_dataset, num_samples, mode=mode
        )
        
        # Check if we have translations
        if len(translations['predictions']) == 0:
            logger.error("No translations generated!")
            return
        
        # Compute BLEU (official WMT metric)
        logger.info("\n" + "="*70)
        logger.info("Computing Evaluation Metrics (SacreBLEU)")
        logger.info("="*70)
        
        bleu_scores = self.compute_bleu(
            translations['predictions'],
            translations['references'],
        )
        
        # Add sample count
        metrics = {
            **bleu_scores,
            'num_samples': len(translations['predictions']),
            'mode': mode,
        }
        
        # Save results
        self.save_results(metrics, translations, mode)
        
        # Print summary
        logger.info("\n" + "="*70)
        logger.info(f"Evaluation Results - Mode: {mode.upper()}")
        logger.info("="*70)
        logger.info(f"Number of samples: {metrics['num_samples']}")
        logger.info(f"\nSacreBLEU Scores:")
        logger.info(f"  BLEU: {metrics['bleu']:.2f}")
        logger.info(f"  BLEU-1: {metrics['bleu_1']:.2f}")
        logger.info(f"  BLEU-2: {metrics['bleu_2']:.2f}")
        logger.info(f"  BLEU-3: {metrics['bleu_3']:.2f}")
        logger.info(f"  BLEU-4: {metrics['bleu_4']:.2f}")
        
        # Interpretation
        if mode == "debug":
            logger.info("\n" + "="*70)
            logger.info("Debug Mode Interpretation")
            logger.info("="*70)
            if metrics['bleu'] > 80:
                logger.info(" EXCELLENT: Model memorized training data")
                logger.info("   → Ready for full dataset training")
            elif metrics['bleu'] > 50:
                logger.info("  MODERATE: Partial learning")
                logger.info("   → Check: epochs, learning rate, data format")
            else:
                logger.info(" FAILED: Model did not learn")
                logger.info("   → Check: preprocessing, labels, model setup")
        
        elif mode == "test":
            logger.info("\n" + "="*70)
            logger.info("WMT14 Benchmark Results")
            logger.info("="*70)
            logger.info("  • My model: {:.2f} BLEU".format(metrics['bleu']))
            logger.info("\nNote: Decoder-only models (Mistral) typically")
        
        logger.info("="*70)


def main():
    """Main entry point."""
    
    # Configuration
    BASE_MODEL = "mistralai/Mistral-7B-v0.3"
    ADAPTER_PATH = "./outputs/final_model"
    DATA_DIR = "./data"
    OUTPUT_DIR = "./evaluation_results"
    MAX_NEW_TOKENS = 128 
    
    
    
    MODE = "test"  
    
    # Number of samples 
    #NUM_SAMPLES = None  # Evaluate all samples
    
    # For quick testing:
    NUM_SAMPLES = 100
    
    # Initialize evaluator
    evaluator = WMT14Evaluator(
        base_model_name=BASE_MODEL,
        adapter_path=ADAPTER_PATH,
        data_dir=DATA_DIR,
        output_dir=OUTPUT_DIR,
        max_new_tokens=MAX_NEW_TOKENS,
    )
    
    # Run evaluation
    logger.info("Starting WMT14 evaluation pipeline...")
    evaluator.evaluate(num_samples=NUM_SAMPLES, mode=MODE)
    
    logger.info(f"\n Evaluation complete!")
    logger.info(f"Results saved to: {OUTPUT_DIR}/{MODE}/")


if __name__ == "__main__":
    main()