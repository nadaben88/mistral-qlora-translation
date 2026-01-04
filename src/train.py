"""
Mistral-7B LoRA Fine-Tuning for English→French Translation (WMT14)
====================================================================

This script fine-tunes Mistral-7B-v0.3 using QLoRA (4-bit quantization + LoRA)
on the WMT14 fr-en dataset with proper MT evaluation metrics.

"""

import os
import logging
from pathlib import Path
from typing import Optional, Dict
import torch
import numpy as np
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    TrainerCallback,
    EarlyStoppingCallback,
    EvalPrediction,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)
import evaluate
import warnings
warnings.filterwarnings("always")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CustomTrainer(Trainer):
    """
    Custom Trainer that keeps model in train mode during evaluation.
    """
    def evaluation_loop(self, *args, **kwargs):
        # Keep model in training mode during evaluation
        self.model.train()
        return super().evaluation_loop(*args, **kwargs)


class LossLoggingCallback(TrainerCallback):
    """Logs training loss at each logging step."""
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            logger.info(f"Step {state.global_step} | loss = {logs['loss']:.4f}")
        if logs and "eval_bleu" in logs:
            logger.info(f"Step {state.global_step} | eval_bleu = {logs['eval_bleu']:.2f}")


class SampleTranslationCallback(TrainerCallback):
    """Generates sample translations during training to monitor quality."""
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.samples = [
            "English: Hello, how are you?\nFrench:",
            "English: This is a test sentence.\nFrench:",
            "English: The weather is nice today.\nFrench:"
        ]

    def on_evaluate(self, args, state, control, **kwargs):
        """Generate samples after each evaluation."""
        if state.global_step > 0:
            model = kwargs.get("model")
            if model is None:
                return
                
            model.eval()
            logger.info(f"\n{'='*60}")
            logger.info(f"Sample Translations @ Step {state.global_step}")
            logger.info(f"{'='*60}")
            
            for s in self.samples:
                inputs = self.tokenizer(s, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    out = model.generate(
                        **inputs, 
                        max_new_tokens=40,
                        num_beams=1,  # Greedy for speed
                        do_sample=False,
                    )
                decoded = self.tokenizer.decode(out[0], skip_special_tokens=True)
                # Extract French part
                french = decoded.split("French:")[-1].split("\n")[0].strip()
                logger.info(f"Input: {s.split('French:')[0].strip()}")
                logger.info(f"Output: {french}\n")
            
            logger.info(f"{'='*60}\n")
            model.train()


class MistralTrainer:
    """
    Trainer class for Mistral-7B fine-tuning with QLoRA for MT.
    
    Implements:
    - 4-bit model quantization
    - LoRA adapter training
    - BLEU metric computation during training
    - Early stopping based on BLEU
    - Efficient memory management
    """
    
    def __init__(
        self,
        model_name: str = "mistralai/Mistral-7B-v0.3",
        data_dir: str = "./data",
        output_dir: str = "./outputs",
        use_wandb: bool = False,
        wandb_project: str = "mistral-wmt14-translation",
    ):
        """
        Initialize trainer.
        
        """
        self.model_name = model_name
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_wandb = use_wandb
        self.wandb_project = wandb_project
        
        # Check CUDA availability
        if not torch.cuda.is_available():
            logger.warning("CUDA not available! Training will be very slow on CPU.")
        else:
            logger.info(f"CUDA available. Using GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    def load_datasets(self):
        """
        Load preprocessed datasets.
    
        """
        logger.info("Loading preprocessed datasets...")
        
        train_dataset = load_from_disk(str(self.data_dir / "train"))
        eval_dataset = load_from_disk(str(self.data_dir / "validation"))
        test_dataset = load_from_disk(str(self.data_dir / "test"))
        
        logger.info(f"Train: {len(train_dataset)} samples")
        logger.info(f"Validation: {len(eval_dataset)} samples")
        logger.info(f"Test: {len(test_dataset)} samples")
        
        return train_dataset, eval_dataset, test_dataset
    
    def load_tokenizer(self):
        """
        Load tokenizer from preprocessed data.
        
        Returns:
            Tokenizer instance
        """
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            str(self.data_dir / "tokenizer"),
            use_fast=True,
        )
        
        # Ensure pad token is set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        return tokenizer
    
    def create_quantization_config(self):
        """
        Create 4-bit quantization configuration for QLoRA.
        
        Returns:
            BitsAndBytesConfig for 4-bit quantization
        
        """
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        
        logger.info("Created 4-bit quantization config (QLoRA)")
        return bnb_config
    
    def load_model(self, quantization_config):
        """
        Load Mistral-7B model with quantization.
        
        Args:
            quantization_config: BitsAndBytesConfig
            
        Returns:
            Loaded and quantized model
            
        
        """
        logger.info(f"Loading model: {self.model_name}")
        logger.info("This may take several minutes...")
        
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
        )
        
        logger.info(f"Model loaded successfully")
        logger.info(f"Model dtype: {model.dtype}")
        
        # Prepare model for k-bit training
        model = prepare_model_for_kbit_training(model)
        
        return model
    
    def create_lora_config(self):
        """
        Create LoRA configuration.
        
        Returns:
            LoraConfig for parameter-efficient fine-tuning
            
        """
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,  # Restore dropout for production 0.05
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ],
            bias="none",
        )
        
        logger.info("Created LoRA configuration:")
        logger.info(f"  Rank (r): {lora_config.r}")
        logger.info(f"  Alpha: {lora_config.lora_alpha}")
        logger.info(f"  Dropout: {lora_config.lora_dropout}")
        
        return lora_config
    
    def apply_lora(self, model, lora_config):
        """
        Apply LoRA adapters to model.
        
        Args:
            model: Base model
            lora_config: LoRA configuration
            
        Returns:
            Model with LoRA adapters
        """
        logger.info("Applying LoRA adapters to model...")
        
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        
        return model
    
    def create_compute_metrics(self, tokenizer):
        """
        Create compute_metrics function for BLEU evaluation during training.
        
        Args:
            tokenizer: Tokenizer for decoding
            
        Returns:
            compute_metrics function
            
        """
        bleu_metric = evaluate.load("sacrebleu")
        
        def compute_metrics(eval_preds: EvalPrediction) -> Dict[str, float]:
            """
            Compute BLEU score during evaluation.
            
            Args:
                eval_preds: Predictions and labels from evaluation
                
            Returns:
                Dictionary with BLEU score
            """
            preds, labels = eval_preds
            
            # Handle tuple output (some models return loss + logits)
            if isinstance(preds, tuple):
                preds = preds[0]
            
            # Get predicted token IDs (argmax over logits if needed)
            if len(preds.shape) == 3:  # (batch, seq_len, vocab_size)
                preds = np.argmax(preds, axis=-1)
            
            # Replace -100 in labels (masked prompt) with pad token
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
            
            # Decode predictions and labels
            decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
            
            # Extract French part only (after "French:")
            decoded_preds_clean = []
            decoded_labels_clean = []
            
            for pred, label in zip(decoded_preds, decoded_labels):
                # Extract French from prediction
                if "French:" in pred:
                    pred_french = pred.split("French:")[-1].split("\n")[0].strip()
                else:
                    pred_french = pred.strip()
                
                # Extract French from label
                if "French:" in label:
                    label_french = label.split("French:")[-1].split("\n")[0].strip()
                else:
                    label_french = label.strip()
                
                decoded_preds_clean.append(pred_french)
                decoded_labels_clean.append(label_french)
            
            # Compute BLEU
            result = bleu_metric.compute(
                predictions=decoded_preds_clean,
                references=[[label] for label in decoded_labels_clean]
            )
            
            return {"bleu": round(result["score"], 2)}
        
        return compute_metrics
    
    def create_training_arguments(
        self,
        num_train_samples: int,
        learning_rate: float = 2e-4,
        num_epochs: int = 3,
        batch_size: int = 4,
        gradient_accumulation_steps: int = 4,
        warmup_ratio: float = 0.03,
        weight_decay: float = 0.01,
        eval_steps: int = 5000,
        use_early_stopping: bool = True,
    ):
        """
        Create training arguments with BLEU-based evaluation.
            
        Returns:
            TrainingArguments instance
        """
        effective_batch_size = batch_size * gradient_accumulation_steps
        steps_per_epoch = num_train_samples // effective_batch_size
        
        logger.info(f"Training configuration:")
        logger.info(f"  Effective batch size: {effective_batch_size}")
        logger.info(f"  Steps per epoch: {steps_per_epoch}")
        logger.info(f"  Eval every: {eval_steps} steps")
        logger.info(f"  Weight Decay: {weight_decay}")

        
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            
            # Optimizer
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            optim="paged_adamw_8bit",
            
            # LR schedule
            lr_scheduler_type="cosine",
            warmup_ratio=warmup_ratio,
            
            # Evaluation and saving
            eval_strategy="steps",
            eval_steps=eval_steps,
            save_strategy="steps",
            save_steps=eval_steps,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",     
            greater_is_better=False,             
            
            # Logging
            logging_steps=100,
            logging_first_step=True,
            report_to="wandb" if self.use_wandb else "none",
            
            # Mixed precision
            bf16=True,
            
            # Memory optimization
            gradient_checkpointing=True,
            
            # Other
            seed=42,
            data_seed=42,
            remove_unused_columns=False,
            max_grad_norm=0.3,
        )
        
        return training_args
    
    def train(
        self,
        learning_rate: float = 2e-4,
        num_epochs: int = 3,
        batch_size: int = 4,
        gradient_accumulation_steps: int = 4,
        warmup_ratio: float = 0.03,
        weight_decay: float = 0.01,
        eval_steps: int = 5000,
        use_early_stopping: bool = True,
        early_stopping_patience: int = 3,
    ):
        """
        Run full training pipeline with BLEU evaluation.
        
        Args:
            learning_rate: Learning rate
            num_epochs: Number of epochs
            batch_size: Per-device batch size
            gradient_accumulation_steps: Gradient accumulation
            warmup_ratio: Warmup ratio
            weight_decay: Weight decay
            eval_steps: Evaluation frequency
            use_early_stopping: Whether to use early stopping
            early_stopping_patience: Patience for early stopping
        """
        # Clear GPU cache
        torch.cuda.empty_cache()
        
        # Initialize W&B
        if self.use_wandb:
            import wandb
            wandb.init(
                project=self.wandb_project,
                name=f"mistral-7b-lora-wmt14",
                config={
                    "model": self.model_name,
                    "dataset": "WMT14 fr-en",
                    "learning_rate": learning_rate,
                    "num_epochs": num_epochs,
                    "batch_size": batch_size,
                    "gradient_accumulation": gradient_accumulation_steps,
                }
            )
        
        # Load data
        train_dataset, eval_dataset, test_dataset = self.load_datasets()
        tokenizer = self.load_tokenizer()
        
        # Load model
        quantization_config = self.create_quantization_config()
        model = self.load_model(quantization_config)
        
        # Apply LoRA
        lora_config = self.create_lora_config()
        model = self.apply_lora(model, lora_config)
        
        # Create training arguments
        training_args = self.create_training_arguments(
            num_train_samples=len(train_dataset),
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_ratio=warmup_ratio,
            weight_decay=weight_decay,
            eval_steps=eval_steps,
            use_early_stopping=use_early_stopping,
        )
        
        # Create data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
        )
        
        # Create compute_metrics function
        compute_metrics = self.create_compute_metrics(tokenizer)
        
        # Create callbacks
        callbacks = [
            LossLoggingCallback(),
            #SampleTranslationCallback(tokenizer),
        ]
        
        if use_early_stopping:
            callbacks.append(
                EarlyStoppingCallback(
                    early_stopping_patience=early_stopping_patience,
                    early_stopping_threshold=0.0,
                )
            )
            logger.info(f"Early stopping enabled (patience={early_stopping_patience})")
        
        # Create Trainer
        trainer = CustomTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            callbacks=callbacks,
        )
        
        # Train
        logger.info("\n" + "="*60)
        logger.info("Starting training with BLEU evaluation...")
        logger.info("="*60 + "\n")
        
        trainer.train()
        
        # Save final model
        logger.info("\nSaving final model...")
        final_model_path = self.output_dir / "final_model"
        trainer.save_model(str(final_model_path))
        tokenizer.save_pretrained(str(final_model_path))
        logger.info(f"Model saved to {final_model_path}")
        
        # Close W&B
        if self.use_wandb:
            import wandb
            wandb.finish()
        
        logger.info("\n" + "="*60)
        logger.info("Training complete!")
        logger.info("="*60)


def main():
    """Main entry point with DEBUG and PRODUCTION modes."""
    
    # ====================================================================
    # CONFIGURATION
    # ====================================================================
    MODEL_NAME = "mistralai/Mistral-7B-v0.3"
    DATA_DIR = "./data"
    OUTPUT_DIR = "./outputs"
    
    # ====================================================================
    # MODE SELECTION
    # ====================================================================
    DEBUG_MODE = False  # Set to False for production training
    
    if DEBUG_MODE:
        logger.info("="*60)
        logger.info("DEBUG MODE: Small dataset, fast overfitting test")
        logger.info("="*60)
        
        # Debug hyperparameters (overfit on small data)
        LEARNING_RATE = 2e-4
        NUM_EPOCHS = 40
        BATCH_SIZE = 1
        GRADIENT_ACCUMULATION = 2
        WARMUP_RATIO = 0.03
        WEIGHT_DECAY = 0.0
        EVAL_STEPS = 10  # Evaluate frequently
        USE_EARLY_STOPPING = False  # Don't use for debug
        
    else:
        logger.info("="*60)
        logger.info("PRODUCTION MODE: Full WMT14 training with BLEU evaluation")
        logger.info("="*60)
        
        # Production hyperparameters
        LEARNING_RATE = 2e-4
        NUM_EPOCHS = 10
        BATCH_SIZE = 4
        GRADIENT_ACCUMULATION = 4
        WARMUP_RATIO = 0.03
        WEIGHT_DECAY = 0.01
        EVAL_STEPS = 5000  # Every 500 steps
        USE_EARLY_STOPPING = True  # Enable early stopping
    
    # W&B settings
    USE_WANDB = False
    WANDB_PROJECT = "mistral-wmt14-translation"
    
    # Initialize trainer
    trainer = MistralTrainer(
        model_name=MODEL_NAME,
        data_dir=DATA_DIR,
        output_dir=OUTPUT_DIR,
        use_wandb=USE_WANDB,
        wandb_project=WANDB_PROJECT,
    )
    
    # Run training
    trainer.train(
        learning_rate=LEARNING_RATE,
        num_epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        eval_steps=EVAL_STEPS,
        use_early_stopping=USE_EARLY_STOPPING,
        early_stopping_patience=3,
    )


if __name__ == "__main__":
    main()