"""
Gradio Web Demo for English→French Translation
================================================

This script creates a web interface for the fine-tuned Mistral-7B
translation model using Gradio.

"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import gradio as gr
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_translation_model():
    """
    Load fine-tuned model with LoRA adapters.
    
    """
    logger.info("Loading model and tokenizer...")
    
    #Correct paths
    BASE_MODEL = "mistralai/Mistral-7B-v0.3"  
    ADAPTER_PATH = "./outputs/final_model"     
    
    # quantization config 
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,  # Use bfloat16 
        bnb_4bit_use_double_quant=True,
    )
    
    # Load base model
    logger.info(f"Loading base model: {BASE_MODEL}")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Load LoRA adapters
    logger.info(f"Loading LoRA adapters from: {ADAPTER_PATH}")
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    
    # Merge adapters for faster inference
    logger.info("Merging adapters...")
    model = model.merge_and_unload()
    model.eval()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    logger.info("Model loaded successfully!")
    return model, tokenizer


# Load model once at startup
model, tokenizer = load_translation_model()


def translate_en_to_fr(english_text):
    """
    Translate English text to French.
    
    Args:
        english_text: Input English text
        
    Returns:
        French translation
    """
    if not english_text or not english_text.strip():
        return ""
    
    #prompt format 
    prompt = f"English: {english_text}\nFrench:"
    
    # Tokenize
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        max_length=512,
        truncation=True
    ).to(model.device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            num_beams=1,           # Greedy for speed
            do_sample=False,       
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    #Extract French part only
    if "French:" in generated_text:
        translation = generated_text.split("French:")[-1].strip()
        # Stop at newlines or other languages
        for stop_marker in ["\nEnglish:", "\nDeutsch:", "\nItaliano:", "\n\n"]:
            if stop_marker in translation:
                translation = translation.split(stop_marker)[0].strip()
                break
    else:
        translation = generated_text.replace(english_text, "").strip()
    
    return translation


#examples and interface
demo = gr.Interface(
    fn=translate_en_to_fr,
    inputs=gr.Textbox(
        label="English Text",
        lines=5,
        placeholder="Enter English text to translate..."
    ),
    outputs=gr.Textbox(
        label="French Translation",
        lines=5
    ),
    title="🇬🇧 → 🇫🇷 English to French Translation",
    description=(
        "Fine-tuned Mistral-7B-v0.3 with QLoRA for English→French translation.\n\n"
        "**Model:** Mistral-7B-v0.3 + LoRA adapters\n"
        "**Dataset:** WMT14 fr-en (50K samples)\n"
        "**Method:** 4-bit QLoRA fine-tuning"
    ),
    examples=[
        ["Hello, how are you today?"],
        ["The weather is beautiful."],
        ["I love learning new languages."],
        ["This is a test sentence."],
        ["The regulation enters into force tomorrow."]
    ],
    theme=gr.themes.Soft(),
    allow_flagging="never",
)

if __name__ == "__main__":
    logger.info("Starting Gradio interface...")
    demo.launch(
        server_name="0.0.0.0",  # Listen on all interfaces
        server_port=7860,        # Default Gradio port
        share=True,              
        show_error=True,
    )