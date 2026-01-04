# End-to-End MLOps Pipeline for English→French Translation with Mistral-7B and QLoRA

## Overview
This project is an end-to-end MLOps pipeline for English→French translation using Mistral-7B-v0.3 fine-tuned with QLoRA (4-bit quantization + LoRA). The solution achieves high-quality translations on the WMT14 fr-en benchmark, 
enabling real-time interactive translation via a Gradio web demo. The pipeline is designed to be reproducible, scalable, and production-ready, with future plans for API deployment and Docker containerization.
## Model Performance
The fine-tuned Mistral-7B model achieved the following performance metrics on the WMT14 newstest2014 benchmark (100 samples):

  Metric       |   Score     |
 |-------------|-------------|
 | **SacreBLEU**|   34.81    |
 | **BLEU-1**  |     62.18   |
 | **BLEU-2**  |     41.12   |
 | **BLEU-3**  |     28.04   |
 | **BLEU-4**  |     20.49   |



 
