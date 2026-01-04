# End-to-End MLOps Pipeline for English→French Translation with Mistral-7B and QLoRA

## Overview
This project is an end-to-end MLOps pipeline for English→French translation using Mistral-7B-v0.3 fine-tuned with QLoRA (4-bit quantization + LoRA). The solution achieves high-quality translations on the WMT14 fr-en benchmark, 
enabling real-time interactive translation via a Gradio web demo. The pipeline is designed to be reproducible, scalable, and production-ready, with future plans for API deployment and Docker containerization.

<img width="1338" height="559" alt="1 FINE TUNING" src="https://github.com/user-attachments/assets/d7894672-67dc-4b62-b2b1-2024d8792ca6" />
<img width="1324" height="609" alt="2 FINE TUNING" src="https://github.com/user-attachments/assets/490ad932-ca29-40ac-bbe3-300a4478da6c" />
<img width="1326" height="561" alt="3FINE TUNING" src="https://github.com/user-attachments/assets/ceedfd6a-a6ce-4483-a4af-585f3287657a" />

## Model Performance
The fine-tuned Mistral-7B model achieved the following performance metrics on the WMT14 newstest2014 benchmark (100 samples):

  Metric       |   Score     |
 |-------------|-------------|
 | **SacreBLEU**|   34.81    |
 | **BLEU-1**  |     62.18   |
 | **BLEU-2**  |     41.12   |
 | **BLEU-3**  |     28.04   |
 | **BLEU-4**  |     20.49   |

## Interpretation

SacreBLEU (34.81): The model demonstrates strong translation quality, especially for a decoder-only architecture fine-tuned on a subsampled dataset.
BLEU-1 (62.18): High precision for unigram matches, indicating strong lexical overlap with reference translations.
BLEU-2/3/4: Progressive drop reflects the challenge of capturing multi-word expressions and longer dependencies, typical for transformer-based models on limited data.
Qualitative Analysis: Sample translations show high fluency and contextual accuracy, making the model suitable for real-world applications.

## Architecture
The pipeline consists of the following components:

### 1. Data Preprocessing

Cleaning: Minimal noise filtering (short/identical pairs removed).
Tokenization: Right-padded sequences (max length: 512) with Mistral-7B tokenizer.
Prompt Formatting: English: {en}\nFrench: for causal language modeling.

### 2. Model Training

QLoRA (4-bit): Efficient fine-tuning with nf4 quantization and bfloat16 compute.
LoRA Adapters: Targets q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj.
Hyperparameter Tuning: Learning rate (2e-4), batch size (4), and early stopping.

### 3. Evaluation

SacreBLEU: Official WMT14 metric for benchmarking.
Sample Translations: Qualitative analysis of model outputs (saved in evaluation_results/test/sample_translations.json).

### 4. Inference

Gradio Demo: Interactive web interface for real-time translation.
Batch Inference: Utilities for processing multiple inputs.


## Setup Instructions
Prerequisites

Python 3.8+
PyTorch + CUDA (for GPU acceleration)
Conda (recommended for environment management)
### Step 1: Clone the Repository :
```bash
git clone https://github.com/nadaben88/mistral-qlora-translation.git
cd mistral-qlora-translation
```
### Step 2: Create the Conda Environment
```bash
conda create -n mistral-qlora python=3.8
conda activate mistral-qlora
```
### Step 3: Install requirements
```bash
pip install -r requirements.txt
```
### Step 4: Run the Pipeline
```bash
# Preprocess data
python preprocess.py

# Fine-tune the model
python train.py

# Evaluate on WMT14 test set
python evaluate.py --mode test

# Launch the Gradio demo
python gradio_demo.py
```
## Future Improvements

The following features are planned for future development:
API Deployment

FastAPI Service: Deploy the model as a REST API for real-time predictions.

Docker Containerization

Docker Image: Containerize the API service for easy deployment and scalability.
Monitoring

Data Drift Detection: Use tools like Evidently to monitor data drift.
Model Performance Monitoring: Integrate Prometheus and Grafana for real-time monitoring.

## Contributing
Contributions are welcome! Please follow these steps:

Fork the repository.
Create a new branch (git checkout -b feature/your-feature).
Commit your changes (git commit -am 'Add your feature').
Push to the branch (git push origin feature/your-feature).
Open a Pull Request.

📜 License
This project is licensed under the MIT License. See the LICENSE file for details.

📬 Contact
For questions or feedback, please contact:
Nada Bentaouyt
nadabentaouyt@gmail.com
















 
