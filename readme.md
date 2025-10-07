# Medical Review Sentiment Analysis using LoRA and RoBERTa

This project fine-tunes a RoBERTa transformer model using LoRA (Low-Rank Adaptation) for multi-class sentiment analysis on Drug.com medical reviews.  
The task classifies patient drug reviews into three sentiment categories:
- 0 → Negative  
- 1 → Neutral  
- 2 → Positive  

LoRA enables efficient fine-tuning with minimal GPU memory usage while maintaining near full fine-tuning performance.

---

## Project Overview

- **Model Base:** [cardiffnlp/twitter-roberta-base-sentiment](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment)  
- **Fine-tuning Method:** LoRA (Parameter-Efficient Fine-Tuning)  
- **Dataset:** [Drug Review Dataset (UCI Repository)](https://archive.ics.uci.edu/ml/datasets/drug+review+dataset+(drugs.com))  
- **Goal:** Sentiment classification of medical drug reviews  
- **Accuracy Achieved:** ~79.23%  
- **Framework:** PyTorch, Hugging Face Transformers, PEFT  

---

## Model Architecture

- **Base Model:** Encoder-only transformer (RoBERTa-base)  
- **LoRA Applied To:** `query`, `value` layers of the attention heads  
- **Task Head:** 3-class classification layer  

---

## Features

- Complete NLP pipeline: preprocessing, tokenization, fine-tuning, and evaluation  
- Parameter-efficient fine-tuning using LoRA  
- Sentiment grouping from numerical ratings  
- Accuracy and confusion matrix evaluation  
- Model checkpoint saving for reusability  

---

## LoRA Configuration

| Hyperparameter | Value | Description |
|----------------|--------|-------------|
| r | 16 | Rank of LoRA decomposition |
| lora_alpha | 32 | Scaling factor for LoRA updates |
| lora_dropout | 0.05 | Dropout applied on LoRA layers |
| target_modules | ["query", "value"] | Attention projections to fine-tune |
| lr | 2e-5 | Learning rate |
| epochs | 2 | Training epochs |
| batch_size | 16 | Samples per batch |

This configuration achieves a good balance between accuracy and efficiency.

---

## Data Preprocessing

Each review undergoes:
1. HTML tag removal using BeautifulSoup  
2. URL, username, and special character cleanup  
3. Extra whitespace removal  
4. Mapping of numerical ratings to sentiment classes:  
   - 1–3 → Negative  
   - 4–7 → Neutral  
   - 8–10 → Positive  

---

## Results

| Metric | Value |
|--------|--------|
| Accuracy | ~79.23% |
| Fine-tuning Method | LoRA |
| GPU Memory Usage | ~60% lower than full fine-tuning |
| Training Time | ~40–50% faster |

Example confusion matrix:

