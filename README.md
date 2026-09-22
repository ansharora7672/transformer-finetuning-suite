# Transformer Fine-Tuning Suite: Classification, QA, and Text Generation

Three separate transformer fine-tuning tasks using Hugging Face `transformers`, each on a different NLP problem type: sentiment classification, extractive question answering, and causal text generation.

Originally a group project (Ansh, Henil, Kahan) for *PROG74040 — Advanced Topics in AI and ML*. This copy is maintained for portfolio purposes.

## What's inside

- `1_sentiment_classification_distilbert.ipynb` — fine-tunes DistilBERT on a subset of IMDB movie reviews for binary sentiment classification, benchmarked against the off-the-shelf `pipeline` baseline. Reports accuracy and macro F1 on a held-out set.
- `2_extractive_qa_distilbert_squad.ipynb` — fine-tunes DistilBERT on SQuAD for extractive question answering, including the character-to-token offset mapping needed to convert answer spans into start/end token positions.
- `3_text_generation_gpt2.ipynb` — fine-tunes GPT-2 as a causal language model on AG News articles to adopt a consistent "news anchor" writing style, then compares generations against the base pretrained model.

## Tech stack

Python, Hugging Face `transformers`, `datasets`, `accelerate`, PyTorch

## Run it

```bash
pip install transformers datasets accelerate torch scikit-learn
jupyter notebook 1_sentiment_classification_distilbert.ipynb   # or 2_/3_
```

Datasets (IMDB, SQuAD, AG News) download automatically from the Hugging Face Hub on first run.
