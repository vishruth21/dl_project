This Colab notebook is a deep learning project for sentiment analysis on news headlines using a large language model (Llama-2-7B-Chat-HF).
It mainly does 4 big things:

Environment Setup

It installs libraries like bitsandbytes, transformers, peft, trl, and manages CUDA/GPU settings.

It uses 4-bit quantization (bitsandbytes) to load huge models into limited GPU memory.

Data Loading and Preprocessing

Loads a dataset (all-data.csv) with two columns: sentiment (positive, neutral, negative) and text (the news headline).

Splits the data into:

Training (300 examples per class)

Testing (300 examples per class)

Evaluation (50 examples per class, sampled with replacement).

Prepares prompts like:

pgsql
Copy
Edit
Analyze the sentiment of [news text] and classify as positive, neutral, or negative.
Model Preparation and Fine-tuning

Downloads a pretrained Llama-2-7B-Chat model.

Fine-tunes it using QLoRA (Quantized Low-Rank Adaptation), which saves memory and speeds up training.

Uses training arguments such as:

Learning rate = 2e-4

Batch size = 4

1 training epoch

Constant LR scheduler

4-bit quantization

Saving checkpoints every 25 steps

Evaluation

After fine-tuning, the model's predictions are compared against true labels using:

Accuracy

Classification report (Precision, Recall, F1-score)

Confusion matrix

