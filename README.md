# RLHF Summarization with T5
RLHF-Based Summarization with T5: Implementing Human-Guided Model Training

![Screenshot_2024-08-31_at_2 07 02_AM-removebg-preview](https://github.com/user-attachments/assets/ecc15f17-6b02-4798-8feb-b0bd24d27532)



# RLHF Summarization

This repository contains an implementation of **Reinforcement Learning from Human Feedback (RLHF)** for text summarization using the T5 model. It allows for supervised fine-tuning, reward model training, and reinforcement learning to improve summarization quality through human feedback.

## Features

- **Supervised Fine-tuning**: Fine-tune the T5 model on labeled summarization datasets.
- **Reward Model Training**: Train a reward model based on human feedback (or simulated feedback) using the Distill-RoBERTa model.
- **Reinforcement Learning**: Use the reward model to guide the T5 model in generating higher-quality summaries.
- **Customizable**: Easily switch models, datasets, and evaluation metrics.

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/rlhf_summarization.git
   cd rlhf_summarization
   ```

2. **Install the required dependencies:**

   You can install all the dependencies using the following command:

   ```bash
   pip install -r requirements.txt
   ```

3. **Install the package locally:**

   To install the package in "editable" mode (useful for development), run:

   ```bash
   pip install -e .
   ```

## Usage

### 1. Fine-Tuning with Supervised Learning

You can fine-tune the T5 model on your dataset using the `train.py` script. Example:

```bash
python train.py --model_name t5-small --num_epochs 3 --batch_size 16 --dataset_name xsum --output_dir ./outputs
```

### 2. Train the Reward Model

After generating hypotheses and obtaining ranked data, you can train the reward model with:

```bash
python train.py --train_reward_model
```

### 3. Evaluate the Model

To evaluate the fine-tuned or RLHF-improved model on the test dataset:

```bash
python evaluate.py --model_name t5-small --checkpoint_path ./outputs/checkpoint-1000 --output_dir ./eval_outputs
```

### 4. Run Reinforcement Learning

After training the reward model, you can apply reinforcement learning to improve the summarization model:

```bash
python train.py --reinforcement_learning
```

## Example

```bash
# Fine-tune the model
python train.py --model_name t5-small --num_epochs 3 --batch_size 16 --dataset_name xsum --output_dir ./outputs

# Evaluate the model
python evaluate.py --model_name t5-small --checkpoint_path ./outputs/checkpoint-1000 --output_dir ./eval_outputs
```

## Datasets

This project uses the Huggingface Datasets library, making it easy to load common summarization datasets like `xsum`, `cnn_dailymail`, etc. You can specify which dataset to use via command-line arguments or by modifying the dataset loading code.

## Dependencies

The main dependencies for this project are:

- `transformers`
- `datasets`
- `torch`
- `tokenizers`
- `rouge-score`
- `evaluate`
- `einops`
- `numpy`
- `pandas`

See the full list of dependencies in the `requirements.txt` file.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

This project is based on the work from the InstructGPT paper and inspired by reinforcement learning methods for natural language processing tasks.
