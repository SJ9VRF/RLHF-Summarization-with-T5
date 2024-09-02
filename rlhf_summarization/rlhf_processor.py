# Contains RLHFProcessor class
import random
import numpy as np
import torch
from transformers import (
    AutoTokenizer, T5ForConditionalGeneration, Seq2SeqTrainer, Seq2SeqTrainingArguments, Trainer, AutoModel
)
from datasets import load_dataset
from torch import nn
from torch.utils.data import RandomSampler
from dataclasses import dataclass
from typing import Optional, Union, Tuple, Any

@dataclass
class RankingModelOutput:
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class RankingModel(nn.Module):
    def __init__(self, model_name="distilroberta-base"):
        super().__init__()
        self.roberta = AutoModel.from_pretrained(model_name)
        self.regressor = nn.Linear(self.roberta.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]
        rewards = self.regressor(pooled_output)
        return rewards


class RLHFProcessor:
    def __init__(self, model_name="t5-small"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.model.config.max_length = 64
        self.data_collator = None
        self.reward_model = None
        self.reward_model_tokenizer = None

    def load_data(self):
        dataset_labeled_train, dataset_rank_train, dataset_unlabeled_train, dataset_validation, dataset_test = load_dataset(
            "some_dataset_name", split=["train[:1%]", "train[1%:2%]", "train[2%:10%]", "validation", "test"]
        )
        return dataset_labeled_train, dataset_rank_train, dataset_unlabeled_train, dataset_validation, dataset_test

    def tokenize_data(self, examples, remove_target=False):
        inputs = [doc for doc in examples["document"]]
        model_inputs = self.tokenizer(inputs, max_length=1024, truncation=True)
        if not remove_target:
            labels = self.tokenizer(examples["summary"], max_length=128, truncation=True)
            model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def fine_tune_model(self, dataset_labeled_train, dataset_validation):
        training_args = Seq2SeqTrainingArguments(
            output_dir="./outputs",
            learning_rate=1e-4,
            per_device_train_batch_size=16,
            num_train_epochs=3,
            logging_dir="./logs",
            fp16=True,
            predict_with_generate=True,
        )
        
        self.data_collator = DataCollatorForSeq2Seq(tokenizer=self.tokenizer, model=self.model)

        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset_labeled_train,
            eval_dataset=dataset_validation,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
        )
        
        trainer.train()
        return trainer

    def generate_hypotheses(self, trainer, dataset):
        output_dataset = [{"id": item["id"], "document": item["document"], "hypothesis": []} for item in dataset]
        
        for choice_id in range(4):
            generated_hypothesis = []
            dataloader = trainer.get_test_dataloader(dataset)
            for inputs in dataloader:
                inputs = trainer._prepare_inputs(inputs)
                generation_inputs = inputs[trainer.model.main_input_name]
                generated_tokens = trainer.model.generate(
                    generation_inputs,
                    attention_mask=inputs.get("attention_mask", None),
                    num_beams=1,
                    do_sample=True  # random sampling
                )
                generated_hypothesis.extend(
                    self.tokenizer.batch_decode(
                        generated_tokens,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=True
                    )
                )
            for i in range(len(generated_hypothesis)):
                output_dataset[i]["hypothesis"].append(generated_hypothesis[i])
        return output_dataset

    def rank_hypotheses(self, dataset_orig, dataset_for_ranking):
        result = []
        for item0, item1 in zip(dataset_orig, dataset_for_ranking):
            scores = []
            w1 = 0.65 + np.random.rand() * 0.1  # 0.65 to 0.85
            w2 = 0.15 * np.random.rand() * 0.1  # 0.15 to 0.25
            w3 = 1.0 - w1 - w2
            for i in range(4):
                m = rouge.compute(
                    predictions=[item0['summary']], references=[item1['hypothesis'][i]], use_stemmer=True
                )
                scores.append(m['rouge2'] * w1 + m['rougeL'] * w2 + m['rouge1'] * w3 + np.random.rand() * 0.001)
            item2 = item1.copy()
            item2['hypothesis'] = [item1['hypothesis'][j] for j in np.argsort(scores)[::-1]]
            result.append(item2)
        return result

    def train_reward_model(self, dataset_for_ranking_train, dataset_for_ranking_validation):
        self.reward_model_tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")
        self.reward_model = RankingModel(model_name="distilroberta-base")

        tokenize_ranking_fn = functools.partial(self.tokenize_data, self.reward_model_tokenizer)
        dataset_for_ranking_train = dataset_for_ranking_train.map(
            tokenize_ranking_fn, batched=True, remove_columns=dataset_for_ranking_train.column_names
        )
        dataset_for_ranking_validation = dataset_for_ranking_validation.map(
            tokenize_ranking_fn, batched=True, remove_columns=dataset_for_ranking_validation.column_names
        )

        reward_model_data_collator = DataCollatorWithPadding(tokenizer=self.reward_model_tokenizer, pad_to_multiple_of=8)

        training_args = TrainingArguments(
            output_dir="./rank_model_outs",
            learning_rate=1.5e-5,
            weight_decay=0.01,
            warmup_ratio=0.1,
            per_device_train_batch_size=32,
            gradient_accumulation_steps=2,
            num_train_epochs=3,
            logging_dir="./logs",
            fp16=True,
            evaluation_strategy="epoch"
        )

        trainer = Trainer(
            model=self.reward_model,
            args=training_args,
            train_dataset=dataset_for_ranking_train,
            eval_dataset=dataset_for_ranking_validation,
            tokenizer=self.reward_model_tokenizer,
            data_collator=reward_model_data_collator
        )

        trainer.train()
        return trainer

    def reinforcement_learning(self, dataset_unlabeled_train, dataset_validation):
        training_args = Seq2SeqTrainingArguments(
            output_dir="./rlhf_model_outs",
            learning_rate=1e-4,
            weight_decay=0.01,
            warmup_ratio=0.1,
            per_device_train_batch_size=16,
            max_steps=2000,
            logging_dir="./logs",
            fp16=True,
            predict_with_generate=True,
        )

        self.data_collator = DataCollatorForRLHF(tokenizer=self.tokenizer, model=self.model)

        trainer = RLHFSeq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset_unlabeled_train,
            eval_dataset=dataset_validation,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
        )

        trainer.prepare_reward_model(self.reward_model, self.reward_model_tokenizer)

        trainer.train()
