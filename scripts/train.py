# Script to run the training process
import argparse
from rlhf_summarization.rlhf_processor import RLHFProcessor

def parse_args():
    parser = argparse.ArgumentParser(description="Train a summarization model with RLHF.")
    
    parser.add_argument("--model_name", type=str, default="t5-small", help="Name of the model to use.")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training.")
    parser.add_argument("--dataset_name", type=str, default="xsum", help="Dataset to use.")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Directory to save the model outputs.")
    
    args = parser.parse_args()
    return args

def main():
    # Parse command-line arguments
    args = parse_args()

    # Initialize RLHF Processor
    processor = RLHFProcessor(model_name=args.model_name)

    # Load datasets
    dataset_labeled_train, dataset_rank_train, dataset_unlabeled_train, dataset_validation, dataset_test = processor.load_data()

    # Tokenize datasets
    dataset_labeled_train = dataset_labeled_train.map(processor.tokenize_data, batched=True)
    dataset_rank_train = dataset_rank_train.map(lambda examples: processor.tokenize_data(examples, remove_target=True), batched=True)
    dataset_unlabeled_train = dataset_unlabeled_train.map(lambda examples: processor.tokenize_data(examples, remove_target=True), batched=True)
    dataset_validation = dataset_validation.map(processor.tokenize_data, batched=True)
    dataset_test = dataset_test.map(processor.tokenize_data, batched=True)

    # Fine-tune the summarization model
    print("Starting supervised fine-tuning...")
    trainer = processor.fine_tune_model(dataset_labeled_train, dataset_validation)

    # Generate hypotheses for ranking
    print("Generating hypotheses...")
    dataset_for_ranking_train = processor.generate_hypotheses(trainer, dataset_rank_train)
    dataset_for_ranking_validation = processor.generate_hypotheses(trainer, dataset_validation)

    # Rank hypotheses using simulated human feedback
    print("Ranking hypotheses...")
    ranked_train = processor.rank_hypotheses(dataset_rank_train, dataset_for_ranking_train)
    ranked_validation = processor.rank_hypotheses(dataset_validation, dataset_for_ranking_validation)

    # Train the reward model
    print("Training the reward model...")
    reward_trainer = processor.train_reward_model(ranked_train, ranked_validation)

    # Apply reinforcement learning using the reward model
    print("Starting reinforcement learning...")
    processor.reinforcement_learning(dataset_unlabeled_train, dataset_validation)

    print("Training completed!")

if __name__ == "__main__":
    main()
