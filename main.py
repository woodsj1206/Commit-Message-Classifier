# Original Author: Sebastian Raschka (https://github.com/rasbt/LLMs-from-scratch)
# Modified By: woodsj1206 (https://github.com/woodsj1206)
# Last Modified: 12/18/2025
from dataset import CSVDataset
from gpt import GPT_CONFIG, MODEL_CONFIGS, GPTModel
from results import plot_data
import torch
import pandas
import tiktoken
import time
from train import calculate_accuracy_loader, classify_text, train_model
from utils import create_directory, download_gpt2_model, split_data
from torch.utils.data import DataLoader
import os
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Commit Message Classifier using GPT-2")
    parser.add_argument(
        "--csv_file_path",
        type=str,
        required=True,
        help="Required. Path to the CSV file containing text and labels. Expected format: text,label. Default is 'csv_data.csv'.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility. Default is None.",
    )

    # Data split fractions
    parser.add_argument(
        "--train_fraction",
        type=float,
        default=0.7,
        help="Fraction of data to use for training (between 0 and 1). Default is 0.7 (70 percent).",
    )
    parser.add_argument(
        "--validation_fraction",
        type=float,
        default=0.2,
        help="Fraction of data to use for validation (between 0 and 1). Default is 0.2 (20 percent).",
    )
    # (Note: The remaining fraction will be used for testing. Default is 0.1 (10 percent).)

    # DataLoader parameters
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of worker threads for data loading. Default is 0.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for DataLoader. Default is 8.",
    )

    # GPT-2 model selection
    parser.add_argument(
        "--gpt_model",
        type=str,
        default="gpt2-small-124M.pth",
        help="GPT-2 model to use. Options include 'gpt2-small-124M.pth', 'gpt2-medium-355M.pth', 'gpt2-large-774M.pth', 'gpt2-xl-1558M.pth'. Default is 'gpt2-small-124M.pth'.",
    )
    parser.add_argument(
        "--last_trf_blocks_to_finetune",
        type=int,
        default=2,
        help="Number of last transformer blocks to fine-tune. Default is 2.",
    )

    # Training parameters
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=10,
        help="Number of training epochs. Default is 10.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-5,
        help="Learning rate for the optimizer. Default is 5e-5.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.1,
        help="Weight decay for the optimizer. Default is 0.1.",
    )
    parser.add_argument(
        "--eval_freq",
        type=int,
        default=50,
        help="Frequency (in steps) for evaluation during training. Default is 50.",
    )
    parser.add_argument(
        "--eval_iter",
        type=int,
        default=5,
        help="Number of batches to use for evaluation. Default is 5.",
    )
    parser.add_argument(
        "--checkpoint_frequency",
        type=int,
        default=10000,
        help="Frequency (in steps) for saving model checkpoints. Default is 10000.",
    )

    args = parser.parse_args()
    return args


def main(args):
    ################################################
    # Set seed for reproducibility
    ################################################

    seed = args.seed
    if seed is not None:
        print(f"Setting random seed to {seed} for reproducibility...")
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    ################################################
    # Load and preprocess data
    ################################################

    csv_file_path = args.csv_file_path
    if not os.path.isfile(csv_file_path):
        print(f"Error: CSV file '{csv_file_path}' not found.")
        return

    print("Loading and preprocessing data...")
    file_name_extension = os.path.basename(csv_file_path)
    file_name = os.path.splitext(file_name_extension)[0]
    current_time = int(time.time())

    output_dir = create_directory(os.path.join(
        f"{file_name}_output-{current_time}"))
    if output_dir is None:
        return

    text = "Text"
    label = "Label"

    csv_data = pandas.read_csv(csv_file_path, header=0, names=[text, label])

    labels = csv_data[label].unique().tolist()

    mapping = {labels: i for i, labels in enumerate(labels)}
    reverse_mapping = {value: key for key, value in mapping.items()}

    print(f"Number of classes: {len(mapping)}")
    print(f"Label mapping: {mapping}")
    print(f"Reverse mapping: {reverse_mapping}")

    mapped_csv_data = csv_data
    mapped_csv_data[label] = mapped_csv_data[label].map(mapping)

    # 70% for training, 20% for validation, 10% for testing
    train_fraction = args.train_fraction
    validation_fraction = args.validation_fraction

    print("\nSplitting data into training, validation, and test sets...")
    train_data, validation_data, test_data = split_data(
        mapped_csv_data, train_fraction, validation_fraction, seed)

    data_dir = create_directory(os.path.join(output_dir, "data"))
    if data_dir is None:
        return

    train_csv = os.path.join(data_dir, "train.csv")
    val_csv = os.path.join(data_dir, "validation.csv")
    test_csv = os.path.join(data_dir, "test.csv")

    train_data.to_csv(train_csv, index=None)
    validation_data.to_csv(val_csv, index=None)
    test_data.to_csv(test_csv, index=None)

    tokenizer = tiktoken.get_encoding("gpt2")
    train_dataset = CSVDataset(csv_file_path=train_csv, csv_text=text,
                               csv_label=label, max_length=None, tokenizer=tokenizer)

    training_max_length = train_dataset.max_length
    print(f"Determined training max length: {training_max_length}")

    validation_dataset = CSVDataset(csv_file_path=val_csv, csv_text=text,
                                    csv_label=label, max_length=training_max_length, tokenizer=tokenizer)
    test_dataset = CSVDataset(csv_file_path=test_csv, csv_text=text,
                              csv_label=label, max_length=training_max_length, tokenizer=tokenizer)

    ################################################
    # DataLoader parameters
    ################################################

    num_workers = args.num_workers
    batch_size = args.batch_size

    print("\nCreating DataLoaders...")
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True
    )

    validation_loader = DataLoader(
        dataset=validation_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False
    )

    ################################################
    # Download pre-trained GPT-2 model and modify for classification
    ################################################

    # Different models can be found here: https://huggingface.co/rasbt/gpt2-from-scratch-pytorch/tree/main
    # gpt2-small-124M.pth, gpt2-medium-355M.pth, gpt2-large-774M.pth, gpt2-xl-1558M.pth
    gpt_model = args.gpt_model

    print("\nPreparing GPT-2 model...")
    gpt_dir = os.path.join("gpt_models")
    try:
        os.makedirs(gpt_dir, exist_ok=True)
    except OSError as e:
        print(
            f"Error: An error occurred when creating GPT model directory. {e}")
        return

    gpt_model_path = os.path.join(gpt_dir, gpt_model)

    if not os.path.isfile(gpt_model_path):
        print("\nDownloading GPT-2 model...")
        download_gpt2_model(gpt_model, gpt_dir)
    else:
        print(f"\nUsing existing GPT-2 model at: {gpt_model_path}")

    GPT_CONFIG.update(MODEL_CONFIGS[gpt_model])
    model = GPTModel(GPT_CONFIG)
    model.load_state_dict(torch.load(gpt_model_path, weights_only=True))
    model.eval()

    print("\nModifying model for classification task...")
    for param in model.parameters():
        param.requires_grad = False

    num_classes = len(mapping)
    model.out_head = torch.nn.Linear(GPT_CONFIG["emb_dim"], num_classes)

    # Fine-tune last transformer blocks and final norm layer
    last_trf_blocks_to_finetune = args.last_trf_blocks_to_finetune
    for trf_block in model.trf_blocks[-last_trf_blocks_to_finetune:]:
        for param in trf_block.parameters():
            param.requires_grad = True

    for param in model.final_norm.parameters():
        param.requires_grad = True

    ################################################
    # Train model
    ################################################

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    lr = args.lr
    weight_decay = args.weight_decay
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay)
    num_epochs = args.num_epochs
    eval_freq = args.eval_freq
    eval_iter = args.eval_iter
    checkpoint_frequency = args.checkpoint_frequency

    print("\nStarting training...")
    start_time = time.time()
    print(f"Start Time: {time.ctime(start_time)}")

    model_dir = create_directory(os.path.join(output_dir, "models"))
    if model_dir is None:
        return

    train_losses, val_losses, train_accs, val_accs, examples_seen = train_model(
        model, train_loader, validation_loader, optimizer, device, reverse_mapping, num_epochs, eval_freq, eval_iter, model_dir, checkpoint_frequency)

    end_time = time.time()
    print(f"End Time: {time.ctime(end_time)}")

    # Display training time statistics
    print("\nTraining completed.")
    print(f"Start Time: {time.ctime(start_time)}")
    print(f"End Time: {time.ctime(end_time)}")
    print(f"Total Training Time: {end_time - start_time:.2f} seconds")

    ################################################
    # Plot results and evaluate final model
    ################################################

    print("\nPlotting training results...")
    results_dir = create_directory(os.path.join(output_dir, "results"))
    if results_dir is None:
        return

    epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
    examples_seen_tensor = torch.linspace(0, examples_seen, len(train_losses))
    chart_label = "loss"
    plot_data(epochs_tensor, examples_seen_tensor,
              train_losses, val_losses, results_dir, chart_label)

    epochs_tensor = torch.linspace(0, num_epochs, len(train_accs))
    examples_seen_tensor = torch.linspace(0, examples_seen, len(train_accs))
    chart_label = "accuracy"
    plot_data(epochs_tensor, examples_seen_tensor,
              train_accs, val_accs, results_dir, chart_label)

    # Final evaluation on all datasets
    print("\nEvaluating the final model on all datasets...")
    train_accuracy = calculate_accuracy_loader(train_loader, model, device)
    validation_accuracy = calculate_accuracy_loader(
        validation_loader, model, device)
    test_accuracy = calculate_accuracy_loader(test_loader, model, device)

    print(f"Training accuracy: {train_accuracy*100:.2f}%")
    print(f"Validation accuracy: {validation_accuracy*100:.2f}%")
    print(f"Test accuracy: {test_accuracy*100:.2f}%")

    ################################################
    # Example usage
    ################################################

    print("\nClassifying example commit messages...")
    # Example classifications
    text_examples = {
        "build": "add webpack configuration for production bundle",
        "ci/cd": "add GitHub Actions workflow for automated testing",
        "docs": "update contribution guidelines with new code of conduct",
        "feat": "add user registration with email verification",
        "fix": "correct total calculation for discounted items",
        "perf": "implement lazy loading for gallery view",
        "refactor": "simplify state management in user reducer",
        "style": "fix indentation and trailing spaces",
        "test": "cover edge cases in login flow",
        "chore": "update dependencies to latest minor versions",
        "revert": "undo recent changes to authentication module",
    }

    correct_answers = 0
    for key, value in text_examples.items():
        prediction = classify_text(
            value, model, tokenizer, device, max_length=training_max_length)
        isCorrect = reverse_mapping[prediction] == key
        correct_answers += int(isCorrect)
        print(
            f"{isCorrect} | {key} | {reverse_mapping[prediction]} ({prediction}): {value}")

    print(f"Correct Answers: {correct_answers} / {len(text_examples)}")
    print(
        f"Accuracy (Text Examples): {correct_answers / len(text_examples)*100:.2f}%")


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
