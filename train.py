# Original Author: Sebastian Raschka (https://github.com/rasbt/LLMs-from-scratch)
# Modified By: woodsj1206 (https://github.com/woodsj1206)
# Last Modified: 12/18/2025
import os
import torch
from datetime import datetime


def calculate_accuracy_loader(data_loader, model, device, num_batches=None):
    model.eval()
    correct_predictions, total_predictions = 0, 0

    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            input_batch, target_batch = input_batch.to(
                device), target_batch.to(device)

            with torch.no_grad():
                logits = model(input_batch)[:, -1, :]
                predicted_labels = torch.argmax(logits, dim=-1)

                total_predictions += predicted_labels.shape[0]
                correct_predictions += (predicted_labels ==
                                        target_batch).sum().item()
        else:
            break

    return correct_predictions / total_predictions


def calculate_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)[:, -1, :]
    loss = torch.nn.functional.cross_entropy(logits, target_batch)
    return loss


def calculate_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calculate_loss_batch(
                input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches


def train_model(model, train_loader, val_loader, optimizer, device, reverse_mapping, num_epochs, eval_freq, eval_iter, dir_path, checkpoint_frequency=10000):
    train_losses = []
    val_losses = []

    train_accuracies = []
    val_accuracies = []

    examples_seen = 0
    global_step = -1

    for epoch in range(num_epochs):
        model.train()

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calculate_loss_batch(
                input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            examples_seen += input_batch.shape[0]
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)

                train_losses.append(train_loss)
                val_losses.append(val_loss)
                print(f"{datetime.now().time()}: Epoch {epoch + 1} (Step {global_step:06d}): Train Loss - {train_loss:.3f}, Val Loss - {val_loss:.3f}")

            if global_step % checkpoint_frequency == 0 and global_step != 0:
                checkpoint_path = os.path.join(
                    dir_path, f"model-{global_step}.pth")
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "reverse_mapping_dict": reverse_mapping
                }, checkpoint_path)
                print(f"Saved: {checkpoint_path}")

        train_accuracy = calculate_accuracy_loader(
            train_loader, model, device, num_batches=eval_iter)
        val_accuracy = calculate_accuracy_loader(
            val_loader, model, device, num_batches=eval_iter)

        print()
        print(f"Training Accuracy: {train_accuracy * 100:2f}%")
        print(f"Validation Accuracy: {val_accuracy * 100:2f}%")
        print()

        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)

    final_checkpoint_path = os.path.join(
        dir_path, f"model-final-{global_step}.pth")
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "reverse_mapping_dict": reverse_mapping
    }, final_checkpoint_path)

    return train_losses, val_losses, train_accuracies, val_accuracies, examples_seen


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calculate_loss_loader(
            train_loader, model, device, num_batches=eval_iter)
        val_loss = calculate_loss_loader(
            val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def classify_text(text, model, tokenizer, device, max_length=None, pad_token_id=50256):
    inputs_ids = tokenizer.encode(text)
    supported_context_length = model.pos_emb.weight.shape[0]

    inputs_ids = inputs_ids[:min(max_length, supported_context_length)]
    inputs_ids += [pad_token_id] * (max_length - len(inputs_ids))

    inputs_ids = torch.tensor(inputs_ids, device=device).unsqueeze(0)

    with torch.no_grad():
        logits = model(inputs_ids)[:, -1, :]
    predicted_label = torch.argmax(logits, dim=-1).item()

    return int(predicted_label)
