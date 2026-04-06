import os
import sys
from tqdm.auto import tqdm
import torch
from matrices import *
import matplotlib.pyplot as plt

def train_loop(model, device, dataloader, loss_fn, optimizer, scheduler=None):
    """
    Executes a single training epoch.
    """
    model.train()
    running_loss = 0.0
    total_samples = 0
    
    for X_batch, y_batch in dataloader:
        X_batch = torch.as_tensor(X_batch).to(device)
        y_batch = torch.as_tensor(y_batch).to(device).long()
        
        optimizer.zero_grad()
        output = model(X_batch) 
        loss = loss_fn(output.as_subclass(torch.Tensor), y_batch)
        loss.backward()
        optimizer.step()
        
        batch_size = X_batch.size(0)
        running_loss += loss.item() * batch_size
        total_samples += batch_size
        
    if scheduler:
        scheduler.step()
        
    return running_loss / total_samples

def val_loop(model, device, dataloader, loss_fn, is_test=False):
    """
    Executes a validation pass and returns metrics.
    """
    model.eval()
    v_loss, v_iou, v_f1 = 0.0, 0.0, 0.0
    v_recall, v_prec = 0.0, 0.0
    total_samples = 0
    
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = torch.as_tensor(X_batch).to(device)
            y_batch = torch.as_tensor(y_batch).to(device).long()
            output = model(X_batch)
            
            batch_size = X_batch.size(0)
            total_samples += batch_size
            
            v_loss += loss_fn(output.as_subclass(torch.Tensor), y_batch).item() * batch_size
            v_iou += iou_crack(output, y_batch).item() * batch_size
            v_f1 += f1_score_crack(output, y_batch).item() * batch_size

            if is_test:
                v_recall += recall_crack(output, y_batch).item() * batch_size
                v_prec += precision_crack(output, y_batch).item() * batch_size
    
    avg_loss = v_loss / total_samples
    avg_iou = v_iou / total_samples
    avg_f1 = v_f1 / total_samples
    
    if is_test:
        avg_recall = v_recall / total_samples
        avg_prec = v_prec / total_samples
        return avg_loss, avg_iou, avg_f1, avg_recall, avg_prec
        
    return avg_loss, avg_iou, avg_f1

def epochs(model, model_name, device, train_dl, val_dl, loss_fn, optimizer, num_epoch, save_dir="models"):
    """
    Main training execution loop.
    """
    os.makedirs(save_dir, exist_ok=True)

    model = model.to(device)
    best_iou = -float('inf')

    history = {
        'train_loss': [],
        'val_loss': [],
        'val_iou': [],
        'val_f1': []
    }

    pbar = tqdm(range(num_epoch), desc=f"  → {model_name}", leave=False, file=sys.__stdout__)
    
    for epoch in pbar:
        # Perform training and validation steps
        t_loss = train_loop(model, device, train_dl, loss_fn, optimizer)
        v_loss, v_iou, v_f1 = val_loop(model, device, val_dl, loss_fn)

        history['train_loss'].append(t_loss)
        history['val_loss'].append(v_loss)
        history['val_iou'].append(v_iou)
        history['val_f1'].append(v_f1)

        pbar.set_postfix({"IoU": f"{v_iou:.4f}", "F1": f"{v_f1:.4f}"})
        
        if v_iou > best_iou:
            best_iou = v_iou
            save_path = os.path.join(save_dir, f"{model_name}.pth")
            torch.save(model.state_dict(), save_path)
            checkpoint_status = " [Saved Best Model]"
        else:
            checkpoint_status = ""

        print(f"Epoch {epoch}: "
              f"T-Loss: {t_loss:.4f} | V-Loss: {v_loss:.4f} | "
              f"IoU: {v_iou:.4f} | F1: {v_f1:.4f}{checkpoint_status}")
              
    return history

def plot_training_history(history):
    """
    Plots the training and validation progress.
    Graph 1: Training Loss vs. Validation Loss.
    Graph 2: Validation IoU vs. Validation F1 Score.
    """
    epochs = range(len(history['train_loss']))

    # --- Plot 1: Loss History ---
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history['train_loss'], label="Train Loss", color='blue')
    plt.plot(epochs, history['val_loss'], label="Val Loss", color='orange')
    plt.title("Train & Val Loss History")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

    # --- Plot 2: Performance Metrics (IoU & F1) ---
    plt.figure(figsize=(10, 5))
    # Plotting Val F1 and Val IoU as requested
    plt.plot(epochs, history['val_f1'], label="Val F1 Score", color='green')
    plt.plot(epochs, history['val_iou'], label="Val IoU", color='red')
    plt.title("Validation Performance: F1 Score & IoU")
    plt.xlabel("Epoch")
    plt.ylabel("Score (0.0 - 1.0)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

def save_training_history(history, model_name, save_dir="figures"):
    """
    Saves the training and validation progress as PNG files.
    Graph 1: Training Loss vs. Validation Loss.
    Graph 2: Validation IoU vs. Validation F1 Score.
    """
    # Ensure the directory for figures exists 
    os.makedirs(save_dir, exist_ok=True)
    
    epochs = range(len(history['train_loss']))

    # --- Plot 1: Loss History ---
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history['train_loss'], label="Train Loss", color='blue')
    plt.plot(epochs, history['val_loss'], label="Val Loss", color='orange')
    plt.title(f"Loss History: {model_name}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Save the Loss plot [cite: 298]
    loss_path = os.path.join(save_dir, f"{model_name}_loss.png")
    plt.savefig(loss_path)
    plt.close() # Close to free up memory during loops

    # --- Plot 2: Performance Metrics (IoU & F1) ---
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history['val_f1'], label="Val F1 Score", color='green')
    plt.plot(epochs, history['val_iou'], label="Val IoU", color='red')
    plt.title(f"Performance Metrics: {model_name}")
    plt.xlabel("Epoch")
    plt.ylabel("Score (0.0 - 1.0)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Save the Metrics plot [cite: 298]
    metrics_path = os.path.join(save_dir, f"{model_name}_metrics.png")
    plt.savefig(metrics_path)
    plt.close()
    
    print(f"Charts saved to {save_dir}/")

def visualize_first_prediction(model, dataloader, device):
    """
    Loads a trained model and displays the first image's ground truth vs prediction.
    Enhanced to ensure binary masks (0 and 1) are visible.
    """
    model.to(device)
    model.eval()

    # Get first batch
    images, masks = next(iter(dataloader))
    
    images = images.to(device)
    with torch.no_grad():
        output = model(images)
        # argmax results in values 0 and 1
        preds = torch.argmax(output, dim=1)

    # Prepare data for plotting
    # Squeeze out extra dimensions
    true_mask = masks[0].cpu().squeeze().numpy() * 255
    pred_mask = preds[0].cpu().squeeze().numpy() * 255

    # Visualization with improved visibility
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    
    # Using a vibrant colormap like 'magma' or 'jet' makes 1s stand out against 0s
    ax[0].imshow(true_mask, cmap='magma') 
    ax[0].set_title("Ground Truth Label (0-1 Range)")
    ax[0].axis('off')
    
    # Alternatively, you can multiply by 255 if you prefer standard grayscale
    ax[1].imshow(pred_mask, cmap='gray')
    ax[1].set_title("Model Prediction (Scaled to 255)")
    ax[1].axis('off')
    
    plt.tight_layout()
    plt.show()
