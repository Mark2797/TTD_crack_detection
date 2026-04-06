import os
import sys
from tqdm.auto import tqdm
import torch
import numpy as np
from matrices import *
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

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

def show_prediction_overlap(model, dataloader, device):
    """
    Displays a 3-panel comparison: Ground Truth, Model Prediction, and a 
    color-coded Overlap image for error analysis.
    """
    # Set model to evaluation mode and move to the specified device
    model.to(device)
    model.eval()

    # Retrieve the first batch of images and sanitized masks from the pipeline
    images, masks = next(iter(dataloader))
    images = images.to(device)
    
    with torch.no_grad():
        # Perform inference and get class indices (0 for background, 1 for crack)
        output = model(images)
        preds = torch.argmax(output, dim=1) 

    # Extract the first sample and convert to NumPy for plotting
    targ = masks[0].cpu().squeeze().numpy()
    pred = preds[0].cpu().squeeze().numpy()

    # Create an empty RGB image for the overlap visualization
    overlap = np.zeros((targ.shape[0], targ.shape[1], 3))
    
    # Logic for error categories based on binary mask values (0 and 1):
    # Green: Correct Prediction (True Positive)
    overlap[(pred == 1) & (targ == 1)] = [0, 1, 0] 
    # Red: Missed Crack (False Negative)
    overlap[(pred == 0) & (targ == 1)] = [1, 0, 0] 
    # Yellow: False Alarm (False Positive)
    overlap[(pred == 1) & (targ == 0)] = [1, 1, 0]

    # Initialize the plot with three subplots
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    
    # Scale binary masks (0-1) to grayscale range (0-255) for visibility
    ax[0].imshow(targ * 255, cmap='gray')
    ax[0].set_title("Ground Truth Label")
    
    ax[1].imshow(pred * 255, cmap='gray')
    ax[1].set_title("Model Prediction")
    
    # Show the RGB overlap image
    ax[2].imshow(overlap)
    ax[2].set_title("Error Analysis (Overlap)")

    # Clean up the visual by removing axes from all subplots
    for a in ax: a.axis('off')

    # Create custom legend patches for the error analysis
    green_patch = mpatches.Patch(color='green', label='Correct (TP)')
    red_patch = mpatches.Patch(color='red', label='Missed (FN)')
    yellow_patch = mpatches.Patch(color='yellow', label='False Alarm (FP)')
    
    # Anchor the legend to the right of the overlap plot to avoid overlap
    ax[2].legend(handles=[green_patch, red_patch, yellow_patch], 
                 loc='upper left', bbox_to_anchor=(1.05, 1))
    
    plt.tight_layout()
    plt.show() # Display the figure in the notebook

def save_prediction_overlap(model, model_name, dataloader, device, save_dir="figures"):
    """
    Generates and saves the 3-panel visualization to disk without 
    displaying it in the notebook.
    """
    # Ensure the target directory exists
    os.makedirs(save_dir, exist_ok=True) 
    
    model.to(device)
    model.eval()

    # Extract the first batch and run prediction
    images, masks = next(iter(dataloader))
    images = images.to(device)
    with torch.no_grad():
        output = model(images)
        preds = torch.argmax(output, dim=1) 

    # Prepare data for error analysis
    targ = masks[0].cpu().squeeze().numpy()
    pred = preds[0].cpu().squeeze().numpy()

    # Construct the RGB overlap visualization
    overlap = np.zeros((targ.shape[0], targ.shape[1], 3))
    overlap[(pred == 1) & (targ == 1)] = [0, 1, 0] # Correct (TP)
    overlap[(pred == 0) & (targ == 1)] = [1, 0, 0] # Missed (FN)
    overlap[(pred == 1) & (targ == 0)] = [1, 1, 0] # False Alarm (FP)

    # Create the figure for saving
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    
    # Use 255 scaling for the binary grayscale masks
    ax[0].imshow(targ * 255, cmap='gray')
    ax[0].set_title("Ground Truth Label")
    ax[1].imshow(pred * 255, cmap='gray')
    ax[1].set_title("Model Prediction")
    ax[2].imshow(overlap)
    ax[2].set_title("Error Analysis (Overlap)")

    for a in ax: a.axis('off')

    # Add the legend with a tight anchor to the plot
    green_patch = mpatches.Patch(color='green', label='Correct (TP)')
    red_patch = mpatches.Patch(color='red', label='Missed (FN)')
    yellow_patch = mpatches.Patch(color='yellow', label='False Alarm (FP)')
    ax[2].legend(handles=[green_patch, red_patch, yellow_patch], 
                 loc='upper left', bbox_to_anchor=(1.05, 1))
    
    plt.tight_layout()
    
    # Define the save path using the experiment/model name
    visualization_path = os.path.join(save_dir, f"{model_name}_visualization.png")
    
    # bbox_inches='tight' is critical here to ensure the legend isn't cropped
    plt.savefig(visualization_path, bbox_inches='tight')
    
    # Close the figure to free memory and prevent it from showing in the notebook
    plt.close(fig) 
    
    print(f"Visualization saved to {visualization_path}")
