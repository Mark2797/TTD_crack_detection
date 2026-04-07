import os
import sys
from tqdm.auto import tqdm
import torch
import numpy as np
from matrices import *
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from fastai.vision.all import imagenet_stats

class EarlyStopping:
    def __init__(self, patience, delta=0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

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

        # PER-BATCH STEP: Only if it's OneCycle
        # reference: https://medium.com/@heyamit10/pytorch-segmentation-models-a-practical-guide-5bf973a32e30
        if scheduler and isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
            scheduler.step()
        
        batch_size = X_batch.size(0)
        running_loss += loss.item() * batch_size
        total_samples += batch_size
        
    # PER-EPOCH STEP: For everything else (Cosine, Step, etc.)
    # We do this OUTSIDE the for-loop, after all batches are done
    if scheduler and not isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
        # Safety: Plateau needs the validation loss, so it actually 
        # usually stays in the 'epochs' function. But for Cosine, it's fine here.
        if not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
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

def epochs(model, model_name, device, train_dl, val_dl, loss_fn, optimizer, num_epoch, scheduler=None, patience=15, save_dir="models"):
    """
    Main training execution loop.
    """
    os.makedirs(save_dir, exist_ok=True)

    early_stopper = EarlyStopping(patience=patience)

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
        t_loss = train_loop(model, device, train_dl, loss_fn, optimizer, scheduler)
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

        early_stopper(v_loss)
        if early_stopper.early_stop:
            print(f"\nEarly stopping triggered at epoch {epoch}. Stopping training.")
            break
              
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

def show_prediction_overlap(model, dataloader, device, custom_stats=None):
    """
    Displays 3x4 grid. Fixes RuntimeError by ensuring stats are on CPU 
    for denormalization.
    """
    model.to(device)
    model.eval()
    
    stats = custom_stats if custom_stats else imagenet_stats
    # Explicitly set device="cpu" to match img.cpu()
    mean = torch.tensor(stats[0], device="cpu").view(3, 1, 1)
    std = torch.tensor(stats[1], device="cpu").view(3, 1, 1)

    found_samples = []
    for images, masks in dataloader:
        for i in range(len(masks)):
            if masks[i].sum() > 0:
                found_samples.append((images[i], masks[i]))
            if len(found_samples) == 3: break
        if len(found_samples) == 3: break

    if len(found_samples) < 3:
        print(f"Only found {len(found_samples)} samples with defects.")
        return

    fig, axes = plt.subplots(3, 4, figsize=(22, 18))
    
    for idx, (img, mask) in enumerate(found_samples):
        # Now both img.cpu() and stats are on CPU
        img_vis = (img.cpu() * std + mean).permute(1, 2, 0).numpy()
        img_vis = np.clip(img_vis, 0, 1)

        img_input = img.unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(img_input)
            pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
        
        targ = mask.squeeze().cpu().numpy()
        overlap = np.zeros((targ.shape[0], targ.shape[1], 3))
        overlap[(pred == 1) & (targ == 1)] = [0, 1, 0] # TP
        overlap[(pred == 0) & (targ == 1)] = [1, 0, 0] # FN
        overlap[(pred == 1) & (targ == 0)] = [1, 1, 0] # FP

        axes[idx, 0].imshow(img_vis)
        axes[idx, 1].imshow(targ * 255, cmap='gray')
        axes[idx, 2].imshow(pred * 255, cmap='gray')
        axes[idx, 3].imshow(overlap)
        for ax in axes[idx]: ax.axis('off')

    patches = [
        mpatches.Patch(color='green', label='Correct (TP)'),
        mpatches.Patch(color='red', label='Missed (FN)'),
        mpatches.Patch(color='yellow', label='False Alarm (FP)')
    ]
    fig.legend(handles=patches, loc='upper left', bbox_to_anchor=(1.01, 0.95), fontsize=12)
    plt.tight_layout()
    plt.subplots_adjust(right=0.9)
    plt.show()

def save_prediction_overlap(model, model_name, dataloader, device, custom_stats=None, save_dir="figures"):
    """
    Saves 3x4 grid. Fixes device mismatch error by placing stats on CPU.
    """
    os.makedirs(save_dir, exist_ok=True)
    model.to(device)
    model.eval()

    stats = custom_stats if custom_stats else imagenet_stats
    # Force stats to CPU to avoid conflict with default CUDA device
    mean = torch.tensor(stats[0], device="cpu").view(3, 1, 1)
    std = torch.tensor(stats[1], device="cpu").view(3, 1, 1)

    found_samples = []
    for images, masks in dataloader:
        for i in range(len(masks)):
            if masks[i].sum() > 0:
                found_samples.append((images[i], masks[i]))
            if len(found_samples) == 3: break
        if len(found_samples) == 3: break

    if not found_samples:
        print("No defect samples found to save.")
        return

    fig, axes = plt.subplots(3, 4, figsize=(22, 18))
    
    for idx, (img, mask) in enumerate(found_samples):
        # Calculation now happens entirely on CPU
        img_vis = (img.cpu() * std + mean).permute(1, 2, 0).numpy()
        img_vis = np.clip(img_vis, 0, 1)

        img_input = img.unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(img_input)
            pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
        
        targ = mask.squeeze().cpu().numpy()
        overlap = np.zeros((targ.shape[0], targ.shape[1], 3))
        overlap[(pred == 1) & (targ == 1)] = [0, 1, 0] 
        overlap[(pred == 0) & (targ == 1)] = [1, 0, 0] 
        overlap[(pred == 1) & (targ == 0)] = [1, 1, 0]

        axes[idx, 0].imshow(img_vis)
        axes[idx, 1].imshow(targ * 255, cmap='gray')
        axes[idx, 2].imshow(pred * 255, cmap='gray')
        axes[idx, 3].imshow(overlap)
        for ax in axes[idx]: ax.axis('off')

    patches = [
        mpatches.Patch(color='green', label='Correct (TP)'),
        mpatches.Patch(color='red', label='Missed (FN)'),
        mpatches.Patch(color='yellow', label='False Alarm (FP)')
    ]
    fig.legend(handles=patches, loc='upper left', bbox_to_anchor=(1.01, 0.95))
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, f"{model_name}_visualization.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Visualization saved to {save_path}")
