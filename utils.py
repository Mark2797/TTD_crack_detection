import torch
from matrices import iou_crack, f1_score_crack
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

def val_loop(model, device, dataloader, loss_fn):
    """
    Executes a validation pass and returns metrics.
    """
    model.eval()
    v_loss, v_iou, v_f1 = 0.0, 0.0, 0.0
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

    return v_loss / total_samples, v_iou / total_samples, v_f1 / total_samples

def epochs(model, model_name, device, train_dl, val_dl, loss_fn, optimizer, num_epoch):
    """
    Main training execution loop.
    """
    model = model.to(device)
    best_iou = -float('inf')

    history = {
        'train_loss': [],
        'val_loss': [],
        'val_iou': [],
        'val_f1': []
    }
    
    for epoch in range(num_epoch):
        # Perform training and validation steps
        t_loss = train_loop(model, device, train_dl, loss_fn, optimizer)
        v_loss, v_iou, v_f1 = val_loop(model, device, val_dl, loss_fn)

        history['train_loss'].append(t_loss)
        history['val_loss'].append(v_loss)
        history['val_iou'].append(v_iou)
        history['val_f1'].append(v_f1)
        
        if v_iou > best_iou:
            best_iou = v_iou
            torch.save(model.state_dict(), f"{model_name}.pth")
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
