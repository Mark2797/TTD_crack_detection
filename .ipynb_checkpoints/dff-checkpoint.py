import os
import gc
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.decomposition import NMF
from fastai.vision.all import imagenet_stats

class DeepFeatureFactorization:
    def __init__(self, model, target_layer):
        """
        Initializes the DFF hook on a specific model layer to capture feature maps.
        """
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.hook = self.target_layer.register_forward_hook(self._hook_fn)

    def _hook_fn(self, module, input, output):
        # Detach and move to CPU immediately to save GPU memory
        self.activations = output.detach().cpu()

    def remove_hook(self):
        """Removes the hook to prevent memory leaks."""
        self.hook.remove()

    def factorize(self, input_tensor, n_components=5):
        """
        Runs the image through the model and factorizes the captured activations using NMF.
        """
        self.model.eval()
        with torch.no_grad():
            _ = self.model(input_tensor)
            
        # Squeeze batch dim -> (Channels, H_act, W_act)
        A = self.activations.squeeze(0)
        channels, h_act, w_act = A.shape
        
        # Flatten spatial dimensions for NMF: (Channels, H_act * W_act)
        A_flat = A.view(channels, -1).numpy()
        A_flat = np.maximum(A_flat, 0) # NMF requires strictly non-negative inputs
        
        X = A_flat.T 
        
        # Use 'random' init and looser tolerance to prevent freezing when features 
        # are too noisy (e.g., when testing no_weight models)
        nmf = NMF(n_components=n_components, init='random', random_state=42, max_iter=800, tol=1e-3)
        W = nmf.fit_transform(X) 
        
        # Reshape the components back to spatial heatmaps
        heatmaps = W.T.reshape(n_components, h_act, w_act)
        
        # Use torch.from_numpy to prevent GPU device conflicts during upsampling
        _, _, H, W_img = input_tensor.shape
        heatmaps_tensor = torch.from_numpy(heatmaps).unsqueeze(0)
        
        heatmaps_upsampled = F.interpolate(heatmaps_tensor, size=(H, W_img), mode='bilinear', align_corners=False)
        
        # Ensure conversion to a CPU numpy array before returning
        return heatmaps_upsampled.squeeze(0).cpu().numpy()


def plot_nmf_elbow(model, dataloader, target_layer, device, k_range=range(2, 16)):
    """
    Plots the NMF Reconstruction Error for a range of k values to help 
    find the optimal number of components using the Elbow Method.
    """
    model.to(device)
    model.eval()
    
    # 1. Setup a temporary hook to grab activations
    activations = []
    def hook_fn(module, input, output):
        activations.append(output.detach().cpu())
    
    hook = target_layer.register_forward_hook(hook_fn)
    
    # 2. Find a single representative image with a defect (TP or FN candidate)
    sample_img = None
    with torch.no_grad():
        for images, masks in dataloader:
            for i in range(len(masks)):
                if masks[i].sum() > 0: # Ensure there is a crack to factorize
                    sample_img = images[i].unsqueeze(0).to(device)
                    break
            if sample_img is not None:
                break
                
    if sample_img is None:
        print("No defect images found in the dataloader.")
        hook.remove()
        return

    # 3. Forward pass to capture activations
    _ = model(sample_img)
    hook.remove() # Clean up hook immediately
    
    # 4. Prepare data for NMF
    A = activations[0].squeeze(0)
    channels, h_act, w_act = A.shape
    A_flat = A.view(channels, -1).numpy()
    A_flat = np.maximum(A_flat, 0)
    X = A_flat.T 
    
    # 5. Calculate Reconstruction Error for each k
    errors = []
    print(f"Calculating NMF errors for k in {list(k_range)}...")
    for k in k_range:
        # Use the same robust NMF settings to prevent crashes on no_weight models
        nmf = NMF(n_components=k, init='random', random_state=42, max_iter=800, tol=1e-3)
        nmf.fit(X)
        errors.append(nmf.reconstruction_err_)
        
    # 6. Plot the Elbow Curve
    plt.figure(figsize=(8, 5))
    plt.plot(list(k_range), errors, marker='o', linestyle='-', color='b')
    plt.title("NMF Elbow Method for Deep Feature Factorization")
    plt.xlabel("Number of Components (k)")
    plt.ylabel("Reconstruction Error")
    plt.xticks(list(k_range))
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Highlight the typical "sweet spot" area conceptually
    plt.axvspan(4, 8, color='green', alpha=0.1, label='Typical Sweet Spot')
    plt.legend()
    
    plt.show()


def find_situations_and_run_dff(model, dataloader, target_layer, device, save_dir="figures/dff_analysis", custom_stats=None, n_components=6):
    """
    Scans the dataloader for TP, FP, FN, TN situations, runs DFF, 
    and plots them dynamically based on k (Row 1: 4 standard, Row 2: k//2 concepts, Row 3: remainder).
    """
    os.makedirs(save_dir, exist_ok=True)
    model.to(device)
    model.eval()
    
    stats = custom_stats if custom_stats else imagenet_stats
    
    # Fix UserWarning: Avoid double-wrapping tensors by using as_tensor and clone
    mean = torch.as_tensor(stats[0], device="cpu").clone().detach().view(3, 1, 1)
    std = torch.as_tensor(stats[1], device="cpu").clone().detach().view(3, 1, 1)

    found_situations = {"TP": None, "FP": None, "FN": None, "TN": None}
    
    print(f"Scanning dataloader for edge cases (TP, FP, FN, TN) for k={n_components}...")
    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            
            for i in range(len(masks)):
                img = images[i].unsqueeze(0)
                mask = masks[i].squeeze().cpu()
                pred = preds[i].squeeze().cpu()
                
                mask_has_crack = mask.sum() > 0
                pred_has_crack = pred.sum() > 0
                
                # Capture the first instance of each critical edge case
                if mask_has_crack and pred_has_crack and found_situations["TP"] is None:
                    found_situations["TP"] = (img, mask, pred)
                elif not mask_has_crack and pred_has_crack and found_situations["FP"] is None:
                    found_situations["FP"] = (img, mask, pred)
                elif mask_has_crack and not pred_has_crack and found_situations["FN"] is None:
                    found_situations["FN"] = (img, mask, pred)
                elif not mask_has_crack and not pred_has_crack and found_situations["TN"] is None:
                    found_situations["TN"] = (img, mask, pred)
            
            # Break early if all 4 situations are successfully found
            if all(v is not None for v in found_situations.values()):
                break

    # Ensure k is at least 2 for basic functionality
    n_components = max(2, n_components)
    dff = DeepFeatureFactorization(model=model, target_layer=target_layer)
    
    # Calculate the number of images per row for the DFF concepts
    r1_cols = max(1, n_components // 2)
    r2_cols = n_components - r1_cols
    
    # Calculate total GridSpec width to ensure perfect centering for any number of columns
    gs_width = 4 * r1_cols * r2_cols
    w0 = gs_width // 4         # Width of each image in the first row
    w1 = gs_width // r1_cols   # Width of each image in the second row
    w2 = gs_width // r2_cols   # Width of each image in the third row
    
    for sit_name, data in found_situations.items():
        if data is None:
            print(f"Could not find situation [{sit_name}] in the provided batches.")
            continue
            
        img_tensor, mask, pred = data
        mask = mask.numpy()
        pred = pred.numpy()
        
        # Run DFF factorization
        heatmaps = dff.factorize(img_tensor, n_components=n_components)
        
        # Denormalize image for visualization
        img_cpu = img_tensor.squeeze(0).cpu()
        img_vis = (img_cpu * std + mean).permute(1, 2, 0).numpy()
        img_vis = np.clip(img_vis, 0, 1)
        
        # Generate Overlap map
        overlap = np.zeros((mask.shape[0], mask.shape[1], 3))
        overlap[(pred == 1) & (mask == 1)] = [0, 1, 0] # TP (Green)
        overlap[(pred == 0) & (mask == 1)] = [1, 0, 0] # FN (Red)
        overlap[(pred == 1) & (mask == 0)] = [1, 1, 0] # FP (Yellow)
        
        # --- Dynamic Centered Layout System ---
        fig = plt.figure(figsize=(18, 14))
        gs = fig.add_gridspec(3, gs_width) 
        
        # Row 1: Always the 4 standard base views
        ax_img = fig.add_subplot(gs[0, 0*w0 : 1*w0])
        ax_mask = fig.add_subplot(gs[0, 1*w0 : 2*w0])
        ax_pred = fig.add_subplot(gs[0, 2*w0 : 3*w0])
        ax_over = fig.add_subplot(gs[0, 3*w0 : 4*w0])
        
        ax_img.imshow(img_vis)
        ax_img.set_title(f"[{sit_name}] Original Image", fontsize=14)
        ax_img.axis('off')
        
        ax_mask.imshow(mask * 255, cmap='gray')
        ax_mask.set_title("Ground Truth Mask", fontsize=14)
        ax_mask.axis('off')
        
        ax_pred.imshow(pred * 255, cmap='gray')
        ax_pred.set_title("Prediction", fontsize=14)
        ax_pred.axis('off')
        
        ax_over.imshow(overlap)
        ax_over.set_title("Prediction Overlap", fontsize=14)
        ax_over.axis('off')
        
        # Row 2: First half of DFF concepts (k // 2)
        for i in range(r1_cols):
            ax = fig.add_subplot(gs[1, i*w1 : (i+1)*w1])
            heatmap = heatmaps[i]
            # Min-Max normalize heatmap for display clarity
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
            
            ax.imshow(img_vis)
            ax.imshow(heatmap, cmap='jet', alpha=0.5)
            ax.set_title(f"DFF Concept {i+1}", fontsize=14)
            ax.axis('off')
            
        # Row 3: Remaining DFF concepts
        for i in range(r2_cols):
            ax = fig.add_subplot(gs[2, i*w2 : (i+1)*w2])
            heatmap_idx = r1_cols + i
            heatmap = heatmaps[heatmap_idx]
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
            
            ax.imshow(img_vis)
            ax.imshow(heatmap, cmap='jet', alpha=0.5)
            ax.set_title(f"DFF Concept {heatmap_idx+1}", fontsize=14)
            ax.axis('off')

        # Legend labels
        patches = [
            mpatches.Patch(color='green', label='Correct (TP)'),
            mpatches.Patch(color='red', label='Missed (FN)'),
            mpatches.Patch(color='yellow', label='False Alarm (FP)')
        ]
        fig.legend(handles=patches, loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0.05), fontsize=14)

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.1) 
        
        save_path = os.path.join(save_dir, f"DFF_Analysis_{sit_name}.png")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved {sit_name} visual to {save_path}")

        # Force memory cleanup to prevent OOM (Out of Memory) when generating many charts
        del heatmaps, img_vis, overlap, fig
        gc.collect() 

    dff.remove_hook()