import os
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
        Initializes the DFF hook on a specific model layer.
        """
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.hook = self.target_layer.register_forward_hook(self._hook_fn)

    def _hook_fn(self, module, input, output):
        self.activations = output.detach().cpu()

    def remove_hook(self):
        self.hook.remove()

    def factorize(self, input_tensor, n_components=5):
        """
        Runs the image through the model and factorizes the captured activations.
        """
        self.model.eval()
        with torch.no_grad():
            _ = self.model(input_tensor)
            
        # Squeeze batch dim -> (Channels, H_act, W_act)
        A = self.activations.squeeze(0)
        channels, h_act, w_act = A.shape
        
        # Flatten spatial dimensions for NMF: Shape (Channels, H_act * W_act)
        A_flat = A.view(channels, -1).numpy()
        A_flat = np.maximum(A_flat, 0)
        
        # Apply Non-negative Matrix Factorization (NMF)
        X = A_flat.T 
        nmf = NMF(n_components=n_components, init='nndsvd', random_state=42, max_iter=500)
        W = nmf.fit_transform(X) 
        
        # Reshape the components back to spatial heatmaps
        heatmaps = W.T.reshape(n_components, h_act, w_act)
        
        # Upsample heatmaps to match original image size (H, W)
        _, _, H, W_img = input_tensor.shape
        heatmaps_tensor = torch.tensor(heatmaps).unsqueeze(0)
        heatmaps_upsampled = F.interpolate(heatmaps_tensor, size=(H, W_img), mode='bilinear', align_corners=False)
        
        return heatmaps_upsampled.squeeze(0).cpu().numpy()


def find_situations_and_run_dff(model, dataloader, target_layer, device, save_dir="figures/dff_analysis", custom_stats=None, n_components=6):
    """
    Scans the dataloader for TP, FP, FN, TN situations, runs DFF (k=6), 
    and plots them in a perfectly centered 4-3-3 layout.
    """
    os.makedirs(save_dir, exist_ok=True)
    model.to(device)
    model.eval()
    
    stats = custom_stats if custom_stats else imagenet_stats
    mean = torch.tensor(stats[0], device="cpu").view(3, 1, 1)
    std = torch.tensor(stats[1], device="cpu").view(3, 1, 1)

    found_situations = {"TP": None, "FP": None, "FN": None, "TN": None}
    
    print("Scanning dataloader for edge cases (TP, FP, FN, TN)...")
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
                
                if mask_has_crack and pred_has_crack and found_situations["TP"] is None:
                    found_situations["TP"] = (img, mask, pred)
                elif not mask_has_crack and pred_has_crack and found_situations["FP"] is None:
                    found_situations["FP"] = (img, mask, pred)
                elif mask_has_crack and not pred_has_crack and found_situations["FN"] is None:
                    found_situations["FN"] = (img, mask, pred)
                elif not mask_has_crack and not pred_has_crack and found_situations["TN"] is None:
                    found_situations["TN"] = (img, mask, pred)
            
            if all(v is not None for v in found_situations.values()):
                break

    # Initialize the hook
    dff = DeepFeatureFactorization(model=model, target_layer=target_layer)
    
    for sit_name, data in found_situations.items():
        if data is None:
            print(f"Could not find situation [{sit_name}] in the provided batches.")
            continue
            
        img_tensor, mask, pred = data
        mask = mask.numpy()
        pred = pred.numpy()
        
        # Run factorization 
        heatmaps = dff.factorize(img_tensor, n_components=n_components)
        
        # Denormalize
        img_cpu = img_tensor.squeeze(0).cpu()
        img_vis = (img_cpu * std + mean).permute(1, 2, 0).numpy()
        img_vis = np.clip(img_vis, 0, 1)
        
        # Overlap map
        overlap = np.zeros((mask.shape[0], mask.shape[1], 3))
        overlap[(pred == 1) & (mask == 1)] = [0, 1, 0] # TP (Green)
        overlap[(pred == 0) & (mask == 1)] = [1, 0, 0] # FN (Red)
        overlap[(pred == 1) & (mask == 0)] = [1, 1, 0] # FP (Yellow)
        
        # --- Create Custom Centered Layout using GridSpec ---
        fig = plt.figure(figsize=(18, 14))
        gs = fig.add_gridspec(3, 12) # 3 rows, 12 invisible columns
        
        # Row 0: 4 standard views (each spans 3 columns: 12/4 = 3)
        ax_img = fig.add_subplot(gs[0, 0:3])
        ax_mask = fig.add_subplot(gs[0, 3:6])
        ax_pred = fig.add_subplot(gs[0, 6:9])
        ax_over = fig.add_subplot(gs[0, 9:12])
        
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
        
        # Rows 1 & 2: DFF Concepts (each spans 4 columns: 12/3 = 4)
        # This naturally centers the 3 images relative to the 4 above them!
        dff_axes = [
            fig.add_subplot(gs[1, 0:4]), fig.add_subplot(gs[1, 4:8]), fig.add_subplot(gs[1, 8:12]),
            fig.add_subplot(gs[2, 0:4]), fig.add_subplot(gs[2, 4:8]), fig.add_subplot(gs[2, 8:12])
        ]
        
        for i in range(min(n_components, 6)): # Safety check to ensure we only plot 6
            ax = dff_axes[i]
            heatmap = heatmaps[i]
            
            # Normalize heatmap for display
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
            
            ax.imshow(img_vis)
            ax.imshow(heatmap, cmap='jet', alpha=0.5)
            ax.set_title(f"DFF Concept {i+1}", fontsize=14)
            ax.axis('off')

        # Legend
        patches = [
            mpatches.Patch(color='green', label='Correct (TP)'),
            mpatches.Patch(color='red', label='Missed (FN)'),
            mpatches.Patch(color='yellow', label='False Alarm (FP)')
        ]
        
        fig.legend(handles=patches, loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0.05), fontsize=14)

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.1) # Leave room for the legend
        
        save_path = os.path.join(save_dir, f"DFF_Analysis_{sit_name}.png")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved {sit_name} visual to {save_path}")

    dff.remove_hook()

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
    
    # 2. Find a single representative image with a defect
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
    A_flat = np.maximum(A_flat, 0) # NMF requires non-negative inputs
    X = A_flat.T 
    
    # 5. Calculate Reconstruction Error for each k
    errors = []
    print(f"Calculating NMF errors for k in {list(k_range)}...")
    for k in k_range:
        nmf = NMF(n_components=k, init='nndsvd', random_state=42, max_iter=500)
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