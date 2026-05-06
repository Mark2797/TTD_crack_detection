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


def run_dff_for_multiple_models(models, model_names, dataloaders_list_per_model, target_layers, device, save_dir="figures/dff_analysis", custom_stats_list=None, n_components=6):
    """
    Independently scans a list of dataloaders for each model to find its own first TP, FP, FN, TN situations.
    Generates a 4-row layout:
      - Row 1: Channel 0, Channel 1, Channel 2
      - Row 2: GT Mask, Prediction, Overlap
      - Row 3 & 4: DFF Concepts overlaid EXCLUSIVELY on Channel 0
    """
    os.makedirs(save_dir, exist_ok=True)
    num_models = len(models)
    
    # Handle normalization stats: fallback to imagenet if list is missing or contains None
    if custom_stats_list is None:
        custom_stats_list = [imagenet_stats] * num_models
    else:
        custom_stats_list = [s if s is not None else imagenet_stats for s in custom_stats_list]

    n_components = max(2, n_components)
    r1_cols = max(1, n_components // 2)
    r2_cols = n_components - r1_cols
    
    # Calculate GridSpec width to center 3 top columns with the dynamic DFF rows below
    gs_width = 3 * r1_cols * r2_cols
    w_top = gs_width // 3      # Width for elements in the top two 3-column rows
    w1 = gs_width // r1_cols   
    w2 = gs_width // r2_cols   
    
    for m_idx, model in enumerate(models):
        model_name = model_names[m_idx]
        model_dataloaders = dataloaders_list_per_model[m_idx] 
        target_layer = target_layers[m_idx]
        stats = custom_stats_list[m_idx]
        
        model.to(device)
        model.eval()
        
        mean = torch.as_tensor(stats[0], device="cpu").clone().detach().view(3, 1, 1)
        std = torch.as_tensor(stats[1], device="cpu").clone().detach().view(3, 1, 1)

        found_situations = {"TP": None, "FP": None, "FN": None, "TN": None}
        
        print(f"\n--- Processing {model_name} (k={n_components}) ---")
        
        # --- PHASE 1: Find situations across dataloaders ---
        for dl_idx, dataloader in enumerate(model_dataloaders):
            if all(v is not None for v in found_situations.values()):
                break 
                
            print(f"Scanning dataloader {dl_idx + 1}/{len(model_dataloaders)}...")
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

        # --- PHASE 2: Generate DFF Layouts ---
        print(f"Generating DFF plots for {model_name}...")
        dff = DeepFeatureFactorization(model=model, target_layer=target_layer)
        
        for sit_name, data in found_situations.items():
            if data is None:
                print(f"  -> Warning: Could not find situation [{sit_name}] for {model_name}.")
                continue
                
            img_tensor, mask, pred = data
            mask_np = mask.numpy()
            pred_np = pred.numpy()
            
            heatmaps = dff.factorize(img_tensor.to(device), n_components=n_components)
            
            img_cpu = img_tensor.squeeze(0).cpu()
            
            # --- SEPARATE AND DENORMALIZE THE 3 CHANNELS INDEPENDENTLY ---
            ch0_vis = np.clip((img_cpu[0] * std[0, 0, 0] + mean[0, 0, 0]).numpy(), 0, 1)
            ch1_vis = np.clip((img_cpu[1] * std[1, 0, 0] + mean[1, 0, 0]).numpy(), 0, 1)
            ch2_vis = np.clip((img_cpu[2] * std[2, 0, 0] + mean[2, 0, 0]).numpy(), 0, 1)
            
            overlap = np.zeros((mask_np.shape[0], mask_np.shape[1], 3))
            overlap[(pred_np == 1) & (mask_np == 1)] = [0, 1, 0] 
            overlap[(pred_np == 0) & (mask_np == 1)] = [1, 0, 0] 
            overlap[(pred_np == 1) & (mask_np == 0)] = [1, 1, 0] 
            
            # Increased height slightly to comfortably fit the 4th row
            fig = plt.figure(figsize=(18, 18))
            gs = fig.add_gridspec(4, gs_width) 
            
            # --- Row 1: The 3 Channels ---
            ax_ch0 = fig.add_subplot(gs[0, 0*w_top : 1*w_top])
            ax_ch1 = fig.add_subplot(gs[0, 1*w_top : 2*w_top])
            ax_ch2 = fig.add_subplot(gs[0, 2*w_top : 3*w_top])
            
            ax_ch0.imshow(ch0_vis, cmap='gray', vmin=0, vmax=1)
            ax_ch0.set_title(f"[{sit_name}] Channel 0", fontsize=14)
            ax_ch0.axis('off')

            ax_ch1.imshow(ch1_vis, cmap='gray', vmin=0, vmax=1)
            ax_ch1.set_title(f"[{sit_name}] Channel 1", fontsize=14)
            ax_ch1.axis('off')

            ax_ch2.imshow(ch2_vis, cmap='gray', vmin=0, vmax=1)
            ax_ch2.set_title(f"[{sit_name}] Channel 2", fontsize=14)
            ax_ch2.axis('off')
            
            # --- Row 2: Mask, Pred, Overlap ---
            ax_mask = fig.add_subplot(gs[1, 0*w_top : 1*w_top])
            ax_pred = fig.add_subplot(gs[1, 1*w_top : 2*w_top])
            ax_over = fig.add_subplot(gs[1, 2*w_top : 3*w_top])
            
            ax_mask.imshow(mask_np * 255, cmap='gray')
            ax_mask.set_title("Ground Truth Mask", fontsize=14)
            ax_mask.axis('off')
            
            ax_pred.imshow(pred_np * 255, cmap='gray')
            ax_pred.set_title("Prediction", fontsize=14)
            ax_pred.axis('off')
            
            ax_over.imshow(overlap)
            ax_over.set_title("Prediction Overlap", fontsize=14)
            ax_over.axis('off')
            
            # --- Row 3 & 4: DFF Concepts overlaid ONLY on Channel 0 ---
            for i in range(r1_cols):
                ax = fig.add_subplot(gs[2, i*w1 : (i+1)*w1])
                heatmap = heatmaps[i]
                heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
                
                ax.imshow(ch0_vis, cmap='gray', vmin=0, vmax=1) # Background is strict Ch0
                ax.imshow(heatmap, cmap='jet', alpha=0.5)
                ax.set_title(f"DFF Concept {i+1}", fontsize=14)
                ax.axis('off')
                
            for i in range(r2_cols):
                ax = fig.add_subplot(gs[3, i*w2 : (i+1)*w2])
                heatmap_idx = r1_cols + i
                heatmap = heatmaps[heatmap_idx]
                heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
                
                ax.imshow(ch0_vis, cmap='gray', vmin=0, vmax=1) # Background is strict Ch0
                ax.imshow(heatmap, cmap='jet', alpha=0.5)
                ax.set_title(f"DFF Concept {heatmap_idx+1}", fontsize=14)
                ax.axis('off')

            patches = [
                mpatches.Patch(color='green', label='Correct (TP)'),
                mpatches.Patch(color='red', label='Missed (FN)'),
                mpatches.Patch(color='yellow', label='False Alarm (FP)')
            ]
            fig.legend(handles=patches, loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0.05), fontsize=14)

            plt.tight_layout()
            plt.subplots_adjust(bottom=0.08) 
            
            save_path = os.path.join(save_dir, f"{model_name}_DFF_Analysis_{sit_name}.png")
            plt.savefig(save_path, bbox_inches='tight')
            plt.close(fig)

            del heatmaps, ch0_vis, ch1_vis, ch2_vis, overlap, fig
            gc.collect() 

        dff.remove_hook()