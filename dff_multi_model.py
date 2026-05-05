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

def find_consensus_and_run_dff(models, model_names, dataloaders, target_layers, device, save_dir="figures/dff_analysis", custom_stats_list=None, n_components=6):
    """
    Scans multiple dataloaders to find the EXACT same images where all models 
    yield the same classification (TP, FP, FN, TN). Then, generates the standard 
    individual DFF layouts for each model using those consensus images.
    """
    os.makedirs(save_dir, exist_ok=True)
    num_models = len(models)
    
    for model in models:
        model.to(device)
        model.eval()
        
    # Handle normalization stats: fallback to imagenet if list is missing or contains None
    if custom_stats_list is None:
        custom_stats_list = [imagenet_stats] * num_models
    else:
        # Replace any individual None values inside the list with imagenet_stats
        custom_stats_list = [s if s is not None else imagenet_stats for s in custom_stats_list]

    found_situations = {"TP": None, "FP": None, "FN": None, "TN": None}
    
    print(f"Scanning dataloaders to find consensus edge cases across {num_models} models...")
    
    # --- PHASE 1: Find Consensus Images ---
    with torch.no_grad():
        for batches in zip(*dataloaders):
            masks = batches[0][1] # Ground truth mask (identical across dataloaders)
            
            # Gather predictions from all models for this batch
            preds_list = []
            for m_idx, model in enumerate(models):
                images = batches[m_idx][0].to(device)
                outputs = model(images)
                preds_list.append(torch.argmax(outputs, dim=1).cpu())
                
            for i in range(len(masks)):
                mask = masks[i].squeeze().cpu()
                mask_has_crack = mask.sum() > 0
                
                # Check predictions across all models for this specific image
                preds_for_i = [preds[i].squeeze() for preds in preds_list]
                preds_have_crack = [p.sum() > 0 for p in preds_for_i]
                
                all_predict_crack = all(preds_have_crack)
                all_predict_no_crack = not any(preds_have_crack)
                
                # Store the image tensors, the shared mask, and the respective predictions
                if mask_has_crack and all_predict_crack and found_situations["TP"] is None:
                    found_situations["TP"] = {"imgs": [b[0][i].unsqueeze(0) for b in batches], "mask": mask, "preds": preds_for_i}
                elif not mask_has_crack and all_predict_crack and found_situations["FP"] is None:
                    found_situations["FP"] = {"imgs": [b[0][i].unsqueeze(0) for b in batches], "mask": mask, "preds": preds_for_i}
                elif mask_has_crack and all_predict_no_crack and found_situations["FN"] is None:
                    found_situations["FN"] = {"imgs": [b[0][i].unsqueeze(0) for b in batches], "mask": mask, "preds": preds_for_i}
                elif not mask_has_crack and all_predict_no_crack and found_situations["TN"] is None:
                    found_situations["TN"] = {"imgs": [b[0][i].unsqueeze(0) for b in batches], "mask": mask, "preds": preds_for_i}
            
            if all(v is not None for v in found_situations.values()):
                break

    # --- PHASE 2: Generate DFF Layouts for Each Model ---
    n_components = max(2, n_components)
    r1_cols = max(1, n_components // 2)
    r2_cols = n_components - r1_cols
    
    gs_width = 4 * r1_cols * r2_cols
    w0 = gs_width // 4         
    w1 = gs_width // r1_cols   
    w2 = gs_width // r2_cols   
    
    for m_idx, model in enumerate(models):
        model_name = model_names[m_idx]
        target_layer = target_layers[m_idx]
        stats = custom_stats_list[m_idx]
        
        mean = torch.as_tensor(stats[0], device="cpu").clone().detach().view(3, 1, 1)
        std = torch.as_tensor(stats[1], device="cpu").clone().detach().view(3, 1, 1)
        
        print(f"Generating DFF plots for {model_name}...")
        dff = DeepFeatureFactorization(model=model, target_layer=target_layer)
        
        for sit_name, data in found_situations.items():
            if data is None:
                if m_idx == 0: # Only print warning once
                    print(f"  -> Could not find a consensus image for [{sit_name}].")
                continue
                
            img_tensor = data["imgs"][m_idx]
            mask_np = data["mask"].numpy()
            pred_np = data["preds"][m_idx].numpy()
            
            heatmaps = dff.factorize(img_tensor.to(device), n_components=n_components)
            
            img_cpu = img_tensor.squeeze(0).cpu()
            # img_vis = (img_cpu * std + mean).permute(1, 2, 0).numpy()
            # img_vis = np.clip(img_vis, 0, 1)
            
            img_vis = img_cpu[0] * std[0, 0, 0] + mean[0, 0, 0]
            img_vis = img_vis.numpy()
            img_vis = np.clip(img_vis, 0, 1)
            
            overlap = np.zeros((mask_np.shape[0], mask_np.shape[1], 3))
            overlap[(pred_np == 1) & (mask_np == 1)] = [0, 1, 0] 
            overlap[(pred_np == 0) & (mask_np == 1)] = [1, 0, 0] 
            overlap[(pred_np == 1) & (mask_np == 0)] = [1, 1, 0] 
            
            fig = plt.figure(figsize=(18, 14))
            gs = fig.add_gridspec(3, gs_width) 
            
            ax_img = fig.add_subplot(gs[0, 0*w0 : 1*w0])
            ax_mask = fig.add_subplot(gs[0, 1*w0 : 2*w0])
            ax_pred = fig.add_subplot(gs[0, 2*w0 : 3*w0])
            ax_over = fig.add_subplot(gs[0, 3*w0 : 4*w0])
            
            # ax_img.imshow(img_vis)
            ax_img.imshow(img_vis, cmap='gray', vmin=0, vmax=1)
            ax_img.set_title(f"[{sit_name}] Original Image", fontsize=14)
            ax_img.axis('off')
            
            ax_mask.imshow(mask_np * 255, cmap='gray')
            ax_mask.set_title("Ground Truth Mask", fontsize=14)
            ax_mask.axis('off')
            
            ax_pred.imshow(pred_np * 255, cmap='gray')
            ax_pred.set_title("Prediction", fontsize=14)
            ax_pred.axis('off')
            
            ax_over.imshow(overlap)
            ax_over.set_title("Prediction Overlap", fontsize=14)
            ax_over.axis('off')
            
            for i in range(r1_cols):
                ax = fig.add_subplot(gs[1, i*w1 : (i+1)*w1])
                heatmap = heatmaps[i]
                heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
                
                #ax.imshow(img_vis)
                #ax.imshow(heatmap, cmap='jet', alpha=0.5)
                
                ax.imshow(img_vis, cmap='gray', vmin=0, vmax=1)
                ax.imshow(heatmap, cmap='jet', alpha=0.5)
                ax.set_title(f"DFF Concept {i+1}", fontsize=14)
                ax.axis('off')
                
            for i in range(r2_cols):
                ax = fig.add_subplot(gs[2, i*w2 : (i+1)*w2])
                heatmap_idx = r1_cols + i
                heatmap = heatmaps[heatmap_idx]
                heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
                
                #ax.imshow(img_vis)
                #ax.imshow(heatmap, cmap='jet', alpha=0.5)
                
                ax.imshow(img_vis, cmap='gray', vmin=0, vmax=1)
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
            plt.subplots_adjust(bottom=0.1) 
            
            save_path = os.path.join(save_dir, f"{model_name}_DFF_Analysis_{sit_name}.png")
            plt.savefig(save_path, bbox_inches='tight')
            plt.close(fig)

            del heatmaps, img_vis, overlap, fig
            gc.collect() 

        dff.remove_hook()