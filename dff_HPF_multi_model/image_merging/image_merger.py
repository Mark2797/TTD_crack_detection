import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def create_tight_image_chain(image_paths, titles, output_name="high_res_tight_chain.png"):
    """
    Creates a chain with minimal, clean spacing and high-res output.
    """
    if len(image_paths) != len(titles):
        print("Error: The number of images must match the number of titles.")
        return

    num_images = len(image_paths)
    
    # figsize: Keep height constant (7), width scales with images.
    fig, axes = plt.subplots(1, num_images, figsize=(5 * num_images, 7))

    if num_images == 1:
        axes = [axes]

    for i, (path, title) in enumerate(zip(image_paths, titles)):
        try:
            img = mpimg.imread(path)
            h, w = img.shape[:2]
            
            # --- REDUCED INTERNAL PADDING ---
            # Set to 1% or 2% for just a hint of a margin inside the box
            pad_percent = 0.015 
            axes[i].imshow(img, interpolation='lanczos')
            
            axes[i].set_xlim(-w * pad_percent, w * (1 + pad_percent))
            axes[i].set_ylim(h * (1 + pad_percent), -h * pad_percent)
            
            # Formatting
            axes[i].set_title(title, fontsize=14, fontweight='bold', pad=10)
            axes[i].set_xticks([])
            axes[i].set_yticks([])
            
            # Slimmer border for a cleaner look
            for spine in axes[i].spines.values():
                spine.set_linewidth(1.5)
                spine.set_color('#333333')
            
        except Exception as e:
            print(f"Error: {e}")

    # --- REDUCED EXTERNAL GAP ---
    # 0.05 is a very small gap. Increase to 0.1 if it feels too cramped.
    plt.subplots_adjust(wspace=0.05) 

    # Save with high DPI for crispness
    plt.savefig(output_name, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.show()

# --- RUN IT ---
# img_list = [...] 
# title_list = [...]
# create_tight_image_chain(img_list, title_list)