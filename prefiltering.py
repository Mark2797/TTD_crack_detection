# -*- coding: utf-8 -*-
import os
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm

#%%

def normalize_to_uint8(arr):
    """
    Normalize a float array to uint8 [0, 255].
    """
    arr = arr.astype(np.float32)
    arr_min = arr.min()
    arr_max = arr.max()

    if arr_max - arr_min < 1e-8:
        return np.zeros_like(arr, dtype=np.uint8)

    arr = (arr - arr_min) / (arr_max - arr_min)
    arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
    return arr


def make_sobel_xy_3ch_image(img_path):
    """
    3-channel output:
        channel 1 = original grayscale
        channel 2 = |Sobel-x|
        channel 3 = |Sobel-y|
    """
    gray = np.array(Image.open(img_path).convert('L'))

    sobel_x = cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)

    ch1 = gray.astype(np.uint8)
    ch2 = normalize_to_uint8(np.abs(sobel_x))
    ch3 = normalize_to_uint8(np.abs(sobel_y))

    img_3ch = np.stack([ch1, ch2, ch3], axis=-1)
    return img_3ch


def make_sobel_mag_lap_3ch_image(img_path):
    """
    3-channel output:
        channel 1 = original grayscale
        channel 2 = Sobel magnitude
        channel 3 = |Laplacian|
    """
    gray = np.array(Image.open(img_path).convert('L')).astype(np.float32)

    sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    sobel_mag = np.sqrt(sobel_x**2 + sobel_y**2)
    
    # Guassian blur and laplacian (LoG)
    gray_blur = cv2.GaussianBlur(gray, (3,3), 0)
    lap = cv2.Laplacian(gray_blur, cv2.CV_32F, ksize=3)

    ch1 = gray.astype(np.uint8)
    ch2 = normalize_to_uint8(sobel_mag)
    ch3 = normalize_to_uint8(np.abs(lap))

    img_3ch = np.stack([ch1, ch2, ch3], axis=-1)
    return img_3ch


def make_sobel_mag_gabor_3ch_image(
    img_path,
    ksize=31,
    sigma=4,
    lambd=10,
    gamma=0.5,
    psi=0,
):
    """
    3-channel output:
        channel 1 = original grayscale
        channel 2 = Sobel magnitude
        channel 3 = max |Gabor response| over orientations
    """
    gray = np.array(Image.open(img_path).convert('L')).astype(np.float32)

    sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    sobel_mag = np.sqrt(sobel_x**2 + sobel_y**2)

    responses = []
    for theta in np.arange(0, np.pi, np.pi / 4):
        kernel = cv2.getGaborKernel(
            (ksize, ksize),
            sigma,
            theta,
            lambd,
            gamma,
            psi,
            ktype=cv2.CV_32F,
        )
        response = cv2.filter2D(gray, cv2.CV_32F, kernel)
        responses.append(np.abs(response))

    gabor_max = np.max(np.stack(responses, axis=0), axis=0)

    ch1 = gray.astype(np.uint8)
    ch2 = normalize_to_uint8(sobel_mag)
    ch3 = normalize_to_uint8(gabor_max)

    img_3ch = np.stack([ch1, ch2, ch3], axis=-1)
    return img_3ch


def process_folder(input_folder, output_folder, mode="sobelxy"):
    """
    Read all images in input_folder, process them, and save to output_folder.

    mode:
        "sobelxy"         -> save as originalname_sobelxy.ext
        "sobelmaglap"     -> save as originalname_sobelmaglap.ext
        "sobelmag_gabor"  -> save as originalname_sobelmag_gabor.ext
    """
    os.makedirs(output_folder, exist_ok=True)

    valid_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')

    file_list = [
        fname for fname in os.listdir(input_folder)
        if fname.lower().endswith(valid_exts)
    ]

    for fname in tqdm(file_list, desc=f"Processing {mode}", unit="img"):
        input_path = os.path.join(input_folder, fname)
        base, ext = os.path.splitext(fname)

        try:
            if mode == "sobelxy":
                img_3ch = make_sobel_xy_3ch_image(input_path)
                output_name = f"{base}_sobelxy{ext}"
            elif mode == "sobelmaglap":
                img_3ch = make_sobel_mag_lap_3ch_image(input_path)
                output_name = f"{base}_sobelmaglap{ext}"
            elif mode == "sobelmag_gabor":
                img_3ch = make_sobel_mag_gabor_3ch_image(input_path)
                output_name = f"{base}_sobelmag_gabor{ext}"
            else:
                raise ValueError(f"Unknown mode: {mode}")

            output_path = os.path.join(output_folder, output_name)
            Image.fromarray(img_3ch).save(output_path)

        except Exception as e:
            print(f"Error processing {fname}: {e}")


if __name__ == "__main__":
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_folder = os.path.join(parent_dir, "TACK_Tunnel_Data")
    img_source_dir = os.path.join(dataset_folder, "3_img")

    input_folder = img_source_dir

    output_folder_sobelxy = os.path.join(dataset_folder, "4_img_sobelxy")
    output_folder_sobelmaglap = os.path.join(dataset_folder, "4_img_sobelmaglap")
    output_folder_sobelmag_gabor = os.path.join(dataset_folder, "4_img_sobelmag_gabor")
#%% Save the Sobel-x Sobel-y in the additional two channels
    print("Processing Sobel-x / Sobel-y version...")
    process_folder(
        input_folder=input_folder,
        output_folder=output_folder_sobelxy,
        mode="sobelxy"
    )

#%% Save the magnitude Sobelxy and LoG in the additional two channels
    print("Processing Sobel magnitude / Laplacian version...")
    process_folder(
        input_folder=input_folder,
        output_folder=output_folder_sobelmaglap,
        mode="sobelmaglap"
    )

#%% Save the Sobel magnitude and Gabor max response in the additional two channels
    print("Processing Sobel magnitude / Gabor max-response version...")
    process_folder(
        input_folder=input_folder,
        output_folder=output_folder_sobelmag_gabor,
        mode="sobelmag_gabor"
    )

    print("All done.")
