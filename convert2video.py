import os
from pathlib import Path
from PIL import Image # Requires: pip install Pillow
import re
import math # Needed for ceiling function or rounding

# --- Configuration for Resizing ---
TARGET_WIDTH = 1600

def convert_to_rgb_video(sorted_image_dir: str, output_filepath: str, file_extension: str = '.jpg'):
    """
    Reads sequentially named JPEG (or PNG) images from a directory, extracts 
    the raw RGB pixel data for each frame, and concatenates them into a single 
    raw .rgb video file.
    
    The images are resized to TARGET_WIDTH (1600) while maintaining aspect ratio,
    and the resulting height is ensured to be divisible by 2.

    Args:
        sorted_image_dir (str): Directory containing frames like frame_00000.jpg.
        output_filepath (str): The path to the final raw .rgb video file.
        file_extension (str): The extension of the input images (e.g., '.jpg').
    """
    
    source_path = Path(sorted_image_dir)
    
    # 1. Collect and sort sequentially named files
    # We use glob and natural sorting to ensure correct order: frame_00000, frame_00001, ...
    frame_files = sorted(
        [f for f in source_path.glob(f"*{file_extension}")],
        key=lambda f: int(re.search(r'_(\d+)', f.stem).group(1)) # Extract sequential number for sorting
    )
    
    if not frame_files:
        print(f"Error: No frames found in {sorted_image_dir} with extension {file_extension}.")
        return

    num_frames = len(frame_files)
    print(f"Found {num_frames} frames to process.")
    
    # 2. Determine frame dimensions, calculate new resolution
    original_width, original_height = 0, 0
    resized_width, resized_height = 0, 0
    
    try:
        with Image.open(frame_files[0]) as img:
            original_width, original_height = img.size
            
            # --- RESIZING LOGIC START ---
            
            # Calculate the new height proportionally
            ratio = original_height / original_width
            temp_height = TARGET_WIDTH * ratio
            
            # Round the height to the nearest integer
            resized_height = round(temp_height)
            resized_width = TARGET_WIDTH
            
            # Ensure height is divisible by 2 (standard video requirement)
            if resized_height % 2 != 0:
                # We round down to the nearest even number
                resized_height -= 1 
            
            # --- RESIZING LOGIC END ---
            
        print(f"Original resolution: {original_width}x{original_height}")
        print(f"Target resolution (Resized): {resized_width}x{resized_height} (Height is even)")
        
    except Exception as e:
        print(f"Error opening or reading first image ({frame_files[0]}): {e}")
        return
    
    # Check if the calculated dimensions are valid
    if resized_width <= 0 or resized_height <= 0:
        print("Error: Calculated resize dimensions are invalid.")
        return

    # 3. Concatenate raw pixel data
    total_bytes_written = 0
    try:
        with open(output_filepath, 'wb') as f_out:
            for i, frame_file in enumerate(frame_files):
                try:
                    with Image.open(frame_file) as img:
                        
                        # Apply resizing using a high-quality filter
                        img_resized = img.resize(
                            (resized_width, resized_height), 
                            Image.LANCZOS # High quality filter
                        )
                        
                        # Convert to 'RGB' mode explicitly
                        img_rgb = img_resized.convert('RGB')
                        
                        # Get the raw byte data (R, G, B, R, G, B, ...)
                        raw_data = img_rgb.tobytes()
                        
                        f_out.write(raw_data)
                        total_bytes_written += len(raw_data)
                        
                    if (i + 1) % 100 == 0:
                        print(f"  -> Processed {i+1}/{num_frames} frames...")
                        
                except Exception as e:
                    print(f"Error processing frame {frame_file.name}: {e}. Skipping.")

        print("\n" + "="*60)
        print(f"✅ Conversion successful! Raw RGB video saved to: {output_filepath}")
        print(f"Total size: {total_bytes_written / (1024*1024):.2f} MB")
        print(f"Metadata needed for playback: {resized_width}x{resized_height}, 24-bit RGB, {num_frames} frames")
        print("="*60)

    except Exception as e:
        print(f"An error occurred during file writing: {e}")


if __name__ == "__main__":
    
    # The directory where the reorder_images.py script placed the sequentially numbered frames
    SORTED_IMAGE_DIRECTORY = "data/mip-nerf360/bicycle/images_sorted"
    
    # The output raw video file
    OUTPUT_VIDEO_FILE = "bicycle_sorted.rgb"
    
    # Assuming the input files are JPEGs as defined in the previous script
    INPUT_EXTENSION = '.jpg' 
    
    # This function creates a single raw RGB video file
    convert_to_rgb_video(
        SORTED_IMAGE_DIRECTORY, 
        OUTPUT_VIDEO_FILE, 
        INPUT_EXTENSION
    )
