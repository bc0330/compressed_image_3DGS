import os
import glob
import shutil
import csv
from natsort import natsorted

# --- 1. CORE IMAGE SELECTION LOGIC (REVISED) ---

def safe_append(lst, group, index, list_name):
    """Safely appends an element from 'group' to 'lst' if the index exists."""
    if index < len(group):
        lst.append(group[index])
        # print(f"  -> Added {os.path.basename(group[index])} to {list_name}") # Uncomment for debugging
    # else:
        # print(f"  -> Skipped index {index} (Image #{index+1}) for a partial group of size {len(group)}") # Uncomment for debugging

def select_and_sort_images(folder_path, extensions=['.JPG']):
    """
    Reads images from a folder, sorts them by name, and splits specific
    images from each group of 8 into 'odd_images' and 'even_images' lists,
    **processing partial groups if they remain**.
    """
    odd_images = []
    even_images = []

    # Get all files with specified image extensions
    all_files = []
    for ext in extensions:
        # Note: glob is case-sensitive on some systems. Using both .JPG and .jpg is safer.
        pattern = os.path.join(folder_path, f"*{ext}")
        all_files.extend(glob.glob(pattern))

    if not all_files:
        print(f"⚠️ No images found in the folder: {folder_path}")
        return odd_images, even_images

    # Sort the images naturally (file1, file2, ..., file10)
    try:
        sorted_images = natsorted(all_files)
        print(f"✅ Found and naturally sorted {len(sorted_images)} images.")
    except NameError:
        sorted_images = sorted(all_files)
        print(f"✅ Found and standard sorted {len(sorted_images)} images.")


    # Process the images in sets of 8 (or fewer for the last group)
    GROUP_SIZE = 8
    for i in range(0, len(sorted_images), GROUP_SIZE):
        current_group = sorted_images[i:i + GROUP_SIZE]

        if len(current_group) < GROUP_SIZE:
             print(f"\n⚠️ Processing partial group of size: {len(current_group)}.")
        else:
             print(f"\nProcessing full group (index {i}).")

        # The core logic now uses safe_append to handle missing indices.
        # IGNORE the 1st image (index 0)

        # ODD_IMAGES (3rd, 5th, 7th) -> Indices 2, 4, 6 (0-based)
        safe_append(odd_images, current_group, 2, 'odd_images') # 3rd image
        safe_append(odd_images, current_group, 4, 'odd_images') # 5th image
        safe_append(odd_images, current_group, 6, 'odd_images') # 7th image

        # EVEN_IMAGES (2nd, 4th, 6th, 8th) -> Indices 1, 3, 5, 7
        safe_append(even_images, current_group, 1, 'even_images') # 2nd image
        safe_append(even_images, current_group, 3, 'even_images') # 4th image
        safe_append(even_images, current_group, 5, 'even_images') # 6th image
        safe_append(even_images, current_group, 7, 'even_images') # 8th image

    return odd_images, even_images

# --- 2. FILE PROCESSING FUNCTIONS (NO CHANGE NEEDED) ---

def process_and_save_images(odd_list, even_list, output_folder):
    """
    Appends the even list to the odd list, saves the images to a new folder
    with frame_### naming, and returns the mapping for the CSV file.
    """
    combined_images = odd_list + even_list
    filename_mapping = []

    os.makedirs(output_folder, exist_ok=True)
    print(f"\nCreated output folder: {output_folder}")

    for index, original_path in enumerate(combined_images):
        frame_number = index + 1
        _, ext = os.path.splitext(original_path)
        new_filename = f"frame_{frame_number:03d}{ext}"
        new_path = os.path.join(output_folder, new_filename)
        original_filename = os.path.basename(original_path)

        try:
            # We use copy2 to preserve metadata like timestamps
            shutil.copy2(original_path, new_path)
            filename_mapping.append((original_filename, new_filename))
        except Exception as e:
            print(f"🔴 Error copying file {original_filename}: {e}")

    return filename_mapping


def save_mapping_to_csv(filename_mapping, csv_path):
    """
    Saves the list of (original_filename, new_filename) tuples to a CSV file.
    """
    try:
        with open(csv_path, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['Original Filename', 'New Filename (frame_###)'])
            csv_writer.writerows(filename_mapping)

        print(f"\n✅ Filename mapping saved successfully to: {csv_path}")
    except Exception as e:
        print(f"🔴 Error saving CSV file: {e}")

# --- EXECUTION ---

# 🛑 NOTE: Keep your configured paths as they were:
INPUT_FOLDER = 'data/mip-nerf360/bicycle/images'
OUTPUT_FOLDER = 'data/mip-nerf360/bicycle/images_sorted_odd_even'
CSV_LOG_PATH = os.path.join(OUTPUT_FOLDER, 'filename_mapping.csv')

# 1. Select and sort the images
odd_list, even_list = select_and_sort_images(INPUT_FOLDER)

# 2. Process, move, rename, and get the mapping
mapping_data = process_and_save_images(odd_list, even_list, OUTPUT_FOLDER)

# 3. Save the mapping to a CSV file
if mapping_data:
    save_mapping_to_csv(mapping_data, CSV_LOG_PATH)
    
    print("\n" + "="*60)
    print(f"Total processed images: {len(mapping_data)}")
    print(f"Odd images (3rd, 5th, 7th): {len(odd_list)}")
    print(f"Even images (2nd, 4th, 6th, 8th): {len(even_list)}")
    print("="*60)
else:
    print("No images were selected or processed, so no CSV file was created.")