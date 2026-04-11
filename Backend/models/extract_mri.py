import os
import pydicom
import numpy as np
import cv2

# ==========================================
# 1. PATH & EXTRACTION CONFIGURATION
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, "..", "datasets", "raw_3d_mri")
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "datasets", "mri_slices")

CATEGORIES = ["healthy", "parkinson"]

# We extract the middle 15 slices to capture the core brain structures (Axial view)
SLICES_TO_EXTRACT = 15 

def process_mri_data():
    """
    Automates the conversion of medical DICOM volumes into standardized PNG images.
    """
    for category in CATEGORIES:
        source_folder = os.path.join(RAW_DIR, category)
        target_folder = os.path.join(OUTPUT_DIR, category)
        os.makedirs(target_folder, exist_ok=True)
        
        if not os.path.exists(source_folder):
            print(f"[ERROR] Source not found: {source_folder}")
            continue
            
        patient_folders = os.listdir(source_folder)
        
        for patient_id in patient_folders:
            patient_path = os.path.join(source_folder, patient_id)
            if not os.path.isdir(patient_path): continue
                
            print(f"[INFO] Processing {category} subject: {patient_id}")
            
            # 2. SEARCH: Locate all .dcm files within nested directory structures
            dcm_files = []
            for root, dirs, files in os.walk(patient_path):
                for file in files:
                    if file.lower().endswith('.dcm'):
                        dcm_files.append(os.path.join(root, file))
                        
            if not dcm_files:
                print(f"  -> No DICOM data found for {patient_id}.")
                continue
                
            # 3. UNPACK: Extract raw pixel arrays from DICOM metadata
            patient_slices = []
            for f in dcm_files:
                try:
                    ds = pydicom.dcmread(f)
                    if hasattr(ds, 'pixel_array'):
                        arr = ds.pixel_array
                        # Handles multi-frame (3D) and single-frame (2D) DICOM formats
                        if arr.ndim == 3:
                            for i in range(arr.shape[0]):
                                patient_slices.append((i, arr[i]))
                        elif arr.ndim == 2:
                            inst_num = int(ds.InstanceNumber) if hasattr(ds, 'InstanceNumber') else 0
                            patient_slices.append((inst_num, arr))
                except Exception: pass
                    
            # 4. SORT & SLICE: Isolate the middle section of the brain volume
            if len(patient_slices) > 0:
                patient_slices.sort(key=lambda x: x[0]) # Sort slices in anatomical order
                patient_slices = [x[1] for x in patient_slices]
                
            total_slices = len(patient_slices)
            if total_slices == 0: continue
                
            # Calculate indices to centered extraction
            start_idx = max(0, total_slices // 2 - (SLICES_TO_EXTRACT // 2))
            end_idx = min(total_slices, start_idx + SLICES_TO_EXTRACT)
            middle_slices = patient_slices[start_idx:end_idx]
            
            # 5. NORMALIZE & EXPORT: Standardize images for CNN consumption
            for i, img in enumerate(middle_slices):
                try:
                    img = img.astype("float32")
                    
                    # MIN-MAX NORMALIZATION: Ensures consistent brightness across different MRI machines
                    img_min, img_max = np.min(img), np.max(img)
                    if img_max != img_min:
                        img = (img - img_min) / (img_max - img_min)
                    
                    # Convert to 8-bit grayscale for standard image viewing
                    img = (img * 255).astype(np.uint8)
                    
                    # Standardize resolution to 128x128 to match CNN input layer
                    img = cv2.resize(img, (128, 128))
                    
                    save_path = os.path.join(target_folder, f"{patient_id}_slice_{i}.png")
                    cv2.imwrite(save_path, img)
                except Exception as e:
                    print(f"  -> Export Error: {e}")
                
    print("\n[SUCCESS] Dataset preprocessing complete.")

if __name__ == "__main__":
    process_mri_data()