#!/usr/bin/env python3
"""
Utility script to convert all patient ECG data to PNG/JPG images.
Saves images in patient directories for easy access and visualization.
"""

from pathlib import Path
from dataloader import PatientDataLoader
from visualizer import PatientVisualizer


def save_all_ecg_images(data_dir: str = "./data", output_format: str = "png") -> dict:
    """
    Convert all patient ECG data to image files.
    
    Args:
        data_dir: Path to data directory
        output_format: 'png' or 'jpg'
    
    Returns:
        Dictionary mapping patient_id to result status
    """
    data_loader = PatientDataLoader(data_dir)
    visualizer = PatientVisualizer()
    
    print(f"\n{'='*70}")
    print(f"Converting ECG data to {output_format.upper()} images")
    print(f"{'='*70}\n")
    
    results = {}
    data_path = Path(data_dir)
    
    # Get all patient directories
    patient_dirs = sorted([d for d in data_path.iterdir() if d.is_dir() and d.name.startswith('p')])
    
    for patient_dir in patient_dirs:
        patient_id = patient_dir.name
        print(f"Processing {patient_id}...", end=" ")
        
        try:
            # Load ECG data
            ecg_data = data_loader.load_ecg(patient_id)
            if ecg_data is None:
                print("❌ ECG data not found")
                results[patient_id] = "not_found"
                continue
            
            data, metadata = ecg_data
            
            # Save as image
            output_filename = f"ecg.{output_format.lower()}"
            output_path = patient_dir / output_filename
            
            visualizer.save_ecg_as_image(
                data,
                str(output_path),
                title=f"{patient_id} - 12-lead ECG",
                format=output_format
            )
            
            results[patient_id] = "success"
            print(f"✓ Saved to {output_filename}")
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            results[patient_id] = f"error: {str(e)}"
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    success_count = sum(1 for v in results.values() if v == "success")
    print(f"✓ Successful: {success_count}/{len(results)}")
    
    for patient_id, status in results.items():
        if status != "success":
            print(f"⚠ {patient_id}: {status}")
    
    print(f"\nAll ECG images are now in patient directories as 'ecg.{output_format.lower()}'")
    return results


if __name__ == "__main__":
    data_dir = Path(__file__).resolve().parent / "data"
    save_all_ecg_images(str(data_dir), output_format="png")
