#!/usr/bin/env python3
"""
Simple visualization script for a single patient's medical imaging data.
Displays ECG, Angiography, and Echocardiography frames.
"""

from pathlib import Path
import sys
from dataloader import PatientDataLoader
from visualizer import PatientVisualizer
import matplotlib.pyplot as plt
from PIL import Image


def visualize_patient_ecg(data_loader: PatientDataLoader, patient_id: str):
    """Display ECG for a patient."""
    print(f"\n{'='*60}")
    print(f"Loading ECG for {patient_id}...")
    print(f"{'='*60}")
    
    ecg_data = data_loader.load_ecg(patient_id)
    if ecg_data is None:
        print(f"❌ ECG data not found for {patient_id}")
        return
    
    data, metadata = ecg_data
    print(f"✓ ECG loaded successfully")
    print(f"  Shape: {metadata['shape']}")
    print(f"  Channels: {metadata['channels']}")
    
    visualizer = PatientVisualizer()
    temp_file = visualizer.plot_ecg(data)
    visualizer.display_image(temp_file)
    visualizer.cleanup_temp_files()


def visualize_patient_angio(data_loader: PatientDataLoader, patient_id: str):
    """Display Angiography for a patient."""
    print(f"\n{'='*60}")
    print(f"Loading Angiography for {patient_id}...")
    print(f"{'='*60}")
    
    angio_data = data_loader.load_angio(patient_id)
    if angio_data is None:
        print(f"❌ Angiography data not found for {patient_id}")
        return
    
    data, metadata = angio_data
    print(f"✓ Angiography loaded successfully")
    print(f"  Size: {metadata['size']}")
    print(f"  Format: {metadata['format']}")
    
    visualizer = PatientVisualizer()
    visualizer.display_image(visualizer.frame_to_temp_file(data))
    visualizer.cleanup_temp_files()


def visualize_patient_echo(data_loader: PatientDataLoader, patient_id: str, frames_to_show: int = 5):
    """Display Echocardiography frames with tracings."""
    print(f"\n{'='*60}")
    print(f"Loading Echocardiography for {patient_id}...")
    print(f"{'='*60}")
    
    echo_data = data_loader.load_echo(patient_id)
    if echo_data is None:
        print(f"❌ Echocardiography data not found for {patient_id}")
        return
    
    frames, metadata = echo_data
    print(f"✓ Echo loaded successfully")
    print(f"  Total frames: {len(frames)}")
    print(f"  FPS: {metadata.get('fps', 'N/A')}")
    
    # Display FileList metadata if available
    if 'filelist_data' in metadata:
        filelist = metadata['filelist_data']
        print(f"  Cardiac metrics:")
        print(f"    - EF (Ejection Fraction): {filelist['ef']:.2f}%")
        print(f"    - ESV (End Systolic Volume): {filelist['esv']:.2f}")
        print(f"    - EDV (End Diastolic Volume): {filelist['edv']:.2f}")
    
    # Display tracing info
    if 'volume_tracings' in metadata:
        tracings = metadata['volume_tracings']
        print(f"  Volume tracings: {len(tracings)} frames marked")
    
    # Show sample frames
    visualizer = PatientVisualizer()
    n_frames = min(frames_to_show, len(frames))
    indices = [int(i * len(frames) / n_frames) for i in range(n_frames)]
    
    fig, axes = plt.subplots(1, n_frames, figsize=(15, 5))
    if n_frames == 1:
        axes = [axes]
    
    for idx, frame_idx in enumerate(indices):
        frame = frames[frame_idx]
        
        # Draw tracings if available
        if 'volume_tracings' in metadata and frame_idx in metadata['volume_tracings']:
            frame = visualizer.draw_tracings_on_frame(frame, metadata['volume_tracings'][frame_idx])
        
        axes[idx].imshow(frame)
        axes[idx].set_title(f"Frame {frame_idx}/{len(frames)-1}")
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.show()
    visualizer.cleanup_temp_files()


def main():
    """Main visualization function."""
    # Configuration
    data_dir = Path(__file__).parent / "data"
    patient_id = "p001"  # Change this to visualize different patient
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        patient_id = sys.argv[1]
    
    print(f"\n🏥 Patient Imaging Visualizer")
    print(f"Patient ID: {patient_id}")
    print(f"Data directory: {data_dir}")
    
    # Initialize data loader
    try:
        data_loader = PatientDataLoader(str(data_dir))
    except Exception as e:
        print(f"❌ Error initializing data loader: {e}")
        return
    
    # Visualize each modality
    try:
        visualize_patient_ecg(data_loader, patient_id)
        visualize_patient_angio(data_loader, patient_id)
        visualize_patient_echo(data_loader, patient_id, frames_to_show=5)
    except Exception as e:
        print(f"❌ Error during visualization: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
