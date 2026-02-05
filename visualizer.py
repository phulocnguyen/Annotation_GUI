"""
Core visualization module for medical imaging data.
Handles frame display, tracing overlay, and image processing.
"""

import tempfile
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt


class PatientVisualizer:
    """Handles visualization of patient medical imaging data."""
    
    def __init__(self):
        self.temp_files = []
    
    def cleanup_temp_files(self):
        """Clean up temporary files created during visualization."""
        import os
        for temp_file in self.temp_files:
            try:
                if Path(temp_file).exists():
                    os.remove(temp_file)
            except Exception as e:
                print(f"Error removing temp file {temp_file}: {e}")
        self.temp_files.clear()
    
    @staticmethod
    def draw_tracings_on_frame(frame: np.ndarray, tracings: list) -> np.ndarray:
        """
        Draw volume tracing contours on a frame.
        
        Args:
            frame: Frame array (H, W, 3)
            tracings: List of tracing points [{x1, y1, x2, y2}, ...]
        
        Returns:
            Frame with tracings drawn
        """
        frame = frame.copy()
        
        # Draw each tracing point
        for tracing in tracings:
            x1 = int(tracing['x1'])
            y1 = int(tracing['y1'])
            x2 = int(tracing['x2'])
            y2 = int(tracing['y2'])
            
            # Ensure points are within frame bounds
            h, w = frame.shape[:2]
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(0, min(x2, w - 1))
            y2 = max(0, min(y2, h - 1))
            
            # Draw a small circle at each point
            cv2.circle(frame, (x1, y1), 2, (0, 255, 0), -1)  # Green
            cv2.circle(frame, (x2, y2), 2, (255, 0, 0), -1)  # Blue
            
            # Draw line connecting the points
            cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 0), 1)  # Yellow
        
        return frame
    
    def frame_to_pil_image(self, frame: np.ndarray) -> Image.Image:
        """Convert frame array to PIL Image, handling various formats and data types."""
        # First, normalize data types
        if frame.dtype != np.uint8:
            if frame.dtype == np.float32 or frame.dtype == np.float64:
                # Float: assume 0-1 range
                if frame.max() > 1.0:
                    frame = (frame / frame.max() * 255).astype(np.uint8)
                else:
                    frame = (frame * 255).astype(np.uint8)
            elif frame.dtype == np.uint16:
                # uint16: scale to 0-255
                frame = (frame / 256).astype(np.uint8)
            else:
                # Other int types
                frame = frame.astype(np.uint8)
        
        # Handle different image formats
        if len(frame.shape) == 2:
            # Grayscale image (H, W)
            return Image.fromarray(frame, mode='L')
        elif len(frame.shape) == 3:
            if frame.shape[2] == 3:
                # RGB image (H, W, 3)
                return Image.fromarray(frame, mode='RGB')
            elif frame.shape[2] == 4:
                # RGBA image (H, W, 4)
                return Image.fromarray(frame, mode='RGBA')
            else:
                # Unknown format, try default
                return Image.fromarray(frame)
        else:
            # Fallback
            return Image.fromarray(frame)
    
    def frame_to_temp_file(self, frame: np.ndarray) -> str:
        """Convert frame to temporary file and return path. Handles various image formats."""
        try:
            # Handle different dtypes (uint8, uint16, float32, etc.)
            if frame.dtype != np.uint8:
                # Normalize to 0-255 range if needed
                if frame.dtype == np.float32 or frame.dtype == np.float64:
                    frame = (frame * 255).astype(np.uint8) if frame.max() <= 1.0 else frame.astype(np.uint8)
                else:
                    # For integer types, just convert
                    frame = frame.astype(np.uint8)
            
            img = self.frame_to_pil_image(frame)
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                img.save(tmp.name)
                self.temp_files.append(tmp.name)
                return tmp.name
        except Exception as e:
            print(f"Error converting frame to temp file: {e}")
            raise
    
    @staticmethod
    def plot_ecg(ecg_data: np.ndarray, title: str = "ECG Signal") -> str:
        """
        Create and save ECG plot to temporary file.
        
        Args:
            ecg_data: ECG signal array
            title: Plot title
        
        Returns:
            Path to temporary PNG file
        """
        fig, ax = plt.subplots(figsize=(10, 4), dpi=100)
        
        if len(ecg_data.shape) > 1:
            # Multi-channel, plot first channel
            ax.plot(ecg_data[:, 0], linewidth=0.8, color='#1f77b4')
        else:
            ax.plot(ecg_data, linewidth=0.8, color='#1f77b4')
        
        ax.set_title(f"{title} (Shape: {ecg_data.shape})", fontsize=12, fontweight='bold')
        ax.set_xlabel("Sample", fontsize=10)
        ax.set_ylabel("Amplitude", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Save with high quality
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            fig.savefig(tmp.name, bbox_inches='tight', dpi=150)
            plt.close(fig)
            return tmp.name
    
    @staticmethod
    def display_image(image_path: str):
        """Display image using matplotlib."""
        img = Image.open(image_path)
        plt.figure(figsize=(10, 7))
        plt.imshow(img)
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def save_frame_with_tracings(frame: np.ndarray, tracings: list, output_path: str):
        """Save frame with tracings overlay to file."""
        frame_with_tracings = PatientVisualizer.draw_tracings_on_frame(frame, tracings)
        img = Image.fromarray(frame_with_tracings)
        img.save(output_path)
        print(f"Saved frame with tracings to {output_path}")
    
    @staticmethod
    def save_ecg_as_image(ecg_data: np.ndarray, output_path: str, title: str = "ECG Signal", format: str = "png") -> str:
        """
        Save ECG data as PNG or JPG image file.
        
        Args:
            ecg_data: ECG signal array
            output_path: Path to save the image (should include filename)
            title: Plot title
            format: Image format ('png' or 'jpg')
        
        Returns:
            Path to the saved image file
        """
        from pathlib import Path
        
        # Ensure output directory exists
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 5), dpi=100)
        
        if len(ecg_data.shape) > 1:
            # Multi-channel, plot first channel
            ax.plot(ecg_data[:, 0], linewidth=0.8, color='#1f77b4')
        else:
            ax.plot(ecg_data, linewidth=0.8, color='#1f77b4')
        
        ax.set_title(f"{title}", fontsize=14, fontweight='bold')
        ax.set_xlabel("Sample", fontsize=11)
        ax.set_ylabel("Amplitude", fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Save with high quality
        fig.savefig(str(output_file), bbox_inches='tight', dpi=150, format=format.lower())
        plt.close(fig)
        print(f"✓ ECG saved to {output_file}")
        return str(output_file)
