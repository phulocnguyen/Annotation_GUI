from pathlib import Path
from typing import Optional, Union, Tuple, Dict, List
import numpy as np
import pandas as pd
from PIL import Image
import cv2
from scipy import io as scipy_io


class PatientDataLoader:
    """
    DataLoader for patient medical imaging data.
    Supports ECG (.mat), Echocardiography (.avi, .mp4), and Angiography (.png).
    """

    def __init__(self, data_dir: Union[str, Path]):
        """
        Initialize the dataloader with a data directory.

        Args:
            data_dir: Path to the directory containing patient data.
                     Expected structure:
                     data_dir/
                     ├── p001/
                     │   ├── ecg.mat
                     │   ├── echo.avi (or echo.mp4)
                     │   └── angio.png
                     └── p002/
        """
        self.data_dir = Path(data_dir)
        self.filelist_csv = None
        self.volume_tracings_csv = None
        self._load_csv_metadata()

    def _load_csv_metadata(self):
        """Load CSV metadata files from Echocardiography directory."""
        echo_dir = self.data_dir.parent / "Echocardiography"
        
        # Load FileList.csv
        filelist_path = echo_dir / "FileList.csv"
        if filelist_path.exists():
            try:
                self.filelist_csv = pd.read_csv(filelist_path)
            except Exception as e:
                print(f"Error loading FileList.csv: {e}")
        
        # Load VolumeTracings.csv
        tracings_path = echo_dir / "VolumeTracings.csv"
        if tracings_path.exists():
            try:
                self.volume_tracings_csv = pd.read_csv(tracings_path)
            except Exception as e:
                print(f"Error loading VolumeTracings.csv: {e}")

    def get_filelist_metadata(self, filename: str) -> Optional[Dict]:
        """
        Get metadata for a specific echo file from FileList.csv.
        
        Args:
            filename: Echo file name (e.g., '0X100009310A3BD7FC')
        
        Returns:
            Dictionary with metadata (EF, ESV, EDV, etc.) or None
        """
        if self.filelist_csv is None:
            return None
        
        matching_rows = self.filelist_csv[self.filelist_csv['FileName'] == filename]
        if matching_rows.empty:
            return None
        
        row = matching_rows.iloc[0]
        return {
            'filename': row['FileName'],
            'ef': float(row['EF']),
            'esv': float(row['ESV']),
            'edv': float(row['EDV']),
            'frame_height': int(row['FrameHeight']),
            'frame_width': int(row['FrameWidth']),
            'fps': float(row['FPS']),
            'number_of_frames': int(row['NumberOfFrames']),
        }

    def get_volume_tracings(self, filename: str) -> Dict[int, List[Dict]]:
        """
        Get volume tracing points for a specific echo file.
        
        Args:
            filename: Echo file name with extension (e.g., '0X100009310A3BD7FC.avi')
        
        Returns:
            Dictionary mapping frame number to list of tracing points
            Example: {46: [{'x1': 51.26, 'y1': 15.34, 'x2': 64.93, 'y2': 69.12}, ...], ...}
        """
        if self.volume_tracings_csv is None:
            return {}
        
        # Match by filename (with .avi extension if needed)
        matching_rows = self.volume_tracings_csv[self.volume_tracings_csv['FileName'] == filename]
        if matching_rows.empty:
            # Try without extension
            base_filename = filename.split('.')[0]
            matching_rows = self.volume_tracings_csv[
                self.volume_tracings_csv['FileName'].str.startswith(base_filename)
            ]
        
        if matching_rows.empty:
            return {}
        
        # Group by frame number
        tracings_by_frame = {}
        for _, row in matching_rows.iterrows():
            frame = int(row['Frame'])
            if frame not in tracings_by_frame:
                tracings_by_frame[frame] = []
            
            tracings_by_frame[frame].append({
                'x1': float(row['X1']),
                'y1': float(row['Y1']),
                'x2': float(row['X2']),
                'y2': float(row['Y2']),
            })
        
        return tracings_by_frame
        self.filelist_csv = None
        self.volume_tracings_csv = None
        self._load_csv_metadata()

    def load_ecg(self, patient_id: str) -> Optional[Tuple[np.ndarray, dict]]:
        """
        Load ECG data from PNG image file (ecg_visualization_X.png).

        Args:
            patient_id: Patient identifier (e.g., 'p001')

        Returns:
            Tuple of (image array, metadata dict) or None if file not found
        """
        patient_dir = self.data_dir / patient_id
        
        # Find ecg_visualization_*.png file in the patient directory
        ecg_png_files = list(patient_dir.glob("ecg_visualization_*.png"))
        if not ecg_png_files:
            return None
        
        png_file = ecg_png_files[0]

        try:
            img = Image.open(png_file)
            img_array = np.array(img)

            # Get image dimensions
            if len(img_array.shape) == 3:
                height, width, channels = img_array.shape
            else:
                height, width = img_array.shape
                channels = 1

            metadata = {
                "modality": "ECG",
                "format": "png",
                "shape": img_array.shape,
                "size": img.size,
                "mode": img.mode,
                "dimensions": {"width": width, "height": height, "channels": channels},
            }
            return img_array, metadata
        except Exception as e:
            print(f"Error loading ECG for {patient_id}: {e}")
            return None

    def load_echo(self, patient_id: str) -> Optional[Tuple[list, dict]]:
        """
        Load Echocardiography video from .avi or .mp4 file with metadata and tracings.

        Args:
            patient_id: Patient identifier (e.g., 'p001')

        Returns:
            Tuple of (list of frames, metadata dict) or None if file not found
            
            metadata includes:
                - modality, format, fps, total_frames, width, height, frame_count
                - filelist_data: EF, ESV, EDV, frame_height, frame_width, number_of_frames, split
                - volume_tracings: Dictionary mapping frame -> list of tracing points
        """
        patient_dir = self.data_dir / patient_id

        # Find any .mp4 or .avi file in the patient directory
        video_files = list(patient_dir.glob("*.mp4")) + list(patient_dir.glob("*.avi"))
        if not video_files:
            return None
        
        video_file = video_files[0]

        try:
            cap = cv2.VideoCapture(str(video_file))
            frames = []
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                # Convert BGR to RGB for consistency
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)

            cap.release()

            # Get filename without extension for CSV lookup
            filename_stem = video_file.stem  # e.g., '0X100009310A3BD7FC'
            filename_with_ext = video_file.name  # e.g., '0X100009310A3BD7FC.avi'

            metadata = {
                "modality": "Echocardiography",
                "format": video_file.suffix,
                "fps": fps,
                "total_frames": total_frames,
                "width": width,
                "height": height,
                "frame_count": len(frames),
            }
            
            # Load filelist metadata
            filelist_data = self.get_filelist_metadata(filename_stem)
            if filelist_data:
                metadata["filelist_data"] = filelist_data
            
            # Load volume tracings
            volume_tracings = self.get_volume_tracings(filename_with_ext)
            if volume_tracings:
                metadata["volume_tracings"] = volume_tracings
            
            return frames, metadata
        except Exception as e:
            print(f"Error loading Echo for {patient_id}: {e}")
            return None

    def load_angio(self, patient_id: str) -> Optional[Tuple[np.ndarray, dict]]:
        """
        Load Angiography image from .png file.

        Args:
            patient_id: Patient identifier (e.g., 'p001')

        Returns:
            Tuple of (image array, metadata dict) or None if file not found
        """
        patient_dir = self.data_dir / patient_id
        
        # Find any .png file in the patient directory
        png_files = list(patient_dir.glob("*.png"))
        if not png_files:
            return None
        
        image_file = png_files[0]

        try:
            img = Image.open(image_file)
            img_array = np.array(img)

            metadata = {
                "modality": "Cardiac Angiography",
                "format": "png",
                "shape": img_array.shape,
                "mode": img.mode,
                "size": img.size,
            }
            return img_array, metadata
        except Exception as e:
            print(f"Error loading Angio for {patient_id}: {e}")
            return None

    def load_patient_data(
        self, patient_id: str
    ) -> dict:
        """
        Load all available data for a patient.

        Args:
            patient_id: Patient identifier (e.g., 'p001')

        Returns:
            Dictionary with keys: 'ecg', 'echo', 'angio', each containing
            (data, metadata) tuple or None if not available
        """
        return {
            "ecg": self.load_ecg(patient_id),
            "echo": self.load_echo(patient_id),
            "angio": self.load_angio(patient_id),
        }

    def get_modality_image(self, patient_id: str, modality: str) -> Optional[str]:
        """
        Get the file path for a specific modality's primary visualization.
        Useful for displaying in the PyQt6 viewer.

        Args:
            patient_id: Patient identifier
            modality: One of 'ECG', 'Echocardiography', 'Cardiac Angiography'

        Returns:
            Path to the file as string, or None if not found
        """
        patient_dir = self.data_dir / patient_id

        if modality == "ECG":
            mat_files = list(patient_dir.glob("*.mat"))
            return str(mat_files[0]) if mat_files else None
        elif modality == "Echocardiography":
            video_files = list(patient_dir.glob("*.mp4")) + list(patient_dir.glob("*.avi"))
            return str(video_files[0]) if video_files else None
        elif modality == "Cardiac Angiography":
            png_files = list(patient_dir.glob("*.png"))
            return str(png_files[0]) if png_files else None
        
        return None


# Example usage
if __name__ == "__main__":
    # Create a dataloader for a data directory
    loader = PatientDataLoader("./data")

    # Load all data for a patient
    patient_data = loader.load_patient_data("p001")

    # Access specific modalities
    if patient_data["ecg"]:
        ecg_data, ecg_meta = patient_data["ecg"]
        print(f"ECG loaded: {ecg_meta}")

    if patient_data["echo"]:
        echo_frames, echo_meta = patient_data["echo"]
        print(f"Echo loaded: {echo_meta}")

    if patient_data["angio"]:
        angio_data, angio_meta = patient_data["angio"]
        print(f"Angio loaded: {angio_meta}")
