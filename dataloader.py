from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd
import wfdb
from PIL import Image


DataPayload = Tuple[object, dict]


class ECGDataLoader:
    """Loader dedicated to ECG data."""

    def __init__(self, data_dir: Union[str, Path]):
        self.data_dir = Path(data_dir)

    def _patient_dir(self, patient_id: str) -> Path:
        return self.data_dir / patient_id

    def _find_record_path(self, patient_dir: Path) -> Optional[Path]:
        hea_files = sorted(patient_dir.glob("*.hea"))
        if not hea_files:
            return None
        return hea_files[0].with_suffix("")

    def _find_ecg_image_path(self, patient_dir: Path) -> Optional[Path]:
        ecg_png_files = sorted(patient_dir.glob("ecg_visualization_*.png"))
        if not ecg_png_files:
            return None
        return ecg_png_files[0]

    def has_data(self, patient_id: str) -> bool:
        patient_dir = self._patient_dir(patient_id)
        return (
            self._find_record_path(patient_dir) is not None
            or self._find_ecg_image_path(patient_dir) is not None
        )

    def get_primary_path(self, patient_id: str) -> Optional[str]:
        patient_dir = self._patient_dir(patient_id)
        record_path = self._find_record_path(patient_dir)
        if record_path is not None:
            return str(record_path.with_suffix(".hea"))
        image_path = self._find_ecg_image_path(patient_dir)
        return str(image_path) if image_path is not None else None

    def load(self, patient_id: str) -> Optional[DataPayload]:
        patient_dir = self._patient_dir(patient_id)

        record_path = self._find_record_path(patient_dir)
        if record_path is not None:
            try:
                signal, meta = wfdb.rdsamp(str(record_path))
                signal = np.asarray(signal, dtype=np.float32).transpose(1, 0)
                metadata = {
                    "modality": "ECG",
                    "format": "wfdb",
                    "shape": signal.shape,
                    "fs": meta.get("fs"),
                    "sig_name": meta.get("sig_name"),
                    "units": meta.get("units"),
                }
                return signal, metadata
            except Exception as exc:
                print(f"Error loading ECG wfdb for {patient_id}: {exc}")

        image_path = self._find_ecg_image_path(patient_dir)
        if image_path is None:
            return None

        try:
            img = Image.open(image_path)
            img_array = np.array(img)

            if img_array.ndim == 3:
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
        except Exception as exc:
            print(f"Error loading ECG for {patient_id}: {exc}")
            return None


class EchoDataLoader:
    """Loader dedicated to echocardiography data."""

    def __init__(self, data_dir: Union[str, Path]):
        self.data_dir = Path(data_dir)
        self.filelist_csv: Optional[pd.DataFrame] = None
        self.volume_tracings_csv: Optional[pd.DataFrame] = None
        self._load_csv_metadata()

    def _patient_dir(self, patient_id: str) -> Path:
        return self.data_dir / patient_id

    def _load_csv_metadata(self) -> None:
        echo_dir = self.data_dir.parent / "Echocardiography"

        filelist_path = echo_dir / "FileList.csv"
        if filelist_path.exists():
            try:
                self.filelist_csv = pd.read_csv(filelist_path)
            except Exception as exc:
                print(f"Error loading FileList.csv: {exc}")

        tracings_path = echo_dir / "VolumeTracings.csv"
        if tracings_path.exists():
            try:
                self.volume_tracings_csv = pd.read_csv(tracings_path)
            except Exception as exc:
                print(f"Error loading VolumeTracings.csv: {exc}")

    def _find_video_path(self, patient_dir: Path) -> Optional[Path]:
        video_files = sorted(patient_dir.glob("*.mp4")) + sorted(patient_dir.glob("*.avi"))
        if not video_files:
            return None
        return video_files[0]

    def has_data(self, patient_id: str) -> bool:
        patient_dir = self._patient_dir(patient_id)
        return self._find_video_path(patient_dir) is not None

    def get_primary_path(self, patient_id: str) -> Optional[str]:
        patient_dir = self._patient_dir(patient_id)
        video_path = self._find_video_path(patient_dir)
        return str(video_path) if video_path is not None else None

    def get_filelist_metadata(self, filename: str) -> Optional[dict]:
        if self.filelist_csv is None:
            return None

        matching_rows = self.filelist_csv[self.filelist_csv["FileName"] == filename]
        if matching_rows.empty:
            return None

        row = matching_rows.iloc[0]
        return {
            "filename": row["FileName"],
            "ef": float(row["EF"]),
            "esv": float(row["ESV"]),
            "edv": float(row["EDV"]),
            "frame_height": int(row["FrameHeight"]),
            "frame_width": int(row["FrameWidth"]),
            "fps": float(row["FPS"]),
            "number_of_frames": int(row["NumberOfFrames"]),
        }

    def get_volume_tracings(self, filename: str) -> Dict[int, List[dict]]:
        if self.volume_tracings_csv is None:
            return {}

        matching_rows = self.volume_tracings_csv[
            self.volume_tracings_csv["FileName"] == filename
        ]
        if matching_rows.empty:
            base_filename = filename.split(".")[0]
            matching_rows = self.volume_tracings_csv[
                self.volume_tracings_csv["FileName"].str.startswith(base_filename)
            ]

        if matching_rows.empty:
            return {}

        tracings_by_frame: Dict[int, List[dict]] = {}
        for _, row in matching_rows.iterrows():
            frame = int(row["Frame"])
            if frame not in tracings_by_frame:
                tracings_by_frame[frame] = []

            tracings_by_frame[frame].append(
                {
                    "x1": float(row["X1"]),
                    "y1": float(row["Y1"]),
                    "x2": float(row["X2"]),
                    "y2": float(row["Y2"]),
                }
            )

        return tracings_by_frame

    def load(self, patient_id: str) -> Optional[DataPayload]:
        patient_dir = self._patient_dir(patient_id)
        video_file = self._find_video_path(patient_dir)
        if video_file is None:
            return None

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
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)

            cap.release()

            filename_stem = video_file.stem
            filename_with_ext = video_file.name

            metadata = {
                "modality": "Echocardiography",
                "format": video_file.suffix,
                "fps": fps,
                "total_frames": total_frames,
                "width": width,
                "height": height,
                "frame_count": len(frames),
            }

            filelist_data = self.get_filelist_metadata(filename_stem)
            if filelist_data:
                metadata["filelist_data"] = filelist_data

            volume_tracings = self.get_volume_tracings(filename_with_ext)
            if volume_tracings:
                metadata["volume_tracings"] = volume_tracings

            return frames, metadata
        except Exception as exc:
            print(f"Error loading Echo for {patient_id}: {exc}")
            return None


class AngioDataLoader:
    """Loader dedicated to angiography data."""

    IMAGE_EXTENSIONS = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff")

    def __init__(self, data_dir: Union[str, Path]):
        self.data_dir = Path(data_dir)

    def _patient_dir(self, patient_id: str) -> Path:
        return self.data_dir / patient_id

    def _find_image_path(self, patient_dir: Path) -> Optional[Path]:
        image_files: List[Path] = []
        for pattern in self.IMAGE_EXTENSIONS:
            image_files.extend(sorted(patient_dir.glob(pattern)))

        image_files = [
            path
            for path in image_files
            if not path.name.lower().startswith("ecg_visualization_")
        ]
        if not image_files:
            return None
        return image_files[0]

    def has_data(self, patient_id: str) -> bool:
        patient_dir = self._patient_dir(patient_id)
        return self._find_image_path(patient_dir) is not None

    def get_primary_path(self, patient_id: str) -> Optional[str]:
        patient_dir = self._patient_dir(patient_id)
        image_path = self._find_image_path(patient_dir)
        return str(image_path) if image_path is not None else None

    def load(self, patient_id: str) -> Optional[DataPayload]:
        patient_dir = self._patient_dir(patient_id)
        image_file = self._find_image_path(patient_dir)
        if image_file is None:
            return None

        try:
            img = Image.open(image_file)
            img_array = np.array(img)
            metadata = {
                "modality": "Cardiac Angiography",
                "format": image_file.suffix.lower().lstrip("."),
                "shape": img_array.shape,
                "mode": img.mode,
                "size": img.size,
            }
            return img_array, metadata
        except Exception as exc:
            print(f"Error loading Angio for {patient_id}: {exc}")
            return None


class PatientDataLoader:
    """
    Combined loader that orchestrates ECG, Echo, and Angio modality loaders.
    Keeps a single app-facing API.
    """

    def __init__(self, data_dir: Union[str, Path]):
        self.data_dir = Path(data_dir)
        self.ecg_loader = ECGDataLoader(self.data_dir)
        self.echo_loader = EchoDataLoader(self.data_dir)
        self.angio_loader = AngioDataLoader(self.data_dir)

        self.modalities = [
            {"id": "ecg", "label": "ECG", "detail": "Electrocardiogram"},
            {"id": "angio", "label": "Cardiac Angiography", "detail": "X-ray imaging"},
            {"id": "echo", "label": "Echocardiography", "detail": "Ultrasound imaging"},
        ]
        self._modality_loaders: Dict[str, Callable[[str], Optional[DataPayload]]] = {
            "ecg": self.ecg_loader.load,
            "angio": self.angio_loader.load,
            "echo": self.echo_loader.load,
        }
        self._modality_availability_checkers: Dict[str, Callable[[str], bool]] = {
            "ecg": self.ecg_loader.has_data,
            "angio": self.angio_loader.has_data,
            "echo": self.echo_loader.has_data,
        }
        self._modality_primary_path_getters: Dict[str, Callable[[str], Optional[str]]] = {
            "ecg": self.ecg_loader.get_primary_path,
            "angio": self.angio_loader.get_primary_path,
            "echo": self.echo_loader.get_primary_path,
        }

    def _normalize_modality_id(self, modality: str) -> str:
        normalized = modality.strip().lower()
        aliases = {
            "electrocardiogram": "ecg",
            "cardiac angiography": "angio",
            "angiography": "angio",
            "echocardiography": "echo",
        }
        return aliases.get(normalized, normalized)

    def list_patients(self) -> List[str]:
        if not self.data_dir.exists():
            return []
        patient_ids = [
            path.name
            for path in self.data_dir.iterdir()
            if path.is_dir() and not path.name.startswith(".")
        ]
        return sorted(patient_ids)

    def get_modalities(self) -> List[Dict[str, str]]:
        return [dict(modality) for modality in self.modalities]

    def register_modality(
        self,
        modality_id: str,
        label: str,
        detail: str,
        loader: Callable[[str], Optional[DataPayload]],
        availability_checker: Optional[Callable[[str], bool]] = None,
        primary_path_getter: Optional[Callable[[str], Optional[str]]] = None,
    ) -> None:
        """Register or replace a modality in the combined loader."""
        normalized = self._normalize_modality_id(modality_id)
        self._modality_loaders[normalized] = loader
        if availability_checker is not None:
            self._modality_availability_checkers[normalized] = availability_checker
        if primary_path_getter is not None:
            self._modality_primary_path_getters[normalized] = primary_path_getter

        for modality in self.modalities:
            if modality["id"] == normalized:
                modality["label"] = label
                modality["detail"] = detail
                return
        self.modalities.append({"id": normalized, "label": label, "detail": detail})

    def available_modalities(self, patient_id: str) -> List[str]:
        available = []
        for modality in self.modalities:
            modality_id = modality["id"]
            checker = self._modality_availability_checkers.get(modality_id)
            if checker is not None:
                if checker(patient_id):
                    available.append(modality_id)
                continue
            if self.load_modality(patient_id, modality_id) is not None:
                available.append(modality_id)
        return available

    def load_modality(self, patient_id: str, modality_id: str) -> Optional[DataPayload]:
        normalized = self._normalize_modality_id(modality_id)
        loader = self._modality_loaders.get(normalized)
        if loader is None:
            return None
        return loader(patient_id)

    def load_ecg(self, patient_id: str) -> Optional[DataPayload]:
        return self.ecg_loader.load(patient_id)

    def load_echo(self, patient_id: str) -> Optional[DataPayload]:
        return self.echo_loader.load(patient_id)

    def load_angio(self, patient_id: str) -> Optional[DataPayload]:
        return self.angio_loader.load(patient_id)

    def load_patient_data(self, patient_id: str) -> dict:
        return {
            "ecg": self.load_modality(patient_id, "ecg"),
            "echo": self.load_modality(patient_id, "echo"),
            "angio": self.load_modality(patient_id, "angio"),
        }

    def get_modality_image(self, patient_id: str, modality: str) -> Optional[str]:
        normalized = self._normalize_modality_id(modality)
        getter = self._modality_primary_path_getters.get(normalized)
        if getter is None:
            return None
        return getter(patient_id)


if __name__ == "__main__":
    loader = PatientDataLoader("./data")
    patient_data = loader.load_patient_data("p001")

    if patient_data["ecg"]:
        ecg_data, ecg_meta = patient_data["ecg"]
        print(f"ECG loaded: {ecg_meta}")

    if patient_data["echo"]:
        echo_frames, echo_meta = patient_data["echo"]
        print(f"Echo loaded: {echo_meta}")

    if patient_data["angio"]:
        angio_data, angio_meta = patient_data["angio"]
        print(f"Angio loaded: {angio_meta}")
