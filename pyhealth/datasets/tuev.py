import logging
import os
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from pyhealth.datasets.sample_dataset import SampleDataset
from pyhealth.processors.base_processor import FeatureProcessor
from pyhealth.processors.signal_processor import SignalProcessor
from pyhealth.tasks.base_task import BaseTask

# from ..tasks import COVID19CXRClassification
from .base_dataset import BaseDataset

logger = logging.getLogger(__name__)


# class TUEVDataset(BaseSignalDataset):
#     """Base EEG dataset for the TUH EEG Events Corpus

#     Dataset is available at https://isip.piconepress.com/projects/tuh_eeg/html/downloads.shtml

#     This corpus is a subset of TUEG that contains annotations of EEG segments as one of six classes: (1) spike and sharp wave (SPSW), (2) generalized periodic epileptiform discharges (GPED), (3) periodic lateralized epileptiform discharges (PLED), (4) eye movement (EYEM), (5) artifact (ARTF) and (6) background (BCKG).

#     Files are named in the form of bckg_032_a_.edf in the eval partition:
#         bckg: this file contains background annotations.
# 		032: a reference to the eval index	
# 		a_.edf: EEG files are split into a series of files starting with a_.edf, a_1.ef, ... These represent pruned EEGs, so the  original EEG is split into these segments, and uninteresting parts of the original recording were deleted.
#     or in the form of 00002275_00000001.edf in the train partition:
#         00002275: a reference to the train index. 
# 		0000001: indicating that this is the first file inssociated with this patient. 

#     Citations:
#     ---------
#     If you use this dataset, please cite:
#     1. Harati, A., Golmohammadi, M., Lopez, S., Obeid, I., & Picone, J. (2015, December). Improved EEG event classification using differential energy. In 2015 IEEE Signal Processing in Medicine and Biology Symposium (SPMB) (pp. 1-4). IEEE.
#     2. Obeid, I., & Picone, J. (2016). The temple university hospital EEG data corpus. Frontiers in neuroscience, 10, 196.
    
#     Args:
#         dataset_name: name of the dataset.
#         root: root directory of the raw data. *You can choose to use the path to Cassette portion or the Telemetry portion.*
#         dev: whether to enable dev mode (only use a small subset of the data).
#             Default is False.
#         refresh_cache: whether to refresh the cache; if true, the dataset will
#             be processed from scratch and the cache will be updated. Default is False.

#     Attributes:
#         task: Optional[str], name of the task (e.g., "EEG_events").
#             Default is None.
#         samples: Optional[List[Dict]], a list of samples, each sample is a dict with
#             patient_id, record_id, and other task-specific attributes as key.
#             Default is None.
#         patient_to_index: Optional[Dict[str, List[int]]], a dict mapping patient_id to
#             a list of sample indices. Default is None.
#         visit_to_index: Optional[Dict[str, List[int]]], a dict mapping visit_id to a
#             list of sample indices. Default is None.

#     Examples:
#         >>> from pyhealth.datasets import TUEVDataset
#         >>> dataset = TUEVDataset(
#         ...         root="/srv/local/data/TUH/tuh_eeg_events/v2.0.0/edf/",
#         ...     )
#         >>> dataset.stat()
#         >>> dataset.info()
#     """

#     def process_EEG_data(self):
#         # get all file names
#         all_files = {}

#         train_files = os.listdir(os.path.join(self.root, "train/"))
#         for id in train_files:
#             if id != ".DS_Store":
#                 all_files["0_{}".format(id)] = [name for name in os.listdir(os.path.join(self.root, "train/", id)) if name.endswith(".edf")]

#         eval_files = os.listdir(os.path.join(self.root, "eval/"))
#         for id in eval_files:
#             if id != ".DS_Store":
#                 all_files["1_{}".format(id)] = [name for name in os.listdir(os.path.join(self.root, "eval/", id)) if name.endswith(".edf")]

#         # get all patient ids
#         patient_ids = list(set(list(all_files.keys())))

#         if self.dev:
#             patient_ids = patient_ids[:20]
#             # print(patient_ids)

#         # get patient to record maps
#         #    - key: pid:
#         #    - value: [{"load_from_path": None, "patient_id": None, "signal_file": None, "label_file": None, "save_to_path": None}, ...]
#         patients = {
#             pid: []
#             for pid in patient_ids
#         }
           
#         for pid in patient_ids:
#             split = "train" if pid.split("_")[0] == "0" else "eval"
#             id = pid.split("_")[1]

#             patient_visits = all_files[pid]
            
#             for visit in patient_visits:
#                 if split == "train":
#                     visit_id = visit.strip(".edf").split("_")[1]
#                 else:
#                     visit_id = visit.strip(".edf")
                    
#                 patients[pid].append({
#                     "load_from_path": os.path.join(self.root, split, id),
#                     "patient_id": pid,
#                     "visit_id": visit_id,
#                     "signal_file": visit,
#                     "label_file": visit,
#                     "save_to_path": self.filepath,
#                 })
        
#         return patients


class TUEVDataset(BaseDataset):
    """Base EEG dataset for the TUH EEG Events Corpus (TUEV).

    Dataset is available at https://isip.piconepress.com/projects/tuh_eeg/html/downloads.shtml

    This corpus is a subset of TUEG that contains annotations of EEG segments
    across six classes:
      (1) SPSW - spike and sharp wave
      (2) GPED - generalized periodic epileptiform discharges
      (3) PLED - periodic lateralized epileptiform discharges
      (4) EYEM - eye movement
      (5) ARTF - artifact
      (6) BCKG - background
      
      
    Files are named in the form of bckg_032_a_.edf in the eval partition:
        bckg: this file contains background annotations.
		032: a reference to the eval index	
		a_.edf: EEG files are split into a series of files starting with a_.edf, a_1.ef, ... These represent pruned EEGs, so the  original EEG is split into these segments, and uninteresting parts of the original recording were deleted.
    or in the form of 00002275_00000001.edf in the train partition:
        00002275: a reference to the train index. 
		0000001: indicating that this is the first file inssociated with this patient. 

    Citations:
    ---------
    If you use this dataset, please cite:
    1. Harati, A., Golmohammadi, M., Lopez, S., Obeid, I., & Picone, J. (2015, December). Improved EEG event classification using differential energy. In 2015 IEEE Signal Processing in Medicine and Biology Symposium (SPMB) (pp. 1-4). IEEE.
    2. Obeid, I., & Picone, J. (2016). The temple university hospital EEG data corpus. Frontiers in neuroscience, 10, 196.
    
    Args:
        root: Root directory of the raw data containing the dataset files.
        dataset_name: Optional name of the dataset. Defaults to "tuev".
        config_path: Optional path to the configuration file. If not provided,
            uses the default config in the configs directory.

    Attributes:
        root: Root directory of the raw data.
        dataset_name: Name of the dataset.
        config_path: Path to the configuration file.

    Examples:
        >>> from pyhealth.datasets import TUEVDataset
        >>> dataset = TUEVDataset(
        ...     root="/srv/local/data/TUH/tuh_eeg_events/v2.0.0/edf"
        ... )
        >>> dataset.stats()
        >>> samples = dataset.set_task()
        >>> print(samples[0])
    """
    def __init__(
        self,
        root: str,
        dataset_name: Optional[str] = None,
        config_path: Optional[str] = None,
        dev: bool = False,
        ) -> None:
        if config_path is None:
            logger.info("No config path provided, using default config")
            config_path = Path(__file__).parent / "configs" / "tuev.yaml"
        if not os.path.exists(os.path.join(root, "tuev-metadata-pyhealth.csv")):
            self.prepare_metadata(root)
        default_tables = ["tuev"]
        super().__init__(
                root=root,
                tables=default_tables,
                dataset_name=dataset_name or "tuev",
                config_path=config_path,
            )
        return
    
    def prepare_metadata(self, root: str) -> None:
        """Prepare metadata for the dataset.
        
        Args:
            root: Root directory of the raw data.
            
        This methods:
        1. Scans the directory structure under `root`, collecting all `.edf` files from
        the `train/` and `eval/` splits and writes a single CSV:
            tuev-metadata-pyhealth.csv
        2. The CSV has the following columns:
            - path: absolute path to the .edf file
            - split: 'train' or 'eval'
            - patient_id: derived from folder name (prefixed with '0_' for train and '1_' for eval)
            - recording_id: derived from filename (format differs slightly by split)
            - signal_file: filename of the .edf
            - label_file: filename of the annotation carrier (here same as signal_file)

        """
        logger.info("Preparing metadata for the dataset")
        
        rows = []

        def add_rows_for_split(split_name: str, pid_prefix: str) -> None:
            split_dir = os.path.join(root, split_name)
            if not os.path.isdir(split_dir):
                logger.warning("Split directory not found: %s", split_dir)
                return

            for pid in os.listdir(split_dir):
                if pid.startswith("."):  # skip hidden/system entries (e.g., .DS_Store)
                    continue
                pid_path = os.path.join(split_dir, pid)
                if not os.path.isdir(pid_path):
                    continue

                edf_files = [f for f in os.listdir(pid_path) if f.lower().endswith(".edf")]
                full_pid = f"{pid_prefix}_{pid}"

                for fname in edf_files:
                    fpath = os.path.abspath(os.path.join(pid_path, fname))
                    stem = os.path.splitext(fname)[0]

                    if split_name == "train":
                        # e.g., 00002275_00000001.edf -> recording_id = second token
                        parts = stem.split("_")
                        recording_id = parts[1] if len(parts) > 1 else stem
                    else:
                        # e.g., bckg_032_a_.edf -> use full stem
                        recording_id = stem

                    rows.append(
                        {
                            "path": fpath,
                            "split": split_name,
                            "patient_id": full_pid,
                            "recording_id": recording_id,
                            "signal_file": fname,
                            "label_file": fname,  # placeholder: same as signal_file
                        }
                    )

        add_rows_for_split("train", "0")
        add_rows_for_split("eval", "1")

        if not rows:
            raise FileNotFoundError(
                f"No EDF files found under {root!r}. Expected subfolders 'train/' and/or 'eval/'."
            )

        df = pd.DataFrame(rows)

        # Sanity check: ensure files exist
        missing = [p for p in df["path"] if not os.path.isfile(p)]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} files listed but not found on disk. Example: {missing[0]}"
            )

        out_csv = os.path.join(root, "tuev-metadata-pyhealth.csv")
        df.to_csv(out_csv, index=False)
        logger.info("Wrote TUEV metadata to %s (%d rows)", out_csv, len(df))




if __name__ == "__main__":
    dataset = TUEVDataset(
        root="/srv/local/data/TUH/tuh_eeg_events/v2.0.0/edf",
        dev=True,
        refresh_cache=True,
    )
    dataset.stat()
    dataset.info()
    print(list(dataset.patients.items())[0])
