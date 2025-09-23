
from pathlib import Path
import json
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, List, Callable
import torch
from torch.utils.data import Dataset
from .schema import RunMeta

def query_index(root: str, where: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None) -> pd.DataFrame:
    index_path = Path(root) / "index.csv"
    if not index_path.exists():
        raise FileNotFoundError(f"index.csv not found in {root}")
    df = pd.read_csv(index_path)
    if where is not None:
        df = where(df)
    return df

class LatticeRun:
    def __init__(self, run_dir: str):
        self.run_dir = Path(run_dir)
        self.meta = json.loads((self.run_dir / "metadata.json").read_text())
        self.correlators = np.load(self.run_dir / "correlators.npz")
        self.observables = None
        if (self.run_dir / "observables.json").exists():
            self.observables = json.loads((self.run_dir / "observables.json").read_text())
        self.fourpoint = None
        if (self.run_dir / "fourpoint.npz").exists():
            self.fourpoint = np.load(self.run_dir / "fourpoint.npz")
        self.configs = None
        if (self.run_dir / "configs.npz").exists():
            self.configs = np.load(self.run_dir / "configs.npz")["configs"]

class LatticeDataset(Dataset):
    """PyTorch dataset for correlator-centric training.
    By default returns (input_dict, target_dict) suitable for denoising/super-res/self-supervised tasks.
    """
    def __init__(self, root: str,
                 select: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
                 input_keys: List[str] = ["C_t"],
                 target_keys: Optional[List[str]] = None,
                 transform: Optional[Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]] = None):
        self.root = Path(root)
        self.df = query_index(root, where=select)
        self.input_keys = input_keys
        self.target_keys = target_keys or input_keys
        self.transform = transform

        self.run_paths = [self.root / p for p in self.df["relpath"].tolist()]

    def __len__(self):
        return len(self.run_paths)

    def __getitem__(self, idx):
        run = LatticeRun(str(self.run_paths[idx]))
        # Pack inputs
        x = {}
        for k in self.input_keys:
            if k not in run.correlators:
                raise KeyError(f"Correlator key '{k}' not found in run {self.run_paths[idx]}")
            x[k] = torch.tensor(run.correlators[k].astype('float32'))
        # Add metadata channels if needed
        meta = run.meta
        meta_vec = torch.tensor([
            float(meta.get("g2") or 0.0),
            float(meta.get("g4") or 0.0),
            float(meta.get("beta") or 0.0),
            float(meta["Lx"]), float(meta["Lt"]),
            float(meta.get("a") or 0.0),
        ], dtype=torch.float32)
        x["meta"] = meta_vec

        # Targets
        y = {}
        for k in self.target_keys:
            y[k] = torch.tensor(run.correlators[k].astype('float32'))

        sample = {"inputs": x, "targets": y, "meta": meta, "observables": run.observables}
        if self.transform is not None:
            sample = self.transform(sample)
        return sample
