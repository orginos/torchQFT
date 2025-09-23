
import os, json, uuid
from pathlib import Path
import numpy as np
from typing import Dict, Optional, Any, List
from .schema import RunMeta, Observables

class LatticeDBWriter:
    """Append-only writer for the lattice database.
    Stores one folder per run with NPZ arrays + JSON metadata, and maintains an index.csv for discovery.
    """
    def __init__(self, root: str):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "runs").mkdir(exist_ok=True)
        self.index_path = self.root / "index.csv"
        if not self.index_path.exists():
            self.index_path.write_text("run_id,model,g2,g4,beta,Lx,Lt,a,seed,algorithm,n_therm,n_meas,sep,commit,relpath\n")

    def _append_index(self, run_id: str, meta: RunMeta, relpath: str):
        # None -> empty string for CSV
        def fmt(x): return "" if x is None else x
        line = f"{run_id},{meta.model},{fmt(meta.g2)},{fmt(meta.g4)},{fmt(meta.beta)},{meta.Lx},{meta.Lt},{fmt(meta.a)},{meta.seed},{meta.algorithm},{meta.n_therm},{meta.n_meas},{meta.sep},{meta.commit},{relpath}\n"
        with open(self.index_path, "a") as f:
            f.write(line)

    def save_run(self,
                 meta: RunMeta,
                 correlators: Dict[str, np.ndarray],
                 observables: Optional[Observables] = None,
                 configs: Optional[np.ndarray] = None,
                 fourpoint: Optional[Dict[str, np.ndarray]] = None,
                 run_id: Optional[str] = None) -> str:
        """Save a simulation run.
        Args:
          meta: RunMeta describing the run
          correlators: dict of arrays (e.g., {'C_t': (Nt,), 'C_pt': (Np,Nt)})
          observables: simple scalars (chi, binder, xi, ...)
          configs: optional configurations array (n_cfg, Lx, Lt) or similar
          fourpoint: dict of 4-pt correlators or moments
          run_id: optional external id; otherwise uuid4
        Returns:
          run_id
        """
        rid = run_id or str(uuid.uuid4())
        rdir = self.root / "runs" / rid
        rdir.mkdir(parents=True, exist_ok=True)

        # Write metadata
        (rdir / "metadata.json").write_text(meta.to_json())

        # Correlators
        np.savez_compressed(rdir / "correlators.npz", **correlators)

        # Observables
        if observables is not None:
            (rdir / "observables.json").write_text(observables.to_json())

        # 4-point or moments
        if fourpoint is not None:
            # save each as separate npz or combine
            np.savez_compressed(rdir / "fourpoint.npz", **fourpoint)

        # Configurations (optional, can be large)
        if configs is not None:
            np.savez_compressed(rdir / "configs.npz", configs=configs)

        # Relpath for index
        relpath = str(Path("runs") / rid)
        self._append_index(rid, meta, relpath)
        return rid
