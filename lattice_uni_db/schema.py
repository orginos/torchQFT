
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List
import time, json

@dataclass
class RunMeta:
    # Core metadata for one simulation run
    model: str              # "phi4_2d", "o3_2d", "qcd_4d", ...
    g2: Optional[float]     # scalar phi4 mass/coupling (optional)
    g4: Optional[float]     # scalar phi4 quartic coupling (optional)
    beta: Optional[float]   # inverse coupling for O(3) or others
    Lx: int
    Lt: int
    a: Optional[float]      # lattice spacing (if known/mapped from beta)
    seed: int
    algorithm: str          # e.g. "HMC", "Wolff", "Metropolis"
    n_therm: int
    n_meas: int
    sep: int                # MC steps between saved measurements
    commit: str             # code version/hash
    extra: Optional[Dict[str, Any]] = None  # anything else

    def to_json(self) -> str:
        d = asdict(self)
        d["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        return json.dumps(d, indent=2)

@dataclass
class Observables:
    # Low-order observables (per run)
    chi: Optional[float] = None           # susceptibility
    binder: Optional[float] = None        # Binder cumulant
    xi: Optional[float] = None            # correlation length
    energy: Optional[float] = None        # action density or energy
    extras: Optional[Dict[str, Any]] = None

    def to_json(self) -> str:
        return json.dumps({k:v for k,v in self.__dict__.items()}, indent=2)
