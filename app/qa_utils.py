import numpy as np

def mask_energy(delta: np.ndarray) -> float:
    """Mean heatmap intensity (evidence strength)."""
    return float(np.asarray(delta, dtype=np.float32).mean())

def monotonicity(alphas, energies) -> float:
    """Pearson corr between scale steps and resulting energies."""
    a = np.asarray(alphas, float); e = np.asarray(energies, float)
    if a.std() < 1e-8 or e.std() < 1e-8: return 0.0
    return float(np.corrcoef(a, e)[0,1])
