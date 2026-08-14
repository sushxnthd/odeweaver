from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations_with_replacement
import numpy as np


@dataclass(frozen=True)
class SparseModel:
    terms: tuple[str, ...]
    coefficients: np.ndarray

    def equation(self, target: str = "x") -> str:
        parts=[]
        for term, coef in zip(self.terms, self.coefficients):
            if abs(float(coef)) > 1e-12:
                parts.append(f"{float(coef):+.4g}*{term}")
        return f"d{target}/dt = " + (" ".join(parts).lstrip("+") or "0")


def derivative(x: np.ndarray, dt: float) -> np.ndarray:
    a=np.asarray(x,dtype=float)
    if a.ndim != 2 or len(a) < 5 or dt <= 0:
        raise ValueError("x must be a 2D trajectory with at least 5 rows and positive dt")
    return (a[2:] - a[:-2]) / (2.0*dt)


def polynomial_library(x: np.ndarray, degree: int = 2) -> tuple[np.ndarray, tuple[str,...]]:
    a=np.asarray(x,dtype=float)
    if a.ndim != 2 or degree < 1 or degree > 4:
        raise ValueError("x must be 2D and degree must be between 1 and 4")
    cols=[np.ones(len(a))]; names=["1"]
    n=a.shape[1]
    for d in range(1,degree+1):
        for combo in combinations_with_replacement(range(n), d):
            v=np.ones(len(a)); bits=[]
            for j in combo:
                v*=a[:,j]; bits.append(f"x{j}")
            cols.append(v); names.append("*".join(bits))
    return np.column_stack(cols), tuple(names)


def fit(x: np.ndarray, dt: float, *, degree: int = 2, threshold: float = 0.05, iterations: int = 8) -> list[SparseModel]:
    if threshold < 0 or iterations < 1:
        raise ValueError("threshold must be non-negative and iterations positive")
    a=np.asarray(x,dtype=float)
    dx=derivative(a,dt)
    theta,names=polynomial_library(a[1:-1],degree)
    coef=np.linalg.lstsq(theta,dx,rcond=None)[0]
    for _ in range(iterations):
        small=np.abs(coef)<threshold; coef[small]=0.0
        for k in range(coef.shape[1]):
            keep=~small[:,k]
            if np.any(keep): coef[keep,k]=np.linalg.lstsq(theta[:,keep],dx[:,k],rcond=None)[0]
    return [SparseModel(names,coef[:,k].copy()) for k in range(coef.shape[1])]
