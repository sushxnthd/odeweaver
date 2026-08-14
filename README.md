# odeweaver

[Live demo](https://sushxnthd.github.io/odeweaver/) | [Architecture](docs/architecture.md) | [Benchmark](benchmarks/results.json)

`odeweaver` discovers compact polynomial dynamics from sampled trajectories using sequential thresholded least squares. It is a small, inspectable baseline for equation discovery experiments.

```python
from odeweaver import fit
models = fit(trajectory, dt=0.002, degree=2, threshold=0.08)
print(models[0].equation())
```

## Method

1. Estimate time derivatives with centered differences.
2. Build a polynomial feature library.
3. Fit least squares coefficients.
4. Refit after repeatedly removing coefficients below the sparsity threshold.

## Benchmark

The included benchmark generates logistic growth with light measurement noise. The true dynamics are `x' = 1.5 x - 0.75 x^2`, so coefficient recovery can be checked directly rather than judged by an arbitrary score.

## Scope

Derivative estimation is noise-sensitive, and polynomial libraries are not appropriate for every dynamical system. The package is intended as a transparent discovery baseline and diagnostics layer.

## Roadmap

- Savitzky-Golay derivative option
- rational and trigonometric libraries
- train/validation rollout scoring
- coefficient-path plots

MIT licensed.
