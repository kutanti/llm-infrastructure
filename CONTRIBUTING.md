# Contributing

Keep a distinction between a derivation, a numerical simulation, and a measured
runtime result. A new optimization claim needs a matched workload, effective
configuration, quality result, and timing/memory boundaries.

For changes to the connected lab:

```powershell
.\pipeline\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

See [environment setup](pipeline/ENVIRONMENTS.md) for the optional dependencies.
Tests use temporary directories and locally constructed models; they must not
download pretrained checkpoints, require a GPU, or contact an external service.
GPU results belong in a separately labeled experiment, not an offline assertion.

Use the same routing schema when extending the connected path, or introduce a
versioned task explicitly. Do not silently change held-out examples to make a
result improve. Numerical expectations should use justified tolerances; latency
thresholds must not depend on a particular laptop.

Keep generated models, tokenizer caches, traces, prompts from real users, and
raw machine-specific result files out of Git. An included measurement should be
sanitized and labeled with its workload and limitations. Inspect the staged file
list before pushing.

For lesson changes, keep relative links valid and put explanations near their
equations or implementations. Regenerate visuals using the
[checked-in renderer](assets/README.md#rebuild-the-images) rather than editing a
GIF independently of its source.
