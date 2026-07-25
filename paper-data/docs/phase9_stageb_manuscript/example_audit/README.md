# Machine-Readable Audit Example

This directory demonstrates the `jacobian-coverage-audit/1.0` report contract
without adding scientific evidence.

The arrays are deliberately small illustrative fixtures. They are not model
outputs, do not correspond to a manuscript experiment, and must not be used as
effect-size or graph-recovery evidence.

Regenerate from the repository root:

```powershell
python paper-data/docs/phase9_stageb_manuscript/example_audit/generate_example.py
```

Outputs:

- `concat_x_only_audit.json`;
- `concat_x_only_profile.csv`;
- `partial_nominal_example.npy`;
- `total_nominal_example.npy`;
- `ARTIFACT_SHA256SUMS.txt`.
