#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 RELEASE_DIR ENV_PREFIX RELEASE_MANIFEST OUTPUT_BASE" >&2
  exit 2
fi

release_dir="$1"
env_prefix="$2"
release_manifest="$3"
output_base="$4"
python_bin="${env_prefix}/bin/python"
config="${release_dir}/configs/phase9_lorenz_901_confirmation_v1.json"
runner="${release_dir}/experiments/phase9_lorenz_901_confirmation.py"
aggregator="${release_dir}/experiments/aggregate_phase9_lorenz_901_confirmation.py"

smoke_a="${output_base}/smoke_a"
smoke_b="${output_base}/smoke_b"
formal_root="${output_base}/formal"
aggregate_dir="${formal_root}/aggregate"
dry_root="${output_base}/dry_validation"

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=0

cd "${release_dir}"

"${python_bin}" -m pytest -q \
  tests/test_phase9_lorenz_901_confirmation.py \
  tests/test_phase9_lorenz_strong_operating_preflight.py \
  tests/test_phase9_audit_generalization.py

"${python_bin}" "${runner}" \
  --mode dry-run \
  --config "${config}" \
  --release-manifest "${release_manifest}" \
  --output-root "${dry_root}"

"${python_bin}" "${runner}" \
  --mode smoke \
  --config "${config}" \
  --release-manifest "${release_manifest}" \
  --output-root "${smoke_a}" \
  --tag smoke_a \
  --methods baseline,mamba_concat \
  --device cuda

"${python_bin}" "${runner}" \
  --mode smoke \
  --config "${config}" \
  --release-manifest "${release_manifest}" \
  --output-root "${smoke_b}" \
  --tag smoke_b \
  --methods mamba_concat \
  --device cuda

"${python_bin}" "${runner}" \
  --mode validate-smoke \
  --config "${config}" \
  --release-manifest "${release_manifest}" \
  --output-root "${output_base}/smoke_validation" \
  --smoke-root-a "${smoke_a}" \
  --smoke-root-b "${smoke_b}"

"${python_bin}" "${runner}" \
  --mode generate-data \
  --config "${config}" \
  --release-manifest "${release_manifest}" \
  --output-root "${formal_root}"

"${python_bin}" "${runner}" \
  --mode formal \
  --config "${config}" \
  --release-manifest "${release_manifest}" \
  --output-root "${formal_root}" \
  --tag formal \
  --methods baseline,mamba_concat \
  --device cuda \
  --resume

"${python_bin}" "${aggregator}" \
  --config "${config}" \
  --release-manifest "${release_manifest}" \
  --formal-root "${formal_root}" \
  --output-dir "${aggregate_dir}"

