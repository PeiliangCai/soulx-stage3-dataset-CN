#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
official_commit="928b06508ed2de1344208d06fb1f6fb2ebfb1df5"
expected_diff_sha256="b1d55ba7530771cd3df6674456df498e1d7444dc93b55a1405f5557cc618f8da"
expected_audit_sha256="d5cfa8b4c90a57db22e45f2e70b0c3d92e2dcf6ea9aa017528459e2ee33f16c4"
expected_doc_sha256="f8f8571f957a8e9dae3f4950997945fabd6118e0de806e8000407332bcf76e9d"

source_repo="${1:-${project_root}/third_party/SoulX-Duplug-upstream}"
runtime_dir="${2:-${project_root}/runtimes/SoulX-Duplug-928b065-official-continual-v1}"
bundle_dir="${project_root}/patches/soulx_stage3_official_continual_v1"

if [[ ! -d "${source_repo}/.git" ]]; then
  echo "Missing clean SoulX checkout: ${source_repo}" >&2
  echo "Clone the official repository and pin it to ${official_commit} first." >&2
  exit 2
fi
if [[ -e "${runtime_dir}" ]]; then
  echo "Refusing to overwrite existing runtime: ${runtime_dir}" >&2
  exit 2
fi
if [[ "$(git -C "${source_repo}" rev-parse HEAD)" != "${official_commit}" ]]; then
  echo "Source checkout is not pinned to ${official_commit}" >&2
  exit 2
fi
if [[ -n "$(git -C "${source_repo}" status --porcelain)" ]]; then
  echo "Source checkout is not clean: ${source_repo}" >&2
  exit 2
fi

git clone --no-hardlinks "${source_repo}" "${runtime_dir}"
git -C "${runtime_dir}" checkout --detach "${official_commit}"
git -C "${runtime_dir}" apply "${bundle_dir}/tracked_changes.patch"
install -m 0644 "${bundle_dir}/continual_audit.py" "${runtime_dir}/utils/continual_audit.py"
install -m 0644 "${bundle_dir}/OFFICIAL_CONTINUAL_PATCH.md" "${runtime_dir}/OFFICIAL_CONTINUAL_PATCH.md"

actual_diff_sha256="$(git -C "${runtime_dir}" diff --binary | sha256sum | awk '{print $1}')"
actual_audit_sha256="$(sha256sum "${runtime_dir}/utils/continual_audit.py" | awk '{print $1}')"
actual_doc_sha256="$(sha256sum "${runtime_dir}/OFFICIAL_CONTINUAL_PATCH.md" | awk '{print $1}')"

[[ "${actual_diff_sha256}" == "${expected_diff_sha256}" ]]
[[ "${actual_audit_sha256}" == "${expected_audit_sha256}" ]]
[[ "${actual_doc_sha256}" == "${expected_doc_sha256}" ]]

echo "Prepared audited runtime: ${runtime_dir}"
echo "Official commit: ${official_commit}"
echo "Tracked diff SHA-256: ${actual_diff_sha256}"
