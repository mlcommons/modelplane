#!/usr/bin/env bash
set -uo pipefail

dockerfile="Dockerfile.mlflow"
uv_lock="uv.lock"

for file in "${dockerfile}" "${uv_lock}"; do
  [[ -f "${file}" ]] || { echo "Missing ${file}" >&2; exit 1; }
done

read_lock_version() {
  local pkg_path="$1"
  local pkg_name="$2"
  python - "${pkg_path}" "${pkg_name}" <<'PY'
import sys
import tomllib

path, pkg = sys.argv[1:3]
pkg = pkg.lower()

with open(path, "rb") as fp:
  data = tomllib.load(fp)

for package in data.get("package", []):
  if package.get("name", "").lower() == pkg:
    print(package.get("version", ""))
    break
PY
}

lock_mlflow_version="$(read_lock_version "${uv_lock}" "mlflow")"
if [[ -z "${lock_mlflow_version}" ]]; then
  echo "Unable to determine mlflow version from ${uv_lock}" >&2
  exit 1
fi
docker_image_version="$(head -n1 "${dockerfile}" | cut -d ':' -f2 | tr -d 'v')"
if [[ -z "${docker_image_version}" ]]; then
  echo "Unable to determine base image version from ${dockerfile}" >&2
  exit 1
fi
docker_mlflow_version="$(grep -o 'mlflow\[auth\]==[^[:space:]]*' "${dockerfile}" | head -n1 | cut -d '=' -f3)"
if [[ -z "${docker_mlflow_version}" ]]; then
  echo "Unable to determine mlflow version from ${dockerfile}" >&2
  exit 1
fi

mismatches=0

if [[ "${lock_mlflow_version}" != "${docker_image_version}" ]]; then
  echo "Mismatch: uv.lock mlflow (${lock_mlflow_version}) vs Docker base (${docker_image_version})" >&2
  mismatches=$((mismatches + 1))
fi

if [[ "${lock_mlflow_version}" != "${docker_mlflow_version}" ]]; then
  echo "Mismatch: uv.lock mlflow (${lock_mlflow_version}) vs Docker pip install (${docker_mlflow_version})" >&2
  mismatches=$((mismatches + 1))
fi

check_dep() {
  local pkg="$1"
  local token
  token="$(grep -o "${pkg}==[^[:space:]]*" "${dockerfile}" | head -n1)"
  local docker_version="${token##*==}"
  local lock_version_value="$(read_lock_version "${uv_lock}" "${pkg}")"


  if [[ -z "${docker_version}" || -z "${lock_version_value}" ]]; then
    echo "Unable to resolve ${pkg} in Dockerfile or uv.lock" >&2
    mismatches=$((mismatches + 1))
    return
  fi

  if [[ "${docker_version}" != "${lock_version_value}" ]]; then
    echo "Mismatch: ${pkg} Docker (${docker_version}) vs uv.lock (${lock_version_value})" >&2
    mismatches=$((mismatches + 1))
  fi
}

# Other dependencies from the dockerfile
check_dep "psycopg2-binary"
check_dep "boto3"
check_dep "google-cloud-storage"

if [[ ${mismatches} -ne 0 ]]; then
  exit 1
fi

echo "Dependency versions are consistent (${lock_mlflow_version})."
