#!/usr/bin/env bash
set -uo pipefail

pyproject="pyproject.toml"
dockerfile="Dockerfile.mlflow"
poetry_lock="poetry.lock"

for file in "${pyproject}" "${dockerfile}" "${poetry_lock}"; do
  [[ -f "${file}" ]] || { echo "Missing ${file}" >&2; exit 1; }
done

pyproject_mlflow_version="$(grep '^mlflow[[:space:]]*=' "${pyproject}" | head -n1 | sed -E 's/.*version[[:space:]]*=[[:space:]]*"([^"]+)".*/\1/')"
if [[ -z "${pyproject_mlflow_version}" ]]; then
  echo "Unable to determine mlflow version from ${pyproject}" >&2
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

if [[ "${pyproject_mlflow_version}" != "${docker_image_version}" ]]; then
  echo "Mismatch: pyproject mlflow (${pyproject_mlflow_version}) vs Docker base (${docker_image_version})" >&2
  mismatches=$((mismatches + 1))
fi

if [[ "${pyproject_mlflow_version}" != "${docker_mlflow_version}" ]]; then
  echo "Mismatch: pyproject mlflow (${pyproject_mlflow_version}) vs Docker pip install (${docker_mlflow_version})" >&2
  mismatches=$((mismatches + 1))
fi

lock_version() {
  local pkg="$1"
  local name_line
  name_line="$(grep -n "^name = \"${pkg}\"$" "${poetry_lock}" | head -n1)"
  [[ -z "${name_line}" ]] && return 1
  local line_number="${name_line%%:*}"
  sed -n "$((line_number + 1))p" "${poetry_lock}" | cut -d '"' -f2
}

check_dep() {
  local pkg="$1"
  local token
  token="$(grep -o "${pkg}==[^[:space:]]*" "${dockerfile}" | head -n1)"
  local docker_version="${token##*==}"
  local lock_version_value="$(lock_version "${pkg}")"

  if [[ -z "${docker_version}" || -z "${lock_version_value}" ]]; then
    echo "Unable to resolve ${pkg} in Dockerfile or poetry.lock" >&2
    mismatches=$((mismatches + 1))
    return
  fi

  if [[ "${docker_version}" != "${lock_version_value}" ]]; then
    echo "Mismatch: ${pkg} Docker (${docker_version}) vs poetry.lock (${lock_version_value})" >&2
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

echo "Dependency versions are consistent (${pyproject_mlflow_version})."
