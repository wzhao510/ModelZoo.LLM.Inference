#!/usr/bin/env bash

set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MANIFEST_FILE="${ROOT_DIR}/test_manifest.txt"
ARTIFACT_ROOT="${ROOT_DIR}/artifacts/test_runs"
TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
RUN_DIR="${ARTIFACT_ROOT}/${TIMESTAMP}"
SUMMARY_TSV="${RUN_DIR}/summary.tsv"
SUMMARY_MD="${RUN_DIR}/summary.md"
CONSOLE_LOG="${RUN_DIR}/console.log"

PYTHON_BIN="${PYTHON_BIN:-python3}"
STOP_ON_FAIL=0
LIST_ONLY=0
ONLY_PATTERN=""

usage() {
  cat <<EOF
Usage: ./run_all_tests.sh [options]

Options:
  --list                 List discovered tests and exit.
  --only <pattern>       Run only test files whose path contains <pattern>.
  --stop-on-fail         Stop after the first failing test file.
  --python-bin <path>    Override python executable. Default: python3
  -h, --help             Show this help message.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --list)
      LIST_ONLY=1
      shift
      ;;
    --only)
      ONLY_PATTERN="${2:-}"
      if [[ -z "${ONLY_PATTERN}" ]]; then
        echo "Missing value for --only" >&2
        exit 2
      fi
      shift 2
      ;;
    --stop-on-fail)
      STOP_ON_FAIL=1
      shift
      ;;
    --python-bin)
      PYTHON_BIN="${2:-}"
      if [[ -z "${PYTHON_BIN}" ]]; then
        echo "Missing value for --python-bin" >&2
        exit 2
      fi
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

discover_tests() {
  local tests=()
  if [[ -f "${MANIFEST_FILE}" ]]; then
    while IFS= read -r line || [[ -n "${line}" ]]; do
      [[ -z "${line}" ]] && continue
      [[ "${line}" =~ ^# ]] && continue
      tests+=("${line}")
    done < "${MANIFEST_FILE}"
  else
    while IFS= read -r file; do
      file="${file#./}"
      tests+=("${file}")
    done < <(cd "${ROOT_DIR}" && find . -type f -name 'test_*.py' | sort)
  fi

  if [[ -n "${ONLY_PATTERN}" ]]; then
    local filtered=()
    local item=""
    for item in "${tests[@]}"; do
      if [[ "${item}" == *"${ONLY_PATTERN}"* ]]; then
        filtered+=("${item}")
      fi
    done
    tests=("${filtered[@]}")
  fi

  printf '%s\n' "${tests[@]}"
}

mapfile -t TEST_FILES < <(discover_tests)

if [[ ${#TEST_FILES[@]} -eq 0 ]]; then
  echo "No test files found." >&2
  exit 1
fi

if [[ ${LIST_ONLY} -eq 1 ]]; then
  printf '%s\n' "${TEST_FILES[@]}"
  exit 0
fi

mkdir -p "${RUN_DIR}"
touch "${CONSOLE_LOG}"

exec > >(tee -a "${CONSOLE_LOG}") 2>&1

echo "Run directory: ${RUN_DIR}"
echo "Using python: ${PYTHON_BIN}"
echo

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "python executable not found: ${PYTHON_BIN}" >&2
  exit 2
fi

printf 'test_file\tstatus\tduration_seconds\tlog_file\tjunit_file\n' > "${SUMMARY_TSV}"

pass_count=0
fail_count=0
start_all="$(date +%s)"

run_one_test() {
  local test_file="$1"
  local test_name="${test_file//\//__}"
  local test_dir="${RUN_DIR}/${test_name}"
  local stdout_log="${test_dir}/pytest.log"
  local junit_log="${test_dir}/junit.xml"
  local start_ts end_ts duration status

  mkdir -p "${test_dir}"

  echo "============================================================"
  echo "Running ${test_file}"
  echo "Log: ${stdout_log}"
  echo "============================================================"

  start_ts="$(date +%s)"
  (
    cd "${ROOT_DIR}"
    "${PYTHON_BIN}" -m pytest -s -vv "${test_file}" --maxfail=1 --junitxml "${junit_log}"
  ) > "${stdout_log}" 2>&1
  local rc=$?
  end_ts="$(date +%s)"
  duration="$((end_ts - start_ts))"

  if [[ ${rc} -eq 0 ]]; then
    status="PASS"
    pass_count=$((pass_count + 1))
  else
    status="FAIL"
    fail_count=$((fail_count + 1))
  fi

  printf '%s\t%s\t%s\t%s\t%s\n' \
    "${test_file}" "${status}" "${duration}" "${stdout_log}" "${junit_log}" >> "${SUMMARY_TSV}"

  echo "[${status}] ${test_file} (${duration}s)"
  if [[ "${status}" == "FAIL" ]]; then
    echo "Last 200 log lines for ${test_file}:"
    tail -n 200 "${stdout_log}" || true
  fi

  return ${rc}
}

for test_file in "${TEST_FILES[@]}"; do
  if ! run_one_test "${test_file}"; then
    if [[ ${STOP_ON_FAIL} -eq 1 ]]; then
      echo "Stopping on first failure."
      break
    fi
  fi
  echo
done

end_all="$(date +%s)"
total_duration="$((end_all - start_all))"

{
  echo "# Test Summary"
  echo
  echo "| Test File | Status | Duration(s) | Log | JUnit |"
  echo "| --- | --- | ---: | --- | --- |"
  tail -n +2 "${SUMMARY_TSV}" | while IFS=$'\t' read -r test_file status duration log_file junit_file; do
    echo "| ${test_file} | ${status} | ${duration} | ${log_file} | ${junit_file} |"
  done
  echo
  echo "Total: ${#TEST_FILES[@]} files, PASS=${pass_count}, FAIL=${fail_count}, Duration=${total_duration}s"
} > "${SUMMARY_MD}"

echo "Summary TSV: ${SUMMARY_TSV}"
echo "Summary MD: ${SUMMARY_MD}"
echo "Total: ${#TEST_FILES[@]} files, PASS=${pass_count}, FAIL=${fail_count}, Duration=${total_duration}s"

if [[ ${fail_count} -gt 0 ]]; then
  exit 1
fi

exit 0

