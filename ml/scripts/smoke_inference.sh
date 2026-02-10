#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${1:-http://localhost:8001}"
TIMEOUT_SECONDS="${SMOKE_TIMEOUT_SECONDS:-30}"

echo "[smoke] Checking inference API at ${BASE_URL}"

health_json=""
for _ in $(seq 1 "${TIMEOUT_SECONDS}"); do
  if health_json="$(curl -fsS "${BASE_URL}/health" 2>/dev/null)"; then
    break
  fi
  sleep 1
done

if [[ -z "${health_json}" ]]; then
  echo "[smoke] ERROR: health endpoint did not become ready: ${BASE_URL}/health" >&2
  exit 1
fi

if [[ "${health_json}" != *"\"ok\":"* ]]; then
  echo "[smoke] ERROR: unexpected health payload: ${health_json}" >&2
  exit 1
fi

tmp_img="$(mktemp "/tmp/studybuddy-smoke-XXXXXX.png")"
cleanup() {
  rm -f "${tmp_img}"
}
trap cleanup EXIT

cat <<'EOF' | base64 --decode > "${tmp_img}"
iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGNgYGAAAAAEAAH2FzhVAAAAAElFTkSuQmCC
EOF

predict_json="$(curl -fsS -X POST "${BASE_URL}/predict" -F "file=@${tmp_img};type=image/png")"

if [[ "${predict_json}" != *"\"label\""* ]] || [[ "${predict_json}" != *"\"confidence\""* ]]; then
  echo "[smoke] ERROR: unexpected predict payload: ${predict_json}" >&2
  exit 1
fi

echo "[smoke] Health OK: ${health_json}"
echo "[smoke] Predict OK: ${predict_json}"
echo "[smoke] Inference smoke test passed."
