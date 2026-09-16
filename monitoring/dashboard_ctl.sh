#!/usr/bin/env bash
set -euo pipefail

# Lightweight controller for the local Streamlit dashboard.
# Usage:
#   ./dashboard_ctl.sh start|stop|restart|status|log|logs|url

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_FILE="$ROOT_DIR/milp_dashboard.py"
LOG_FILE="$ROOT_DIR/.streamlit_dashboard.log"
PID_FILE="$ROOT_DIR/.streamlit_dashboard.pid"
REDIS_FWD_LOG_FILE="$ROOT_DIR/.redis_port_forward.log"
REDIS_FWD_PID_FILE="$ROOT_DIR/.redis_port_forward.pid"

PORT="${STREAMLIT_PORT:-8501}"
HOST="${STREAMLIT_HOST:-0.0.0.0}"
REDIS_HOST_VALUE="${REDIS_HOST:-localhost}"
KUBECTL_BIN_VALUE="${KUBECTL_BIN:-kubectl}"
KUBECONFIG_VALUE="${KUBECONFIG:-$HOME/.kube/config}"
REDIS_AUTO_FORWARD="${REDIS_AUTO_FORWARD:-true}"
REDIS_FWD_NAMESPACE="${REDIS_FWD_NAMESPACE:-default}"
REDIS_FWD_RESOURCE="${REDIS_FWD_RESOURCE:-svc/redis}"
REDIS_FWD_LOCAL_PORT="${REDIS_FWD_LOCAL_PORT:-6379}"
REDIS_FWD_REMOTE_PORT="${REDIS_FWD_REMOTE_PORT:-6379}"

# The dashboard is normally launched by the regular ubuntu user.  If a parent
# shell still exports KUBECTL_BIN="sudo kubectl", sudo can fail under VS Code /
# sandboxed terminals and make `./dashboard_ctl.sh start` look broken.
if [[ "$KUBECTL_BIN_VALUE" == "sudo kubectl" ]]; then
  KUBECTL_BIN_VALUE="kubectl"
fi

_is_pid_file_running() {
  local file="$1"
  if [[ -f "$file" ]]; then
    local pid
    pid="$(cat "$file" 2>/dev/null || true)"
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      return 0
    fi
  fi
  return 1
}

_is_running() {
  if _is_pid_file_running "$PID_FILE"; then
    return 0
  fi

  local pid
  pid="$(pgrep -f "streamlit run ${APP_FILE}" 2>/dev/null | head -n 1 || true)"
  if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
    echo "$pid" >"$PID_FILE"
    return 0
  fi

  rm -f "$PID_FILE"
  return 1
}

_redis_forward_running() {
  _is_pid_file_running "$REDIS_FWD_PID_FILE"
}

_redis_port_listening() {
  ss -ltn "( sport = :${REDIS_FWD_LOCAL_PORT} )" 2>/dev/null | grep -q ":${REDIS_FWD_LOCAL_PORT}"
}

start_redis_forward() {
  if [[ "$REDIS_AUTO_FORWARD" != "true" ]]; then
    return 0
  fi

  if [[ "$REDIS_HOST_VALUE" != "localhost" && "$REDIS_HOST_VALUE" != "127.0.0.1" ]]; then
    return 0
  fi

  if _redis_forward_running; then
    return 0
  fi

  if _redis_port_listening; then
    echo "Redis local port ${REDIS_FWD_LOCAL_PORT} already in use; skip auto port-forward."
    return 0
  fi

  echo "[$(date '+%Y-%m-%d %H:%M:%S')] --- redis port-forward start ---" >>"$REDIS_FWD_LOG_FILE"
  KUBECONFIG="$KUBECONFIG_VALUE" nohup $KUBECTL_BIN_VALUE port-forward -n "$REDIS_FWD_NAMESPACE" "$REDIS_FWD_RESOURCE" \
    "${REDIS_FWD_LOCAL_PORT}:${REDIS_FWD_REMOTE_PORT}" --address 127.0.0.1 \
    >>"$REDIS_FWD_LOG_FILE" 2>&1 &
  echo $! >"$REDIS_FWD_PID_FILE"

  for _ in {1..20}; do
    if _redis_port_listening; then
      return 0
    fi
    sleep 0.2
  done

  printf 'Warning: Redis port-forward may not be ready.\n'
  printf '         Check log: %s\n' "$REDIS_FWD_LOG_FILE"
}

stop_redis_forward() {
  if ! _redis_forward_running; then
    rm -f "$REDIS_FWD_PID_FILE"
    return 0
  fi

  local pid
  pid="$(cat "$REDIS_FWD_PID_FILE")"
  kill "$pid" 2>/dev/null || true

  for _ in {1..20}; do
    if ! kill -0 "$pid" 2>/dev/null; then
      break
    fi
    sleep 0.2
  done

  if kill -0 "$pid" 2>/dev/null; then
    kill -9 "$pid" 2>/dev/null || true
  fi

  rm -f "$REDIS_FWD_PID_FILE"
}

_streamlit_cmd() {
  if command -v streamlit >/dev/null 2>&1; then
    echo "streamlit"
  else
    echo "python3 -m streamlit"
  fi
}

start() {
  if _is_running; then
    echo "Dashboard already running (PID $(cat "$PID_FILE"))."
    echo "URL: http://localhost:$PORT"
    return 0
  fi

  local cmd
  cmd="$(_streamlit_cmd)"

  start_redis_forward

  (
    cd "$ROOT_DIR"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] --- streamlit start ---" >>"$LOG_FILE"
    REDIS_HOST="$REDIS_HOST_VALUE" KUBECTL_BIN="$KUBECTL_BIN_VALUE" KUBECONFIG="$KUBECONFIG_VALUE" nohup setsid $cmd run "$APP_FILE" \
      --server.port "$PORT" \
      --server.address "$HOST" \
      --server.headless true \
      </dev/null \
      >>"$LOG_FILE" 2>&1 &
    echo $! >"$PID_FILE"
  )

  sleep 1
  if _is_running; then
    printf '\nDashboard started\n'
    printf '  PID:          %s\n' "$(cat "$PID_FILE")"
    printf '  URL:          http://localhost:%s\n' "$PORT"
    printf '  REDIS_HOST:   %s\n' "$REDIS_HOST_VALUE"
    printf '  KUBECTL_BIN:  %s\n' "$KUBECTL_BIN_VALUE"
    printf '  KUBECONFIG:   %s\n' "$KUBECONFIG_VALUE"
    printf '  Log:          %s\n' "$LOG_FILE"
    if _redis_forward_running; then
      printf '  Redis fwd:    running (PID %s)\n' "$(cat "$REDIS_FWD_PID_FILE")"
    fi
  else
    echo "Failed to start dashboard. Check log: $LOG_FILE"
    exit 1
  fi
}

stop() {
  if ! _is_running; then
    echo "Dashboard is not running."
    rm -f "$PID_FILE"
    return 0
  fi

  local pid
  pid="$(cat "$PID_FILE")"
  kill "$pid" 2>/dev/null || true

  for _ in {1..20}; do
    if ! kill -0 "$pid" 2>/dev/null; then
      break
    fi
    sleep 0.2
  done

  if kill -0 "$pid" 2>/dev/null; then
    kill -9 "$pid" 2>/dev/null || true
  fi

  rm -f "$PID_FILE"
  stop_redis_forward
  echo "Dashboard stopped."
}

status() {
  if _is_running; then
    echo "Dashboard running (PID $(cat "$PID_FILE"))."
    echo "URL: http://localhost:$PORT"
  else
    echo "Dashboard not running."
  fi

  if _redis_forward_running; then
    echo "Redis forward running (PID $(cat "$REDIS_FWD_PID_FILE")) on 127.0.0.1:${REDIS_FWD_LOCAL_PORT}."
  else
    echo "Redis forward not running."
  fi
}

logs() {
  if [[ ! -f "$LOG_FILE" ]]; then
    echo "No log file yet: $LOG_FILE"
    return 0
  fi
  tail -n 120 -F "$LOG_FILE"
}

url() {
  echo "http://localhost:$PORT"
}

case "${1:-}" in
  start)
    start
    ;;
  stop)
    stop
    ;;
  restart)
    stop
    start
    ;;
  status)
    status
    ;;
  log|logs)
    logs
    ;;
  url)
    url
    ;;
  *)
    echo "Usage: $0 {start|stop|restart|status|log|logs|url}"
    exit 1
    ;;
esac
