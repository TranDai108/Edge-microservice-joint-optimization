#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "[1/3] Checking Node.js and npx..."
command -v node >/dev/null 2>&1 || { echo "node not found"; exit 1; }
command -v npx >/dev/null 2>&1 || { echo "npx not found"; exit 1; }

if ! command -v codegraph >/dev/null 2>&1; then
  echo "[2/3] Installing CodeGraph via npx interactive installer..."
  npx @colbymchenry/codegraph
else
  echo "[2/3] CodeGraph already installed at: $(command -v codegraph)"
fi

echo "[3/3] Initializing CodeGraph index in repository..."
cd "$REPO_ROOT"
codegraph init -i

echo "Done. Configure your MCP client with: codegraph serve --mcp"
echo "Then restart or reload the client and run a smoke-test query."
