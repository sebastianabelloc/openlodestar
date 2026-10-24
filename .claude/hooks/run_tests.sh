#!/bin/bash
# runs the test suite after every edit; if it fails, the output goes back to Claude (exit 2)
cd "$CLAUDE_PROJECT_DIR" || exit 0
if [ -x .venv/bin/python ]; then PY=.venv/bin/python; else PY=python3; fi
out=$("$PY" -m pytest -q -x 2>&1)
if [ $? -ne 0 ]; then
  echo "Tests failed after this edit:" >&2
  echo "$out" | tail -25 >&2
  exit 2
fi
exit 0
