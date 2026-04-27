#!/bin/bash
### COMMON SETUP; DO NOT MODIFY ###
set -e

# --- CONFIGURE THIS SECTION ---
run_all_tests() {
    echo "Running all tests..."

    if [ -d "/eval_assets/tests" ]; then
        TESTS_DIR="/eval_assets/tests"
    else
        TESTS_DIR="/app/tests"
    fi

    # If the agent's pipeline.py is missing, drop in a stub that satisfies
    # the test module's `from pipeline import (...)` so pytest can collect
    # every individual test. Each test then fails at runtime, producing one
    # FAILED entry per test in before.json (instead of a single collection
    # ERROR). The stub is only created when no real pipeline.py is present.
    if [ ! -f /app/pipeline.py ]; then
        cat > /app/pipeline.py <<'PYEOF'
import numpy as np
import pandas as pd

def load_and_merge_data(sleep_stages_path, step_counts_path, screen_time_path, lifestyle_logs_path):
    return pd.DataFrame()

def compute_sleep_quality_score(df):
    return df

def fit_model(df):
    return np.array([]), []

def rank_and_summarize(coefficients, feature_names):
    pass

def main():
    pass
PYEOF
    fi

    cd /app
    PYTHONPATH=/app PIPELINE_REPO_ROOT=/app \
        python3 -m pytest "$TESTS_DIR" -v --tb=short --no-header || true
}
# --- END CONFIGURATION SECTION ---

### COMMON EXECUTION; DO NOT MODIFY ###
run_all_tests
