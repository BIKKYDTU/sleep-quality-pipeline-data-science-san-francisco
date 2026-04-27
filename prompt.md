# Sleep Quality Lifestyle Impact Analysis Pipeline

## Description / Context

This project requires a personal health analytics pipeline written in Python. The pipeline reads one year of nightly sleep-stage estimates and four daily lifestyle metrics from local CSV files, joins all sources on date, derives a numeric sleep quality score from the sleep stage columns, fits a linear regression model linking that score to the lifestyle inputs, and prints all inputs ranked by their signed effect on sleep quality together with a plain English summary identifying the most harmful lifestyle habit.

## Tech Stack

- Python 3.x
- pandas
- scikit-learn
- numpy

## Key Requirements

### Data Sources and Schema

- All input data is read from four CSV files located under the `data/` directory:
  - `data/sleep_stages.csv` with columns: `date`, `awake_min`, `light_min`, `deep_min`, `rem_min`
  - `data/step_counts.csv` with columns: `date`, `steps`
  - `data/screen_time.csv` with columns: `date`, `screen_time_before_bed_min`
  - `data/lifestyle_logs.csv` with columns: `date`, `alcohol_units`, `caffeine_after_noon`
- The `date` column in all files uses the format `YYYY-MM-DD`.

### Data Merging

- All four CSV files are merged into a single daily DataFrame via inner join on the `date` column.
- Any row containing missing values in the merged DataFrame is dropped before further processing.

### Sleep Quality Score

- A `sleep_quality_score` column is computed from the merged DataFrame using the formula: `0.4 * deep_min + 0.3 * rem_min + 0.2 * light_min - 0.1 * awake_min`.

### Model Fitting

- A linear regression model is fit with `sleep_quality_score` as the target variable.
- The four feature columns used are: `steps`, `screen_time_before_bed_min`, `alcohol_units`, `caffeine_after_noon`.
- Each feature column is standardized to zero mean and unit variance before model fitting.

### Ranked Output and Summary

- All four features are ranked by their signed standardized regression coefficient in ascending order (most negative first).
- Each ranked feature is printed to stdout on its own line in the format: `<rank>. <feature_name>: <coefficient>`, where `<coefficient>` is rounded to 4 decimal places.
- After the ranked list, a plain English summary line is printed. The summary considers only the three habit features: `screen_time_before_bed_min`, `alcohol_units`, and `caffeine_after_noon`. If at least one of these three features has a negative standardized coefficient, the summary line is printed in the format: `Most regrettable habit: <feature_name>`, where `<feature_name>` is the habit feature with the most negative standardized coefficient. If none of the three habit features has a negative coefficient, the summary line printed is: `No lifestyle habit showed a negative association in this model.`

## Expected Interface

- **Path:** `pipeline.py`
- **Name:** `load_and_merge_data`
- **Type:** function
- **Input:** `sleep_stages_path: str, step_counts_path: str, screen_time_path: str, lifestyle_logs_path: str`
- **Output:** `pd.DataFrame`
- **Description:** Loads the four CSV files from the provided paths and merges them into a single DataFrame via inner join on the `date` column. Drops any rows with missing values. The returned DataFrame contains all nine columns: `date`, `awake_min`, `light_min`, `deep_min`, `rem_min`, `steps`, `screen_time_before_bed_min`, `alcohol_units`, `caffeine_after_noon`.

---

- **Path:** `pipeline.py`
- **Name:** `compute_sleep_quality_score`
- **Type:** function
- **Input:** `df: pd.DataFrame`
- **Output:** `pd.DataFrame`
- **Description:** Adds a `sleep_quality_score` column to the DataFrame using the formula `0.4 * deep_min + 0.3 * rem_min + 0.2 * light_min - 0.1 * awake_min` and returns the DataFrame with this column appended.

---

- **Path:** `pipeline.py`
- **Name:** `fit_model`
- **Type:** function
- **Input:** `df: pd.DataFrame`
- **Output:** `tuple[np.ndarray, list[str]]`
- **Description:** Standardizes the four feature columns (`steps`, `screen_time_before_bed_min`, `alcohol_units`, `caffeine_after_noon`) to zero mean and unit variance, fits a linear regression model with `sleep_quality_score` as the target, and returns a tuple of `(coefficients, feature_names)` where `coefficients` is a NumPy array of the four standardized regression coefficients and `feature_names` is the list `['steps', 'screen_time_before_bed_min', 'alcohol_units', 'caffeine_after_noon']`.

---

- **Path:** `pipeline.py`
- **Name:** `rank_and_summarize`
- **Type:** function
- **Input:** `coefficients: np.ndarray, feature_names: list[str]`
- **Output:** `None`
- **Description:** Prints all four features ranked by ascending signed coefficient value (most negative first) to stdout, one per line, in the format `<rank>. <feature_name>: <coefficient>` with the coefficient rounded to 4 decimal places. Then prints a summary line derived only from the habit features `screen_time_before_bed_min`, `alcohol_units`, and `caffeine_after_noon`: if at least one of those three has a negative coefficient, prints `Most regrettable habit: <feature_name>` naming the one with the most negative coefficient; otherwise prints `No lifestyle habit showed a negative association in this model.`

---

- **Path:** `pipeline.py`
- **Name:** `main`
- **Type:** function
- **Input:** `None`
- **Output:** `None`
- **Description:** Orchestrates the full pipeline by calling `load_and_merge_data` with the default paths `data/sleep_stages.csv`, `data/step_counts.csv`, `data/screen_time.csv`, and `data/lifestyle_logs.csv`, then `compute_sleep_quality_score`, then `fit_model`, then `rank_and_summarize`. Produces the ranked feature effects and plain English summary on stdout.

## Current State

Empty repository with test files only. All input CSV files are present under `data/`.
