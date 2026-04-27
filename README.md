# Sleep Quality Pipeline

> A data science pipeline that joins nightly sleep stages with daily lifestyle habits, computes a sleep quality score, and ranks which habits hurt your sleep the most.

---

## Description

Most people have a gut feeling about what ruins their sleep — late-night coffee, a glass of wine, too much screen time — but rarely have the data to back it up. This project changes that.

Using a year of real fitness tracker exports, this pipeline automatically:

- **Merges** four data sources (sleep stages, step counts, screen time, lifestyle logs) into a single daily record
- **Scores** each night's sleep using a weighted formula that rewards deep and REM sleep while penalising time spent awake
- **Models** the relationship between your daily habits and your sleep quality using linear regression
- **Ranks** every lifestyle input by how strongly it pushes your sleep score up or down
- **Identifies** the single habit that is doing the most damage — the one you would most regret keeping

The result is a ranked, data-driven answer to the question: *"What is actually costing me my sleep?"*

This project is built entirely in Python using `pandas`, `numpy`, and `scikit-learn`. It is designed to run locally, inside Docker, or as part of a larger data science workflow.

---

## Overview

This project takes a year of fitness tracker data — sleep stages, step counts, screen time before bed, alcohol units, and caffeine intake — merges everything by date, and runs a linear regression to find out which daily habits have the strongest (positive or negative) effect on your sleep quality.

The pipeline ends with a ranked list of habits and a plain-English summary of which lifestyle choice you'd most regret keeping.

---

## Project Structure

```
sleep-quality-pipeline/
├── codebase/
│   ├── pipeline.py          # Core pipeline logic
│   ├── data/
│   │   ├── sleep_stages.csv
│   │   ├── step_counts.csv
│   │   ├── screen_time.csv
│   │   └── lifestyle_logs.csv
│   └── tests/               # Unit tests
├── app/
│   └── pipeline.py          # App entry point
├── Dockerfile               # Docker setup
├── requirements.txt         # Python dependencies
└── README.md
```

---

## Data Format

Each CSV file is joined on the `date` column (`YYYY-MM-DD`).

| File | Columns |
|------|---------|
| `sleep_stages.csv` | `date`, `awake_min`, `light_min`, `deep_min`, `rem_min` |
| `step_counts.csv` | `date`, `steps` |
| `screen_time.csv` | `date`, `screen_time_before_bed_min` |
| `lifestyle_logs.csv` | `date`, `alcohol_units`, `caffeine_after_noon` |

---

## Sleep Quality Score Formula

```
sleep_quality_score = (0.4 × deep_min) + (0.3 × rem_min) + (0.2 × light_min) − (0.1 × awake_min)
```

Deep and REM sleep are weighted highest. Awake time subtracts from the score.

---

## How It Works

1. **Load & Merge** — reads all 4 CSV files and inner-joins them on `date`
2. **Score** — computes a daily `sleep_quality_score` from sleep stage minutes
3. **Fit Model** — scales features with `StandardScaler` and trains a `LinearRegression`
4. **Rank & Summarize** — sorts features by their regression coefficient and prints the habit with the most negative effect

---

## Installation

```bash
# Clone the repository
git clone https://github.com/BIKKYDTU/sleep-quality-pipeline-data-science-san-francisco.git
cd sleep-quality-pipeline-data-science-san-francisco

# Install dependencies
pip install -r requirements.txt
```

**Dependencies:**
- `pandas >= 2.0`
- `numpy >= 1.24`
- `scikit-learn`
- `pytest`

---

## Usage

```bash
python codebase/pipeline.py
```

**Example output:**
```
1. caffeine_after_noon: -0.3821
2. alcohol_units: -0.2914
3. screen_time_before_bed_min: -0.1563
4. steps: 0.4102
Most regrettable habit: caffeine_after_noon
```

---

## Run with Docker

```bash
docker build -t sleep-pipeline .
docker run sleep-pipeline
```

---

## Run Tests

```bash
pytest
```

---

## License

MIT
