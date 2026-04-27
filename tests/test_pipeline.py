

import contextlib
import io
import os
import re
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pipeline import (
    compute_sleep_quality_score,
    fit_model,
    load_and_merge_data,
    main,
    rank_and_summarize,
)

# ---------------------------------------------------------------------------
# Constants — derived directly from the prompt's interface specification
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS = [
    "date",
    "awake_min",
    "light_min",
    "deep_min",
    "rem_min",
    "steps",
    "screen_time_before_bed_min",
    "alcohol_units",
    "caffeine_after_noon",
]

FEATURE_NAMES = [
    "steps",
    "screen_time_before_bed_min",
    "alcohol_units",
    "caffeine_after_noon",
]

HABIT_FEATURES = {"screen_time_before_bed_min", "alcohol_units", "caffeine_after_noon"}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _capture_stdout(fn, *args, **kwargs):
    """Run fn(*args, **kwargs) and return the text written to stdout."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        fn(*args, **kwargs)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Fixtures
# 10 rows are used so the linear regression system is well-determined
# (more samples than features), giving a unique, numerically stable solution.
# ---------------------------------------------------------------------------

_DATES = [f"2023-01-{d:02d}" for d in range(1, 11)]


@pytest.fixture
def sample_csv_paths(tmp_path):
    """
    Writes four consistent CSV files (10 rows each) matching the prompt schema
    and returns their paths as (sleep, steps, screen, lifestyle).
    """
    sleep_data = pd.DataFrame(
        {
            "date": _DATES,
            "awake_min": [30, 20, 25, 35, 15, 40, 22, 28, 18, 32],
            "light_min": [120, 130, 110, 100, 140, 90, 125, 115, 135, 105],
            "deep_min": [90, 80, 100, 70, 110, 60, 95, 85, 105, 75],
            "rem_min": [60, 70, 65, 55, 80, 50, 68, 62, 75, 58],
        }
    )
    steps_data = pd.DataFrame(
        {
            "date": _DATES,
            "steps": [8000, 10000, 7500, 6000, 12000, 5000, 9000, 8500, 11000, 7000],
        }
    )
    screen_data = pd.DataFrame(
        {
            "date": _DATES,
            "screen_time_before_bed_min": [45, 30, 60, 75, 20, 90, 40, 55, 25, 70],
        }
    )
    lifestyle_data = pd.DataFrame(
        {
            "date": _DATES,
            "alcohol_units": [1, 0, 2, 0, 0, 3, 1, 0, 0, 2],
            "caffeine_after_noon": [1, 0, 1, 0, 1, 1, 0, 1, 0, 0],
        }
    )

    sleep_path = str(tmp_path / "sleep_stages.csv")
    steps_path = str(tmp_path / "step_counts.csv")
    screen_path = str(tmp_path / "screen_time.csv")
    lifestyle_path = str(tmp_path / "lifestyle_logs.csv")

    sleep_data.to_csv(sleep_path, index=False)
    steps_data.to_csv(steps_path, index=False)
    screen_data.to_csv(screen_path, index=False)
    lifestyle_data.to_csv(lifestyle_path, index=False)

    return sleep_path, steps_path, screen_path, lifestyle_path


@pytest.fixture
def merged_df(sample_csv_paths):
    """Returns a merged DataFrame produced by load_and_merge_data."""
    sleep_path, steps_path, screen_path, lifestyle_path = sample_csv_paths
    return load_and_merge_data(sleep_path, steps_path, screen_path, lifestyle_path)


@pytest.fixture
def scored_df(merged_df):
    """Returns the merged DataFrame with the sleep_quality_score column added."""
    return compute_sleep_quality_score(merged_df.copy())


# ---------------------------------------------------------------------------
# load_and_merge_data
# ---------------------------------------------------------------------------


class TestLoadAndMergeData:
    def test_load_and_merge_returns_all_required_columns(self, sample_csv_paths):
        """
        Merged DataFrame must be a pd.DataFrame containing every one of the
        nine columns specified in the prompt schema (R-1).
        """
        sleep_path, steps_path, screen_path, lifestyle_path = sample_csv_paths
        df = load_and_merge_data(sleep_path, steps_path, screen_path, lifestyle_path)

        assert isinstance(df, pd.DataFrame)
        for col in REQUIRED_COLUMNS:
            assert col in df.columns, f"Expected column '{col}' is missing"

    def test_load_and_merge_inner_join_excludes_unmatched_dates(self, tmp_path):
        """
        Dates that appear in fewer than all four source files must not appear
        in the merged DataFrame (inner join on date, R-2).
        """
        common_dates = ["2023-03-01", "2023-03-02"]
        extra_date = "2023-03-03"  # only in sleep_stages

        sleep = pd.DataFrame(
            {
                "date": common_dates + [extra_date],
                "awake_min": [10, 15, 20],
                "light_min": [60, 65, 70],
                "deep_min": [50, 55, 60],
                "rem_min": [40, 45, 50],
            }
        )
        steps = pd.DataFrame({"date": common_dates, "steps": [5000, 6000]})
        screen = pd.DataFrame(
            {"date": common_dates, "screen_time_before_bed_min": [20, 30]}
        )
        lifestyle = pd.DataFrame(
            {
                "date": common_dates,
                "alcohol_units": [0, 1],
                "caffeine_after_noon": [1, 0],
            }
        )

        sp = str(tmp_path / "s.csv")
        stp = str(tmp_path / "st.csv")
        sc = str(tmp_path / "sc.csv")
        lp = str(tmp_path / "l.csv")
        sleep.to_csv(sp, index=False)
        steps.to_csv(stp, index=False)
        screen.to_csv(sc, index=False)
        lifestyle.to_csv(lp, index=False)

        df = load_and_merge_data(sp, stp, sc, lp)

        assert len(df) == len(common_dates)
        assert extra_date not in df["date"].values

    def test_load_and_merge_drops_rows_with_missing_values(self, tmp_path):
        """
        Any row whose merged result contains a NaN must be removed before
        the DataFrame is returned (R-3).
        """
        sleep = pd.DataFrame(
            {
                "date": ["2023-05-01", "2023-05-02"],
                "awake_min": [10, np.nan],
                "light_min": [60, 65],
                "deep_min": [50, 55],
                "rem_min": [40, 45],
            }
        )
        steps = pd.DataFrame(
            {"date": ["2023-05-01", "2023-05-02"], "steps": [5000, 6000]}
        )
        screen = pd.DataFrame(
            {
                "date": ["2023-05-01", "2023-05-02"],
                "screen_time_before_bed_min": [20, 30],
            }
        )
        lifestyle = pd.DataFrame(
            {
                "date": ["2023-05-01", "2023-05-02"],
                "alcohol_units": [0, 1],
                "caffeine_after_noon": [1, 0],
            }
        )

        sp = str(tmp_path / "s.csv")
        stp = str(tmp_path / "st.csv")
        sc = str(tmp_path / "sc.csv")
        lp = str(tmp_path / "l.csv")
        sleep.to_csv(sp, index=False)
        steps.to_csv(stp, index=False)
        screen.to_csv(sc, index=False)
        lifestyle.to_csv(lp, index=False)

        df = load_and_merge_data(sp, stp, sc, lp)

        assert df.isnull().sum().sum() == 0
        assert len(df) == 1  # only 2023-05-01 has no NaN


# ---------------------------------------------------------------------------
# compute_sleep_quality_score
# ---------------------------------------------------------------------------


class TestComputeSleepQualityScore:
    def test_compute_sleep_quality_score_appends_column(self, merged_df):
        """
        The function must return a pd.DataFrame that contains the new
        sleep_quality_score column alongside all original columns (R-4).
        """
        result = compute_sleep_quality_score(merged_df.copy())

        assert isinstance(result, pd.DataFrame)
        assert "sleep_quality_score" in result.columns
        for col in REQUIRED_COLUMNS:
            assert col in result.columns

    def test_compute_sleep_quality_score_formula(self):
        """
        sleep_quality_score must equal
        0.4*deep_min + 0.3*rem_min + 0.2*light_min - 0.1*awake_min
        for every row (R-5).
        """
        df = pd.DataFrame(
            {
                "date": ["2023-01-01", "2023-01-02"],
                "awake_min": [30.0, 20.0],
                "light_min": [120.0, 130.0],
                "deep_min": [90.0, 80.0],
                "rem_min": [60.0, 70.0],
                "steps": [8000, 10000],
                "screen_time_before_bed_min": [45, 30],
                "alcohol_units": [1, 0],
                "caffeine_after_noon": [1, 0],
            }
        )
        result = compute_sleep_quality_score(df.copy())

        expected = (
            0.4 * df["deep_min"]
            + 0.3 * df["rem_min"]
            + 0.2 * df["light_min"]
            - 0.1 * df["awake_min"]
        )
        pd.testing.assert_series_equal(
            result["sleep_quality_score"].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
        )


# ---------------------------------------------------------------------------
# fit_model
# ---------------------------------------------------------------------------


class TestFitModel:
    def test_fit_model_returns_ndarray_of_four_coefficients(self, scored_df):
        """
        fit_model must return a 2-element tuple whose first element is a
        NumPy ndarray of exactly four standardized regression coefficients
        matching an independent linear regression fit on the same standardized
        data (R-6, R-7, R-8).
        """
        result = fit_model(scored_df)

        assert isinstance(result, tuple), "fit_model must return a tuple"
        assert len(result) == 2, "fit_model tuple must have exactly 2 elements"

        coefficients, _ = result

        assert isinstance(coefficients, np.ndarray)
        assert coefficients.shape == (4,)

        # Independently compute expected coefficients from the same data.
        # This verifies both the standardization (zero mean / unit variance)
        # and the linear regression fit simultaneously.
        X = scored_df[FEATURE_NAMES].values
        y = scored_df["sleep_quality_score"].values
        X_scaled = StandardScaler().fit_transform(X)
        expected_coef = LinearRegression().fit(X_scaled, y).coef_

        assert np.allclose(coefficients, expected_coef, atol=1e-6), (
            f"Coefficients do not match a linear regression fit on standardized "
            f"features. Got {coefficients}, expected {expected_coef}."
        )

    def test_fit_model_returns_correct_feature_names(self, scored_df):
        """
        The second element of the returned tuple must be the ordered list of
        the four feature names exactly as specified in the prompt (R-9).
        """
        _, feature_names = fit_model(scored_df)

        assert feature_names == FEATURE_NAMES


# ---------------------------------------------------------------------------
# rank_and_summarize
# ---------------------------------------------------------------------------


class TestRankAndSummarize:
    def test_rank_and_summarize_output_format(self):
        """
        Each of the four ranked lines must match '<rank>. <feature_name>: <coef>'
        with exactly 4 decimal places, ranks must be sequential 1-based integers
        1–4, all four distinct feature names must appear, and the final line must
        be one of the two mandated summary formats (R-10, R-11).
        """
        # steps most negative overall; alcohol_units only negative habit
        coefficients = np.array([-0.5, 0.3, -0.1, 0.2])
        feature_names = list(FEATURE_NAMES)

        output = _capture_stdout(rank_and_summarize, coefficients, feature_names)
        lines = output.strip().splitlines()

        assert len(lines) == 5, (
            f"Expected exactly 5 output lines (4 ranked + 1 summary), got {len(lines)}"
        )

        ranked_lines = lines[:4]
        line_re = re.compile(r"^\d+\. ([^\s:]+): (-?\d+\.\d{4})$")
        for line in ranked_lines:
            assert line_re.match(line), (
                f"Line does not match '<rank>. <name>: <coef (4 dp)>' format: {line!r}"
            )

        # Rank numbers must be the sequential 1-based integers 1, 2, 3, 4.
        extracted_ranks = [int(line.split(".")[0]) for line in ranked_lines]
        assert extracted_ranks == [1, 2, 3, 4], (
            f"Expected sequential 1-based rank numbers [1, 2, 3, 4], "
            f"got: {extracted_ranks}"
        )

        # All four distinct feature names must appear exactly once (R-10).
        extracted_names = [line_re.match(line).group(1) for line in ranked_lines]
        assert set(extracted_names) == set(feature_names), (
            f"Expected all 4 distinct feature names in ranked output, "
            f"got: {extracted_names}"
        )

        # Each ranked line must pair the correct feature name with its own
        # coefficient value (R-10).  Build a lookup from the input arrays and
        # verify every output token matches.
        coef_map = dict(zip(feature_names, coefficients))
        for line in ranked_lines:
            m = line_re.match(line)
            name, coef_str = m.group(1), m.group(2)
            assert name in coef_map, f"Unknown feature name in output: {name!r}"
            assert float(coef_str) == pytest.approx(coef_map[name], abs=5e-5), (
                f"Coefficient mismatch for '{name}': output {coef_str!r}, "
                f"expected {coef_map[name]:.4f}"
            )

        # With alcohol_units as the only negative habit (-0.1), the correct
        # summary is deterministic and must be asserted exactly (R-14).
        summary_line = lines[-1]
        assert summary_line == "Most regrettable habit: alcohol_units", (
            f"Expected 'Most regrettable habit: alcohol_units' (most-negative habit "
            f"given the inputs), got: {summary_line!r}"
        )

    def test_rank_and_summarize_ascending_order(self):
        """
        Features must be printed in ascending order of their coefficient value
        (the feature with the most negative coefficient appears first) (R-12).
        """
        coefficients = np.array([0.4, -0.8, 0.1, -0.3])
        feature_names = list(FEATURE_NAMES)

        output = _capture_stdout(rank_and_summarize, coefficients, feature_names)
        ranked_lines = output.strip().splitlines()[:4]

        assert len(ranked_lines) == 4, (
            f"Expected exactly 4 ranked output lines, got {len(ranked_lines)}"
        )

        # Build expected (coef, name) pairs in ascending coefficient order so
        # both the sort order AND the feature-coefficient pairing can be verified
        # in a single pass (R-12).
        sorted_pairs = sorted(zip(coefficients.tolist(), feature_names))
        line_re = re.compile(r"^\d+\. ([^\s:]+): (-?\d+\.\d+)$")
        for (expected_coef, expected_name), line in zip(sorted_pairs, ranked_lines):
            m = line_re.match(line)
            assert m, f"Could not parse ranked line: {line!r}"
            assert m.group(1) == expected_name, (
                f"Expected feature '{expected_name}' at this rank, got '{m.group(1)}'"
            )
            assert float(m.group(2)) == pytest.approx(expected_coef, abs=5e-5), (
                f"Coefficient mismatch for '{expected_name}': "
                f"got {m.group(2)!r}, expected {expected_coef:.4f}"
            )

    def test_rank_and_summarize_coefficient_rounded_to_four_places(self):
        """
        Coefficient values in the output must be printed with exactly 4 decimal
        places as specified in the prompt (R-11).
        """
        coefficients = np.array([-0.123456, 0.654321, -0.111111, 0.222222])
        feature_names = list(FEATURE_NAMES)

        output = _capture_stdout(rank_and_summarize, coefficients, feature_names)
        ranked_lines = output.strip().splitlines()[:4]

        assert len(ranked_lines) == 4, (
            f"Expected exactly 4 ranked output lines, got {len(ranked_lines)}"
        )

        # Build the expected rounded strings from the input coefficients and
        # sort them ascending so the order matches the ranked output.  Comparing
        # each output token directly against the expected string verifies both
        # the 4-decimal-place format AND that rounding (not truncation) was used
        # (R-11).
        expected_rounded = sorted(
            [f"{c:.4f}" for c in coefficients], key=float
        )
        coef_re = re.compile(r"(-?\d+\.\d{4})$")
        coef_tokens = []
        for line in ranked_lines:
            m = coef_re.search(line)
            assert m, f"No 4-decimal coefficient found in: {line!r}"
            coef_tokens.append(m.group(1))
        assert coef_tokens == expected_rounded, (
            f"Coefficients not correctly rounded to 4 decimal places: "
            f"got {coef_tokens}, expected {expected_rounded}"
        )

    def test_rank_and_summarize_negative_habit_summary(self):
        """
        When at least one habit feature has a negative coefficient, the summary
        line must be exactly 'Most regrettable habit: <feature_name>' naming
        the habit feature with the most negative coefficient (R-14).
        """
        # screen_time_before_bed_min is the most-negative habit
        coef_map = {
            "steps": 0.5,
            "screen_time_before_bed_min": -0.9,
            "alcohol_units": -0.3,
            "caffeine_after_noon": 0.1,
        }
        feature_names = list(FEATURE_NAMES)
        coefficients = np.array([coef_map[f] for f in feature_names])

        output = _capture_stdout(rank_and_summarize, coefficients, feature_names)
        summary_line = output.strip().splitlines()[-1]

        assert summary_line == "Most regrettable habit: screen_time_before_bed_min", (
            f"Expected exact summary line, got: {summary_line!r}"
        )

    def test_rank_and_summarize_steps_excluded_from_habit_summary(self):
        """
        The summary must consider only the three habit features. Even when
        steps has the most negative overall coefficient, the summary must name
        the most-negative habit feature instead (R-13).
        """
        # steps is most negative overall, screen_time_before_bed_min is most
        # negative among the three habit features
        coef_map = {
            "steps": -0.9,
            "screen_time_before_bed_min": -0.4,
            "alcohol_units": 0.2,
            "caffeine_after_noon": 0.1,
        }
        feature_names = list(FEATURE_NAMES)
        coefficients = np.array([coef_map[f] for f in feature_names])

        output = _capture_stdout(rank_and_summarize, coefficients, feature_names)
        summary_line = output.strip().splitlines()[-1]

        assert summary_line == "Most regrettable habit: screen_time_before_bed_min", (
            f"Expected most-negative habit (screen_time_before_bed_min) named in "
            f"summary (not steps), got: {summary_line!r}"
        )

    def test_rank_and_summarize_no_negative_habit_summary(self):
        """
        When none of the three habit features has a negative coefficient, the
        summary must be exactly 'No lifestyle habit showed a negative
        association in this model.' (R-15).
        """
        # All habit features are non-negative; steps alone is negative
        coef_map = {
            "steps": -0.5,
            "screen_time_before_bed_min": 0.2,
            "alcohol_units": 0.4,
            "caffeine_after_noon": 0.1,
        }
        feature_names = list(FEATURE_NAMES)
        coefficients = np.array([coef_map[f] for f in feature_names])

        output = _capture_stdout(rank_and_summarize, coefficients, feature_names)
        summary_line = output.strip().splitlines()[-1]

        assert summary_line == "No lifestyle habit showed a negative association in this model.", (
            f"Expected no-negative-habit message, got: {summary_line!r}"
        )

    def test_rank_and_summarize_returns_none(self):
        """
        rank_and_summarize must produce output to stdout AND return None as
        specified by the prompt's interface definition (R-10, R-16).
        """
        coefficients = np.array([0.1, 0.2, 0.3, 0.4])
        feature_names = list(FEATURE_NAMES)

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            result = rank_and_summarize(coefficients, feature_names)

        output = buf.getvalue()
        assert len(output.strip()) > 0, (
            "rank_and_summarize must write output to stdout but produced nothing"
        )
        assert result is None


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


class TestMain:
    def test_main_produces_ranked_output(self):
        """
        main() must run without errors using the default data file paths and
        write a 4-line ranked feature list (1-based sequential ranks, coefficients
        rounded to 4 decimal places) plus a valid summary line to stdout.
        The summary must name only one of the three habit features; steps must
        never appear as the regrettable habit (R-11, R-12, R-13, R-17, R-18).
        """
        output = _capture_stdout(main)
        lines = [line for line in output.strip().splitlines() if line.strip()]

        assert len(lines) == 5, (
            f"Expected exactly 5 output lines from main(), got {len(lines)}"
        )

        # Re-derive the expected coefficient map independently from the same
        # data files so the test can verify the exact (feature, coefficient)
        # pairing in every ranked output line — not merely that some ascending
        # numbers appear (R-10, R-11, R-12).
        default_paths = (
            os.path.join("data", "sleep_stages.csv"),
            os.path.join("data", "step_counts.csv"),
            os.path.join("data", "screen_time.csv"),
            os.path.join("data", "lifestyle_logs.csv"),
        )
        _df = compute_sleep_quality_score(load_and_merge_data(*default_paths))
        _coefs, _fnames = fit_model(_df)
        coef_map = dict(zip(_fnames, _coefs))

        # Each ranked line must carry exactly 4 decimal places (R-11).
        line_re = re.compile(r"^\d+\. ([^\s:]+): (-?\d+\.\d{4})$")
        for line in lines[:4]:
            assert line_re.match(line), (
                f"Ranked line from main() does not match expected format "
                f"'<rank>. <name>: <coef (4 dp)>': {line!r}"
            )

        # Rank numbers must be the sequential 1-based integers 1, 2, 3, 4.
        extracted_ranks = [int(line.split(".")[0]) for line in lines[:4]]
        assert extracted_ranks == [1, 2, 3, 4], (
            f"Expected sequential 1-based rank numbers [1, 2, 3, 4], "
            f"got: {extracted_ranks}"
        )

        # Every ranked line must pair the correct feature name with its own
        # coefficient value, and lines must be in ascending coefficient order
        # (R-10, R-12).
        sorted_pairs = sorted(coef_map.items(), key=lambda kv: kv[1])
        for (expected_name, expected_coef), line in zip(sorted_pairs, lines[:4]):
            m = line_re.match(line)
            assert m.group(1) == expected_name, (
                f"Expected feature '{expected_name}' at this rank, got '{m.group(1)}'"
            )
            assert float(m.group(2)) == pytest.approx(expected_coef, abs=5e-5), (
                f"Coefficient mismatch for '{expected_name}': "
                f"got {m.group(2)!r}, expected {expected_coef:.4f}"
            )

        # Summary must name the habit feature with the most negative coefficient,
        # or the no-negative-habit message when all habit coefficients are >= 0
        # (R-13, R-14, R-15).
        no_negative_summary = "No lifestyle habit showed a negative association in this model."
        summary_line = lines[-1]
        negative_habit_coefs = {f: coef_map[f] for f in HABIT_FEATURES if coef_map[f] < 0}
        if negative_habit_coefs:
            expected_habit = min(negative_habit_coefs, key=negative_habit_coefs.get)
            assert summary_line == f"Most regrettable habit: {expected_habit}", (
                f"Expected 'Most regrettable habit: {expected_habit}' "
                f"(most-negative habit from real data), got: {summary_line!r}"
            )
        else:
            assert summary_line == no_negative_summary, (
                f"Summary line from main() does not match either required format: "
                f"{summary_line!r}"
            )


