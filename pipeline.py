import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

FEATURE_NAMES = [
    "steps",
    "screen_time_before_bed_min",
    "alcohol_units",
    "caffeine_after_noon",
]

HABIT_FEATURES = {"screen_time_before_bed_min", "alcohol_units", "caffeine_after_noon"}


def load_and_merge_data(
    sleep_stages_path: str,
    step_counts_path: str,
    screen_time_path: str,
    lifestyle_logs_path: str,
) -> pd.DataFrame:
    sleep_df = pd.read_csv(sleep_stages_path)
    steps_df = pd.read_csv(step_counts_path)
    screen_df = pd.read_csv(screen_time_path)
    lifestyle_df = pd.read_csv(lifestyle_logs_path)

    merged = sleep_df.merge(steps_df, on="date", how="inner")
    merged = merged.merge(screen_df, on="date", how="inner")
    merged = merged.merge(lifestyle_df, on="date", how="inner")
    merged = merged.dropna().reset_index(drop=True)
    return merged


def compute_sleep_quality_score(df: pd.DataFrame) -> pd.DataFrame:
    df["sleep_quality_score"] = (
        0.4 * df["deep_min"]
        + 0.3 * df["rem_min"]
        + 0.2 * df["light_min"]
        - 0.1 * df["awake_min"]
    )
    return df


def fit_model(df: pd.DataFrame) -> tuple:
    X = df[FEATURE_NAMES].values
    y = df["sleep_quality_score"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LinearRegression()
    model.fit(X_scaled, y)

    return model.coef_, FEATURE_NAMES


def rank_and_summarize(coefficients: np.ndarray, feature_names: list) -> None:
    paired = sorted(zip(coefficients, feature_names), key=lambda pair: pair[0])

    for rank, (coef, name) in enumerate(paired, start=1):
        print(f"{rank}. {name}: {coef:.4f}")

    habit_coefs = {
        name: coef
        for coef, name in paired
        if name in HABIT_FEATURES
    }
    negative_habits = {name: coef for name, coef in habit_coefs.items() if coef < 0}

    if negative_habits:
        worst = min(negative_habits, key=negative_habits.get)
        print(f"Most regrettable habit: {worst}")
    else:
        print("No lifestyle habit showed a negative association in this model.")


def main() -> None:
    df = load_and_merge_data(
        "data/sleep_stages.csv",
        "data/step_counts.csv",
        "data/screen_time.csv",
        "data/lifestyle_logs.csv",
    )
    df = compute_sleep_quality_score(df)
    coefficients, feature_names = fit_model(df)
    rank_and_summarize(coefficients, feature_names)


if __name__ == "__main__":
    main()
