"""
Public Transport Usage Analytics System
College mini project: descriptive analysis and visualization.
The bundled passenger dataset is simulated demonstration data.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "passenger_data.csv"
OUTPUT_DIR = BASE_DIR / "outputs"

REQUIRED_COLUMNS = ["Date", "Route", "Time_Period", "Passengers", "Day_Type"]


def load_data(path: Path) -> pd.DataFrame:
    """Load the dataset and validate its required columns."""
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            "Place passenger_data.csv next to transport_analytics.py."
        )

    df = pd.read_csv(path)
    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(
            "Dataset is missing required column(s): " + ", ".join(missing_columns)
        )

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Passengers"] = pd.to_numeric(df["Passengers"], errors="coerce")
    df["Route"] = df["Route"].astype("string").str.strip()
    df["Time_Period"] = df["Time_Period"].astype("string").str.strip().str.title()
    df["Day_Type"] = df["Day_Type"].astype("string").str.strip().str.title()
    return df


def inspect_data(df: pd.DataFrame) -> None:
    """Print a concise overview of the raw dataset."""
    print("\n" + "=" * 68)
    print("PUBLIC TRANSPORT USAGE ANALYTICS SYSTEM")
    print("=" * 68)
    print("\n[1] DATASET OVERVIEW")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")
    print("\nFirst five records:")
    print(df.head().to_string(index=False))
    print("\nColumn information:")
    df.info()
    print("\nMissing values by column:")
    print(df.isna().sum().to_string())
    print(f"\nDuplicate rows: {df.duplicated().sum()}")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Remove invalid records and duplicates for reliable descriptive analysis."""
    cleaned = df.drop_duplicates().copy()
    cleaned = cleaned.dropna(subset=REQUIRED_COLUMNS)
    cleaned = cleaned[cleaned["Passengers"] >= 0]
    cleaned = cleaned[cleaned["Route"].ne("")]
    cleaned["Passengers"] = cleaned["Passengers"].astype(int)
    cleaned["Date"] = pd.to_datetime(cleaned["Date"])
    return cleaned.sort_values(["Date", "Route", "Time_Period"]).reset_index(drop=True)


def print_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate central tendency and dispersion statistics."""
    values = df["Passengers"]
    modes = values.mode()
    summary = pd.DataFrame(
        {
            "Metric": [
                "Record count",
                "Total passengers (sum of observations)",
                "Mean passengers per record",
                "Median passengers per record",
                "Mode passenger count",
                "Minimum passengers",
                "Maximum passengers",
                "Range",
                "Variance (sample)",
                "Standard deviation (sample)",
            ],
            "Value": [
                int(values.count()),
                int(values.sum()),
                round(float(values.mean()), 2),
                round(float(values.median()), 2),
                ", ".join(str(int(x)) for x in modes.tolist()) if not modes.empty else "N/A",
                int(values.min()),
                int(values.max()),
                int(values.max() - values.min()),
                round(float(values.var()), 2),
                round(float(values.std()), 2),
            ],
        }
    )

    print("\n[2] DESCRIPTIVE STATISTICS")
    print(summary.to_string(index=False))
    return summary


def create_summaries(df: pd.DataFrame):
    """Build route, time-period, daily, and day-type summary tables."""
    route_summary = (
        df.groupby("Route", as_index=False)
        .agg(
            Total_Passengers=("Passengers", "sum"),
            Average_Passengers=("Passengers", "mean"),
            Median_Passengers=("Passengers", "median"),
            Observation_Count=("Passengers", "count"),
        )
        .sort_values("Total_Passengers", ascending=False)
    )
    route_summary["Average_Passengers"] = route_summary["Average_Passengers"].round(2)

    period_summary = (
        df.groupby("Time_Period", as_index=False)["Passengers"]
        .mean()
        .rename(columns={"Passengers": "Average_Passengers"})
    )
    period_summary["Average_Passengers"] = period_summary["Average_Passengers"].round(2)

    daily_summary = (
        df.groupby("Date", as_index=False)["Passengers"]
        .sum()
        .sort_values("Date")
    )

    day_type_summary = (
        df.groupby("Day_Type", as_index=False)["Passengers"]
        .mean()
        .rename(columns={"Passengers": "Average_Passengers"})
    )
    day_type_summary["Average_Passengers"] = day_type_summary["Average_Passengers"].round(2)

    return route_summary, period_summary, daily_summary, day_type_summary


def create_charts(df, route_summary, daily_summary):
    """Save the required bar chart, line chart, and histogram."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Bar chart: total observed passengers by route.
    plt.figure(figsize=(9, 5))
    plt.bar(route_summary["Route"], route_summary["Total_Passengers"])
    plt.title("Total Passenger Observations by Route")
    plt.xlabel("Bus Route")
    plt.ylabel("Sum of Passenger Counts")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "route_comparison.png", dpi=160)
    plt.close()

    # 2. Line chart: daily sum of passenger counts across all routes and periods.
    plt.figure(figsize=(11, 5))
    plt.plot(daily_summary["Date"], daily_summary["Passengers"], marker=".", linewidth=1.5)
    plt.title("Daily Passenger Count Trend")
    plt.xlabel("Date")
    plt.ylabel("Sum of Passenger Counts")
    plt.xticks(rotation=35)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "daily_passenger_trend.png", dpi=160)
    plt.close()

    # 3. Histogram: distribution of passenger counts per record.
    plt.figure(figsize=(9, 5))
    plt.hist(df["Passengers"], bins=20, edgecolor="black")
    plt.title("Distribution of Passenger Counts per Record")
    plt.xlabel("Passengers per Route and Time Period")
    plt.ylabel("Frequency")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "passenger_distribution.png", dpi=160)
    plt.close()


def print_insights(df, route_summary, period_summary, daily_summary, day_type_summary):
    """Print data-derived observations without making unsupported real-world claims."""
    print("\n[3] KEY FINDINGS")
    busiest_route = route_summary.iloc[0]
    busiest_period = period_summary.loc[period_summary["Average_Passengers"].idxmax()]
    busiest_day = daily_summary.loc[daily_summary["Passengers"].idxmax()]
    print(
        f"Route with the largest total passenger-count sum: "
        f"{busiest_route['Route']} ({int(busiest_route['Total_Passengers']):,})."
    )
    print(
        f"Time period with the highest average per-record count: "
        f"{busiest_period['Time_Period']} "
        f"({busiest_period['Average_Passengers']:.2f})."
    )
    print(
        f"Date with the largest combined passenger-count sum: "
        f"{busiest_day['Date'].date()} ({int(busiest_day['Passengers']):,})."
    )
    if len(day_type_summary) >= 2:
        print("\nAverage passenger count per record by day type:")
        print(day_type_summary.to_string(index=False))
    print(
        "\nInterpretation note: results describe the supplied dataset only. "
        "The bundled data is simulated and is not official transport data."
    )


def main():
    try:
        raw_df = load_data(DATA_FILE)
        inspect_data(raw_df)
        df = clean_data(raw_df)

        if df.empty:
            raise ValueError("No valid records remain after data validation.")

        print(f"\nValid records used for analysis: {len(df):,}")
        print(f"Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
        print(f"Routes found: {df['Route'].nunique()}")

        statistical_summary = print_statistics(df)
        route_summary, period_summary, daily_summary, day_type_summary = create_summaries(df)
        create_charts(df, route_summary, daily_summary)
        print_insights(df, route_summary, period_summary, daily_summary, day_type_summary)

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        route_summary.to_csv(OUTPUT_DIR / "route_summary.csv", index=False)
        statistical_summary.to_csv(OUTPUT_DIR / "statistical_summary.csv", index=False)
        period_summary.to_csv(OUTPUT_DIR / "time_period_summary.csv", index=False)
        daily_summary.to_csv(OUTPUT_DIR / "daily_summary.csv", index=False)
        day_type_summary.to_csv(OUTPUT_DIR / "day_type_summary.csv", index=False)

        print("\n[4] OUTPUT FILES")
        for output_file in sorted(OUTPUT_DIR.iterdir()):
            print(f"- {output_file.name}")
        print(f"\nAll outputs saved to: {OUTPUT_DIR}")
        print("Analysis completed successfully.")

    except (FileNotFoundError, ValueError, pd.errors.ParserError) as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
