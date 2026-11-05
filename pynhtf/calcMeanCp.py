#!/usr/bin/env python3
import argparse
import os
import glob
import pandas as pd

def main():
    parser = argparse.ArgumentParser(
        description="Compute mean Cp for turbine CSV files."
    )
    parser.add_argument(
        "--tmin",
        type=float,
        default=None,
        help="Start time for averaging. Ignored if --finalRevolution is used."
    )
    parser.add_argument(
        "--finalRevolution",
        action="store_true",
        help="Average Cp over the final 360 degrees of angle_deg."
    )
    parser.add_argument(
        "--base",
        type=str,
        default="postProcessing/turbines/0",
        help="Directory containing turbine CSV files."
    )

    args = parser.parse_args()
    directory = args.base

    csv_files = sorted(glob.glob(os.path.join(directory, "*.csv")))
    if not csv_files:
        print(f"No CSV files found in {directory}")
        return

    for csv_path in csv_files:
        fname = os.path.basename(csv_path)
        print(f"\nCalculating the mean Cp for {fname}.")

        df = pd.read_csv(csv_path)

        if "cp" not in df.columns:
            print(f"  Skipping {fname}: missing Cp column.")
            continue

        # -------------------------------
        # Case 1: Final revolution mode
        # -------------------------------
        if args.finalRevolution:
            if "angle_deg" not in df.columns:
                print(f"  Skipping {fname}: missing angle_deg column required for --finalRevolution.")
                continue

            max_angle = df["angle_deg"].max()
            start_angle = max_angle - 360.0

            mask = (df["angle_deg"] > start_angle) & (df["angle_deg"] <= max_angle)
            df_sel = df[mask]

            if df_sel.empty:
                print("  No rows found in the final revolution.")
                continue

            n_rows = len(df_sel)
            mean_cp = df_sel["cp"].mean()

            print(f"Averaging {n_rows} Cp values from angle {start_angle:.2f}° to final angle {max_angle:.2f}°.")
            print(f"Mean Cp = {mean_cp}")
            continue

        # -------------------------------
        # Case 2: Time-based averaging
        # -------------------------------
        if "time" not in df.columns:
            print(f"  Skipping {fname}: missing time column.")
            continue

        if args.tmin is None:
            # Use only rows at the maximum time
            t_end = df["time"].max()
            t_start = t_end
            mask = df["time"] == t_end
        else:
            t_start = args.tmin
            t_end = df["time"].max()
            mask = df["time"] >= t_start

        df_sel = df[mask]

        if df_sel.empty:
            print("  No rows found in the selected time range.")
            continue

        n_rows = len(df_sel)
        mean_cp = df_sel["cp"].mean()

        print(f"Averaging {n_rows} Cp values from time {t_start} to end time {t_end}.")
        print(f"Mean Cp = {mean_cp}")

if __name__ == "__main__":
    main()

