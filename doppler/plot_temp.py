#!/usr/bin/env python3

"""
plot_temperature.py

Reads a CSV file containing columns:

epoch,t1,t2,t3,hum1,hum2,hum3,Vbus

The file may or may not include the header line.

Plots temperature vs time with timestamps converted
from Unix epoch (UTC) to US Pacific time.
"""

import sys
import pandas as pd
import matplotlib.pyplot as plt

EXPECTED_COLUMNS = [
    "epoch","t1","t2","t3","hum1","hum2","hum3","Vbus"
]


def read_csv_flexible(filename):

    # First try reading normally
    df = pd.read_csv(filename)

    # If header missing, pandas will assign numeric column names
    if list(df.columns) != EXPECTED_COLUMNS:

        df = pd.read_csv(
            filename,
            header=None,
            names=EXPECTED_COLUMNS
        )

    return df


def main():

    if len(sys.argv) < 2:
        print("Usage: python plot_temperature.py data.csv")
        sys.exit(1)

    filename = sys.argv[1]

    df = read_csv_flexible(filename)

    # Convert epoch -> Pacific time
    df["time"] = pd.to_datetime(df["epoch"], unit="s", utc=True)
    df["time"] = df["time"].dt.tz_convert("America/Los_Angeles")

    # Plot
    plt.figure(figsize=(10,5))

    plt.plot(df["time"], df["t1"], label="t1")
    plt.plot(df["time"], df["t2"], label="t2")
    plt.plot(df["time"], df["t3"], label="t3")

    plt.xlabel("Time (US Pacific)")
    plt.ylabel("Temperature (°C)")
    plt.title("Temperature vs Time")
    plt.legend()
    plt.grid(True)

    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
    