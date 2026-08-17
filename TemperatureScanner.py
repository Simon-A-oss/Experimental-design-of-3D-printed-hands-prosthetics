import re
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import numpy as np



"""
=============================================================================================
Code used to scan across temperature recordings made using the "log_cpu_temp.sh" program to find useful measurements
=============================================================================================
"""

def find_all_sessions(filepath: str):
    """Retourne une liste de numéros de ligne (1-based) où commence chaque session."""
    session_starts = []
    with open(filepath, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            if line.strip().startswith("#") and "started" in line:
                session_starts.append(i + 1)  # +1 pour sauter la ligne header
    return session_starts


def plot_all_sessions(filepath: str):
    session_starts = find_all_sessions(filepath)
    print(f"Found {len(session_starts)} sessions.")

    for idx, start_line in enumerate(session_starts):
        entries = parse_log(filepath, start_line)

        if not entries:
            continue

        timestamps, temperatures = zip(*entries)
        duration = timestamps[-1] - timestamps[0]
        avg_temp = sum(temperatures) / len(temperatures)
        max_temp = max(temperatures)

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(timestamps, temperatures, linewidth=1.0, color="#3a9ee0")
        ax.axhline(80, color='red', linestyle='--', linewidth=0.8, label='Throttle threshold')
        ax.set_title(f"Session {idx+1} — line {start_line}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("°C")
        ax.set_ylim(40, 100)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend()
        ax.text(0.02, 0.95, f"Dur: {duration:.0f}s | Avg: {avg_temp:.1f}°C | Max: {max_temp:.1f}°C",
                transform=ax.transAxes, fontsize=9, verticalalignment='top', color='gray')

        plt.tight_layout()

    plt.show()  # affiche toutes les figures en une fois à la fin

def parse_log(filepath: str, start_line: int):
    pattern = re.compile(
        r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\] CPU Temp: ([\d.]+)°C"
    )

    entries = []
    start_time = None

    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    start_index = start_line - 1

    for line in lines[start_index:]:
        line = line.strip()
    
        if line.startswith("#"):
            break

        match = pattern.match(line)
        if match:
            timestamp = datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S")

            if start_time is None:
                start_time = timestamp  # première entrée = référence t=0

            elapsed = (timestamp - start_time).total_seconds()
            temperature = float(match.group(2))
            entries.append((elapsed, temperature))

    return entries


# --- Run ---
plot_all_sessions("Codes\\cpu_temperature.log")