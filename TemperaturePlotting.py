import re
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime


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


def plot_temperature(entries, title: str = "CPU Temperature"):
    if not entries:
        print("No data to plot.")
        return

    colors = [ "#3a9ee0","#e05c3a", "#6ce03a", "#e0c03a", "#a03ae0"]
    labels = ["Test 1 at ambiant temperature", "Test 2 at 40°C without ventillation", "Test 3 at 40°C with ventillation"]

    fig, ax = plt.subplots(figsize=(10, 5))

    stats_lines = []

    for i, session in enumerate(entries):
        color = colors[i % len(colors)]
        timestamps, temperatures = zip(*session)

        ax.plot(timestamps, temperatures, linewidth=1.5, color=color, label=f"{labels[i]}")
        #ax.fill_between(timestamps, temperatures, alpha=0.15, color=color)

        duration = timestamps[-1] - timestamps[0]
        avg_temp = sum(temperatures) / len(temperatures)
        max_temp = max(temperatures)
        min_temp = min(temperatures)

        stats_lines.append(
            f"Test {i+1} — Duration: {duration:.0f}s  |  "
            f"Min: {min_temp:.1f}°C  |  "
            f"Max: {max_temp:.1f}°C"
        )

    plt.axhline(80, color='red', linestyle='--', label='Throttling temperature threshold')

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Temperature (°C)")
    ax.legend(loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.5)

    stats_text = "\n".join(stats_lines)
    n = len(stats_lines)
    fig.text(0.5, 0.01, stats_text, ha="center", fontsize=9, color="gray")

    plt.tight_layout(rect=[0, 0.03 * n, 1, 1])
    plt.show()


# --- Run ---

# pour le test clim avec 50 -> 80 -> shutdown
# "Runs\\TestsTemperatureRasppi\cpu_temperature.txt", start_line=133, Loaded 378 entries
# pour le test clim avec 50 -> 80 -> ventil: 
# "Runs\\TestsTemperatureRasppi\cpu_temperature.txt", start_line=978, Loaded 1043 entries

LOG_FILE = "Codes\\cpu_temperature.log"
start_line=45
entries3 = parse_log(LOG_FILE, start_line)
print(f"Loaded {len(entries3)} entries 3 from line {start_line} ended on line {start_line + len(entries3)}")

LOG_FILE = "Runs\\TestsTemperatureRasppi\cpu_temperature.txt"

# Parse the recording session starting at line 2 (line 1 is the '#' header)
start_line=158
entries1 = parse_log(LOG_FILE, start_line)
print(f"Loaded {len(entries1)} entries 1 from line {start_line}, ended on line {start_line + len(entries1)}")

start_line=1145
entries2 = parse_log(LOG_FILE, start_line)
print(f"Loaded {len(entries2)} entries 2 from line {start_line} ended on line {start_line + len(entries2)}")
plot_temperature([entries3, entries1, entries2[:740]], title="Raspberry Pi CPU Temperature")

