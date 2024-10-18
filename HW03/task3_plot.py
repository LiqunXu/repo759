import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

# ts values: 2^1, 2^2, ..., 2^10
ts_values = [2**i for i in range(1, 11)]
times = []

# Read data from "task3.out" file, assuming that the output is in the required format
with open("task3.out") as f:
    for _ in ts_values:
        f.readline()  # Title line
        times.append(float(f.readline()))  # Time taken
        f.readline()  # First element
        f.readline()  # Last element
        f.readline()  # Empty line

# Generate the plot with linear–log scale
with PdfPages("task3_ts.pdf") as pdf:
    plt.figure()
    plt.plot(ts_values, times, "o-", label="Time vs ts")

    # Set x-axis to log scale (logarithmic ts values)
    plt.xscale('log')

    # Labeling the plot
    plt.xlabel("Threshold (ts) - log scale")
    plt.ylabel("Time (milliseconds)")
    plt.title("Task3: Time vs Threshold (ts) for n=10^6, t=8")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)

    # Save the plot to a PDF file
    pdf.savefig()
    plt.close()

print("Plot saved as 'task3_ts.pdf'.")
