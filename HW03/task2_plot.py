import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

xticks = range(1, 20)

times = []
with open("task2.out") as f:
    for i in xticks:
        f.readline()  # Title line
        times.append(float(f.readline()))
        f.readline()  # First element
        f.readline()  # Last element
        f.readline()  # Empty line

with PdfPages("task2.pdf") as pdf:
    plt.plot(xticks, times, "o")
#    for x, y in zip(xticks, times):
#        if y > 100:
#            plt.text(x, y, y)
    plt.xlabel("Number of Threads")
    plt.ylabel("Time (milliseconds)")
    plt.title("Task2")
    plt.xticks(xticks)
    pdf.savefig()