import matplotlib.pyplot as plt

# Read results from the output file
n_values = []
times = []

with open('results.txt', 'r') as f:
    for line in f:
        if "Time taken to execute the kernel:" in line:
            time = float(line.split(':')[-1].strip().split()[0])
            times.append(time)
        elif "Running task3 with n=" in line:
            n = int(line.split('=')[-1].strip())
            n_values.append(n)

# Generate the plot
plt.figure(figsize=(10, 6))
plt.plot(n_values, times, marker='o', label='Execution Time')
plt.xlabel('Array Size (n)')
plt.ylabel('Time (ms)')
plt.title('Execution Time vs Array Size (n)')
plt.xscale('log', base=2)  # Log scale for better visualization
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend()
plt.tight_layout()
plt.savefig('task3_plot.pdf')
plt.show()
