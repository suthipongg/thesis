import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file (update the filename)
df = pd.read_csv("throughput_log_speedy_10s_2.csv")  # Replace with actual file
# df = df[df['Iteration'] < 600]   # Remove the first iteration
# Compute the average for each iteration
df_avg = df.groupby('Iteration', as_index=False).agg({'time': 'mean', '#iter/s': 'mean'})

# Plot the results
plt.figure(figsize=(15, 8))
plt.plot(df_avg['time'], df_avg['#iter/s'], marker ='o', linestyle='-', label='Speedy 10s')

# Set x and y axis to start from 0
plt.xlim(0, df_avg['time'].max())  # Set x-axis to start from 0
plt.ylim(0, df_avg['#iter/s'].max())  # Set y-axis to start from 0

plt.legend()
plt.xlabel("Time (s)")
plt.ylabel("# Iter/s")
plt.title("Number of Iterations per Second, Speech recognition (0.5s and heavy 10s)")
plt.savefig("throughput_log_speedy_10s_only.png")  # Save the plot to a file
plt.show()
