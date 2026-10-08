import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file (update the filename)
df = pd.read_csv("training_metrics.csv")  # Replace with actual file
# df = df[df['Iteration'] < 600]   # Remove the first iteration
# Compute the average for each iteration

df_avg = df.groupby('iteration', as_index=False).agg({'throughput(MBs)': 'mean', ' iteration_time': 'mean'})

# Plot the results
plt.figure(figsize=(15, 8))
plt.plot( df_avg[' iteration_time'],df_avg['throughput(MBs)'], marker ='o', linestyle='-', label='DALI object detection')

# Set x and y axis to start from 0
plt.ylim(0, df_avg['throughput(MBs)'].max())  # Set x-axis to start from 0
plt.xlim(0, df_avg[' iteration_time'].max())  # Set y-axis to start from 0

plt.legend()
plt.xlabel("Time (s)")
plt.ylabel("Throughput (MB/s)")
# plt.title("NObject Detection")
plt.savefig("throughput_log_dali.png")  # Save the plot to a file
plt.show()
