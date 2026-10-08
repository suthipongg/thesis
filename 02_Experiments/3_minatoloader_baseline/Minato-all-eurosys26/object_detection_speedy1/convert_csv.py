import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file (update the filename)
df = pd.read_csv("throughput_log_speedy_10s.csv")  # Replace with actual file
df = df[df['Iteration'] <= 260]

# Compute the average for each iteration
df_avg = df.groupby('Iteration', as_index=False).agg({'time': 'mean', '#iter/s': 'mean'})


df2 = pd.read_csv("throughput_log_dali_10s.csv")  # Replace with actual file
df2 = df2[df2['Iteration'] <= 260]

# Compute the average for each iteration
df_avg2 = df2.groupby('Iteration', as_index=False).agg({'time': 'mean', '#iter/s': 'mean'})

df1 = pd.read_csv("throughput_log_pytorch1.csv")  # Replace with actual file
df1.columns = ["Iteration", "time", "#iter/s"]
df1 = df1[df1['Iteration'] <= 260]

# Compute the average for each iteration
df_avg1 = df1.groupby('Iteration', as_index=False).agg({'time': 'mean', '#iter/s': 'mean'})

# Plot the results
plt.figure(figsize=(15, 8))
plt.plot(df_avg['time'], df_avg['#iter/s'], marker ='o', linestyle='-', label='Speedy')
plt.plot(df_avg2['time'], df_avg2['#iter/s'], marker ='o',linestyle='-', label='Dali')
plt.plot(df_avg1['time'], df_avg1['#iter/s'],  marker ='o',linestyle='-', label='Pytorch')
plt.legend()
plt.xlabel("Time (s)")
plt.ylabel("# Iter/s")
plt.title("Number of Iterations per Second, Speech recognition (0.5s and heavy 10s)")
plt.savefig("throughput_log_speedy_10s.pdf")  # Save the plot to a file
plt.show()
