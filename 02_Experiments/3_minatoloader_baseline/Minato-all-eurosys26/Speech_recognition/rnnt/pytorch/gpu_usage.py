import pandas as pd

# Load the CSV file
input_file = '/projects/I20240005/rnouaj/Speech_recognition/rnnt/pytorch/GPU_speech3s_pytorchb8.log'     # Replace with your actual file name
output_file = 'output.csv'   # This will store the modified version

# Read the CSV into a DataFrame
df = pd.read_csv(input_file)
Time(s)	CPU(%) GPU(%) LGPU(%)

# Ensure the necessary columns exist
required_cols = {'Time(s)', 'CPU(%)', 'GPU(%)', 'LGPU(%)'}
if not required_cols.issubset(df.columns):
    raise ValueError(f"CSV must contain the columns: {required_cols}")

# Set gpu to 0 where lgpu < 10
df.loc[df['LGPU(%)'] < 10, 'GPU(%)'] = 0

# Save the modified DataFrame
df.to_csv(output_file, index=False)

print(f"Modified CSV saved to {output_file}")
