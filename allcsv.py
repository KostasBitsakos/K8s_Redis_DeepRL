import pandas as pd
import glob

# Set the path to your CSV files
path = 'metrics/'  # Adjust this to the path of your CSV files
all_files = glob.glob(path + "metrics*.csv")

li = []

for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    li.append(df)

frame = pd.concat(li, axis=0, ignore_index=True)

# Save the combined dataframe to a new CSV file
frame.to_csv('combined_metrics.csv', index=False)

