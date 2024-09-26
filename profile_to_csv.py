import pstats
import pandas as pd

# ChatGPT gnerated code to convert the profile file to CSV

# Load the profiling data from the pstats file
profiler = pstats.Stats("profile_results.pstats")

# Convert the profiling data to a pandas DataFrame
data = []
for func, (cc, nc, tt, ct, callers) in profiler.stats.items():
    data.append({
        'Function': func,
        'Call Count': cc,
        'Non-Recursive Call Count': nc,
        'Total Time (s)': tt,
        'Cumulative Time (s)': ct
    })

df = pd.DataFrame(data)

# Export the profiling data to a CSV file
csv_file = "profile_results.csv"
df.to_csv(csv_file, index=False)

print(f"Profiling results exported to {csv_file}")
