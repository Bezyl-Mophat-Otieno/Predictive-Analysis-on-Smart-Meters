import pandas as pd
import numpy as np
from pathlib import Path

# Original data provided
data = {
    'Time(s)': [0.07] * 25,
    'Current(A)': [8.02, 13.43, 6.45, 2.04, 11.96, 10.91, 3.56, 7.12, 11.58, 10.3, 
                        0.47, 8.02, 13.14, 7.21, 1.9, 11.39, 11.39, 3.84, 6.36, 11.72, 
                        10.34, 0.33, 8.73, 12.81, 6.97],
    'Voltage(V)': [228.36, 254.57, 244.435, 234.58, 228.36, 251.46, 239.91, 231.47, 245.24, 247.46, 
                   237.25, 228.8, 254.57, 245.24, 235.02, 226.58, 251.91, 240.36, 231.91, 242.13, 
                   247.91, 237.69, 228.36, 254.17, 244.35],
    'Power(W)': [1831.4472, 3418.8751, 1576.60575, 478.5432, 2731.1856, 2743.4286, 854.0796, 1648.0664, 
                 2839.8792, 2548.838, 111.5075, 1834.976, 3345.0498, 1768.1804, 446.538, 2580.7462, 
                 2869.2549, 922.9824, 1474.9476, 2837.7636, 2563.3894, 78.4377, 1993.5828, 3255.9177, 1703.1195]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Generate Datetime column (assuming 1 second intervals starting from now)
df['Datetime'] = pd.date_range(start=pd.Timestamp.now(), periods=len(df), freq='S')

# Rename columns to match required format
df.rename(columns={'Power(W)': 'Power', 'Current(A)': 'Current', 
                   'Voltage(V)': 'Voltage'}, inplace=True)

# Estimating Load_Kitchen and Load_Laundry as a percentage of the total Power
df['Load_Kitchen'] = df['Power'] * np.random.uniform(0.3, 0.5, size=len(df))
df['Load_Laundry'] = df['Power'] * np.random.uniform(0.2, 0.4, size=len(df))

# Reorder columns to match the required output
df = df[['Datetime', 'Power', 'Voltage', 'Current', 'Load_Kitchen', 'Load_Laundry']]

# Define output path
output_folder = Path(r'C:\Users\BezylMophatOtieno\source\repos\household_power_consumption_predictive_analysis\scripts\LINEAR\data_preparation\data')
output_folder.mkdir(parents=True, exist_ok=True)
output_path = output_folder / 'manipulated_smart_meter_data.csv'

# Save the DataFrame to a CSV file
df.to_csv(output_path, index=False)

print(f"Data saved to {output_path}")