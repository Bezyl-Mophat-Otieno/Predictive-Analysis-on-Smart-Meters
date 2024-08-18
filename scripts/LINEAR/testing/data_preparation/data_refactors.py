import pandas as pd
import numpy as np
from pathlib import Path

# Define the path to the input excel file
input_excel_path = Path(r'C:\Users\BezylMophatOtieno\source\repos\household_power_consumption_predictive_analysis\scripts\LINEAR\testing\outputs\data\smart_meter_data.xlsx')

# Read the data from the CSV file
df = pd.read_excel(input_excel_path)

# Generate Datetime column (assuming 1-second intervals starting from now)
df['Datetime'] = pd.date_range(start=pd.Timestamp.now(), periods=len(df), freq='S')

# Rename columns to match the required format
df.rename(columns={'Power': 'Power', 'Current': 'Current', 'Voltage': 'Voltage'}, inplace=True)

# Estimating Load_1 and Load_2 as a percentage of the total Power
df['Load_1'] = df['Power'] * np.random.uniform(0.3, 0.5, size=len(df))
df['Load_2'] = df['Power'] * np.random.uniform(0.2, 0.4, size=len(df))

# Reorder columns to match the required output
df = df[['Datetime', 'Power', 'Voltage', 'Current', 'Load_1', 'Load_2']]

# Define output path
output_folder = Path(r'C:\Users\BezylMophatOtieno\source\repos\household_power_consumption_predictive_analysis\scripts\LINEAR\testing\outputs\data')
output_folder.mkdir(parents=True, exist_ok=True)
output_path = output_folder / 'manipulated_smart_meter_data.csv'

# Save the DataFrame to a CSV file
df.to_csv(output_path, index=False)

print(f"Data saved to {output_path}")
