# scripts/data_inspection.py

import pandas as pd

# Load the dataset
data_path = r"C:\Users\BezylMophatOtieno\source\repos\household_power_consumption_predictive_analysis\datasets\household_power_consumption_original.csv"
data = pd.read_csv(data_path, sep=',', low_memory=False, na_values=['nan', '?'])

print(f"Data shape: {data.shape}")


# Display the first few rows of the dataset
print(data.head())

# Check for missing values
print(f"Missing values: \n{data.isnull().sum()}")

# Check data types
print(f"Data types: \n{data.dtypes}")

# Inspect non-numeric values in numeric columns
numeric_columns = ['Global_active_power', 'Global_reactive_power', 'Voltage', 'Global_intensity', 'Sub_metering_1', 'Sub_metering_2']
for column in numeric_columns:
    non_numeric_values = data[~data[column].apply(lambda x: str(x).replace('.', '', 1).isdigit())][column].unique()
    if len(non_numeric_values) > 0:
        print(f"Non-numeric values in {column}: {non_numeric_values}")

# Convert data types
for column in numeric_columns:
    data[column] = pd.to_numeric(data[column], errors='coerce')

# Check for missing values again after conversion
print(f"Missing values after conversion: \n{data.isnull().sum()}")

# Handle missing values (drop rows with any missing values)
data_cleaned = data.dropna()

#Count the dropped missing values
print(f"Missing values after dropping: \n{data_cleaned.isnull().sum()}")

# Confirm the data types
print(f"Data types after cleaning: \n{data_cleaned.dtypes}")

# Save the cleaned data for further use
output_path = r'C:\Users\BezylMophatOtieno\source\repos\household_power_consumption_predictive_analysis\datasets\household_power_consumption_cleaned.csv'
data_cleaned.to_csv(output_path, index=False)
