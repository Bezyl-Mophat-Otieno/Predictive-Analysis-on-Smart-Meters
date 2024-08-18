import pandas as pd

# Load the dataset
url = r'C:\Users\BezylMophatOtieno\source\repos\household_power_consumption_predictive_analysis\datasets\household_power_consumption.txt'
data = pd.read_csv(url, sep=';', parse_dates=['Date'], infer_datetime_format=True, low_memory=False)

# Save the original dataset as a CSV file
output_filename_original = r'C:\Users\BezylMophatOtieno\source\repos\household_power_consumption_predictive_analysis\datasets\household_power_consumption_original.csv'
data.to_csv(output_filename_original, index=False)
 
# Checking for missing values
missing_values = data.isnull().sum()
print(f"Missing values:\n{missing_values}")
# Checking data types
data_types = data.dtypes
print(f"Data types:\n{data_types}")



print(f"Original dataset saved as {output_filename_original}")
