import pandas as pd
from sklearn.model_selection import train_test_split

# Load your cleaned dataset
data_path = r"C:\Users\BezylMophatOtieno\source\repos\household_power_consumption_predictive_analysis\datasets\household_power_consumption_features.csv"
data_cleaned = pd.read_csv(data_path, sep=',', low_memory=False)

# Split into train and test sets
train_data, test_data = train_test_split(data_cleaned, test_size=0.2, random_state=42)

# Save the test data
test_data_path = r'C:\Users\BezylMophatOtieno\source\repos\household_power_consumption_predictive_analysis\datasets\household_power_consumption_features_test_data.csv'
test_data.to_csv(test_data_path, index=False)

print(f"Test data saved to {test_data_path}")
