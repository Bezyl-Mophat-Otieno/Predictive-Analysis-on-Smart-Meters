import pandas as pd

# Load the dataset
url = r'C:\Users\BezylMophatOtieno\source\repos\household_power_consumption_predictive_analysis\datasets\household_power_consumption.txt'
data = pd.read_csv(url, sep=';', parse_dates={'datetime': ['Date', 'Time']}, infer_datetime_format=True, low_memory=False)

# Function to calculate sub-metering values
def calculate_sub_metering(appliances):
    total_energy = 0
    for appliance, specs in appliances.items():
        energy = specs['power'] * specs['duration']
        total_energy += energy
    return total_energy

# Define a function to recalculate sub-metering values for each row
def recalculate_sub_metering(row):
    # Example power ratings (in watts) and usage durations (in hours) for appliances
    # Modify these values as per your scenario
    kitchen_appliances = {
        'dishwasher': {'power': 1200, 'duration': 1},
        'oven': {'power': 2000, 'duration': 0.5},
        'microwave': {'power': 800, 'duration': 0.25}
    }
    laundry_appliances = {
        'washing_machine': {'power': 500, 'duration': 1},
        'dryer': {'power': 3000, 'duration': 0.75},
        'refrigerator': {'power': 150, 'duration': 24}
    }
    heating_cooling_appliances = {
        'water_heater': {'power': 4000, 'duration': 2},
        'air_conditioner': {'power': 3500, 'duration': 5}
    }
    
    row['Sub_metering_1'] = calculate_sub_metering(kitchen_appliances)
    row['Sub_metering_2'] = calculate_sub_metering(laundry_appliances)
    row['Sub_metering_3'] = calculate_sub_metering(heating_cooling_appliances)
    return row

# Apply the function to each row
data = data.apply(recalculate_sub_metering, axis=1)

# Save the modified dataset with the name "kenyan" appended to it
output_filename = 'household_power_consumption_kenyan.csv'
data.to_csv(output_filename, index=False)

print(f"Modified dataset saved as {output_filename}")
