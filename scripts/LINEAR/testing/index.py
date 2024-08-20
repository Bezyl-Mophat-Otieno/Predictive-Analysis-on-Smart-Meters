import pandas as pd
import numpy as np
from datetime import datetime
import joblib
import os



def get_user_input():
    """
    Get user input for start date, end date, granularity, and price per kWh.
    
    :return: Tuple containing start_date, end_date, granularity, and price_per_kwh
    """
    start_date = input("Enter the start date (YYYY-MM-DD): ")
    end_date = input("Enter the end date (YYYY-MM-DD): ")
    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_date = datetime.strptime(end_date, "%Y-%m-%d")
    granularity = input("Enter the granularity (hourly/daily/monthly): ").strip().lower()
    price_per_kwh = float(input("Enter the electricity price per kWh: "))
    return start_date, end_date, granularity, price_per_kwh

def generate_features(start_date, end_date, granularity, average_values):
    """
    Generate a DataFrame with features for prediction based on the given date range and granularity.
    
    :param start_date: Start date for the date range
    :param end_date: End date for the date range
    :param granularity: Granularity of the data ('hourly' or 'daily')
    :param average_values: Dictionary with average values for Current, Voltage, and Load
    :return: DataFrame with generated features
    """
    freq = 'h' if granularity == 'hourly' else 'd'
    date_range = pd.date_range(start=start_date, end=end_date, freq=freq)
    df = pd.DataFrame(date_range, columns=['Datetime'])
    df['Current'] = average_values['Average_Current']
    df['Voltage'] = average_values['Average_Voltage']
    df['Load(Wh)'] = average_values['Average_Load(Wh)']
    return df

def load_model(model_path):
    """
    Load the prediction model from the given path.
    
    :param model_path: Path to the model file
    :return: Loaded model
    """
    model = joblib.load(model_path)
    return model

def load_statistics(output_dir):
    """
    Load the aggregated statistics from the specified directory.
    
    :param output_dir: Directory containing the statistics files
    :return: Tuple containing hourly_stats, daily_stats, and monthly_stats DataFrames
    """
    hourly_stats = pd.read_csv(os.path.join(output_dir, 'hourly_power_statistics.csv'), index_col=0)
    daily_stats = pd.read_csv(os.path.join(output_dir, 'daily_power_statistics.csv'), index_col=0)
    monthly_stats = pd.read_csv(os.path.join(output_dir, 'monthly_power_statistics.csv'), index_col=0)
    return hourly_stats, daily_stats, monthly_stats

def load_average_values(average_values_file):
    """
    Load the average values for Current, Voltage, and Load(Wh) from the saved file.
    
    :param average_values_file: Path to the average values CSV file
    :return: Dictionary with average values
    """
    avg_df = pd.read_csv(average_values_file)
    averages = avg_df.iloc[0].to_dict()
    return averages

def refine_prediction(features_df, model, hourly_stats, daily_stats, monthly_stats, granularity):
    """
    Refine predictions based on the model and aggregated statistics.
    
    :param features_df: DataFrame containing features for prediction
    :param model: Trained model for predictions
    :param hourly_stats: DataFrame with hourly statistics
    :param daily_stats: DataFrame with daily statistics
    :param monthly_stats: DataFrame with monthly statistics
    :param granularity: Granularity of the data ('hourly', 'daily', or 'monthly')
    :return: Refined predictions
    """
    X = features_df[['Current', 'Voltage', 'Load(Wh)']]
    predictions = model.predict(X)
    
    if granularity == 'hourly':
        features_df['Hour'] = features_df['Datetime'].dt.hour
        avg_adjustment = hourly_stats.loc[features_df['Hour'], 'Actual_kWh_mean'].values
    elif granularity == 'daily':
        features_df['DayOfWeek'] = features_df['Datetime'].dt.dayofweek
        avg_adjustment = daily_stats.loc[features_df['DayOfWeek'], 'Actual_kWh_mean'].values
    else:
        features_df['Month'] = features_df['Datetime'].dt.month
        avg_adjustment = monthly_stats.loc[features_df['Month'], 'Actual_kWh_mean'].values
    
    predictions = predictions + avg_adjustment
    return predictions

def calculate_cost(predicted_power, price_per_kwh):
    """
    Calculate the cost based on predicted power and price per kWh.
    
    :param predicted_power: Array of predicted power values
    :param price_per_kwh: Price per kWh
    :return: Array of estimated costs
    """
    cost = predicted_power * price_per_kwh
    return cost

def aggregate_results(predicted_power, estimated_cost):
    """
    Aggregate total power consumption and total cost from the predictions.
    
    :param predicted_power: Array of predicted power values
    :param estimated_cost: Array of estimated costs
    :return: Tuple containing total consumption and total cost
    """
    total_consumption = predicted_power.sum()
    total_cost = estimated_cost.sum()
    return total_consumption, total_cost

def output_results(features_df, predicted_power, estimated_cost, output_path):
    """
    Output the results to a CSV file.
    
    :param features_df: DataFrame containing the features and predictions
    :param predicted_power: Array of predicted power values
    :param estimated_cost: Array of estimated costs
    :param output_path: Path to save the results CSV file
    :return: None
    """
    features_df['Predicted_Power'] = predicted_power
    features_df['Estimated_Cost'] = estimated_cost
    features_df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")

def print_summary(total_consumption, total_cost):
    """
    Print a summary report of total power consumption and cost.
    
    :param total_consumption: Total power consumption
    :param total_cost: Total cost
    :return: None
    """
    border = "+" + "-" * 46 + "+"
    header = "|{:^46}|".format(" SUMMARY REPORT ")
    separator = "+" + "-" * 46 + "+"
    consumption_str = "| Total Power Consumption {:>13.2f} kWh |".format(total_consumption)
    cost_str = "| Total Cost                KSH{:>13.2f}      |".format(total_cost)

    print("\n" + border)
    print(header)
    print(separator)
    print(consumption_str)
    print(cost_str)
    print(border + "\n")

def main():
    # Constants
    MODEL_PATH = r'C:\Users\BezylMophatOtieno\source\repos\household_power_consumption_predictive_analysis\scripts\LINEAR\training\outputs\model\linear_regression_model.pkl'
    OUTPUT_DIR = r'C:\Users\BezylMophatOtieno\source\repos\household_power_consumption_predictive_analysis\scripts\LINEAR\training\outputs\plots\power_statistics'
    RESULTS_PATH = r'C:\Users\BezylMophatOtieno\source\repos\household_power_consumption_predictive_analysis\scripts\LINEAR\testing\predictions\predicted_costs_with_statistics.csv'
    AVERAGE_VALUES_FILE = r'C:\Users\BezylMophatOtieno\source\repos\household_power_consumption_predictive_analysis\scripts\LINEAR\testing\predictions\average_values.csv'
    start_date, end_date, granularity, price_per_kwh = get_user_input()
    average_values = load_average_values(AVERAGE_VALUES_FILE)
    features_df = generate_features(start_date, end_date, granularity, average_values)
    
    model = load_model(MODEL_PATH)
    hourly_stats, daily_stats, monthly_stats = load_statistics(OUTPUT_DIR)
    
    predicted_power = refine_prediction(features_df, model, hourly_stats, daily_stats, monthly_stats, granularity)
    estimated_cost = calculate_cost(predicted_power, price_per_kwh)
    
    # Aggregate results
    total_consumption, total_cost = aggregate_results(predicted_power, estimated_cost)
    
    # Print summary to the console with enhanced design
    print_summary(total_consumption, total_cost)
    
    # Output results to a CSV file
    output_results(features_df, predicted_power, estimated_cost, RESULTS_PATH)

if __name__ == "__main__":
    main()
