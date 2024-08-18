import pandas as pd
import numpy as np
from datetime import datetime
import joblib
import os

def get_user_input():
    start_date = input("Enter the start date (YYYY-MM-DD): ")
    end_date = input("Enter the end date (YYYY-MM-DD): ")
    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_date = datetime.strptime(end_date, "%Y-%m-%d")
    granularity = input("Enter the granularity (hourly/daily): ").strip().lower()
    price_per_kwh = float(input("Enter the electricity price per kWh: "))
    return start_date, end_date, granularity, price_per_kwh

def generate_features(start_date, end_date, granularity):
    if granularity == 'hourly':
        date_range = pd.date_range(start=start_date, end=end_date, freq='h')
    else:
        date_range = pd.date_range(start=start_date, end=end_date, freq='d')
    
    df = pd.DataFrame(date_range, columns=['Datetime'])
    df['Current'] = 10
    df['Voltage'] = 240
    df['Load_1'] = 500
    df['Load_2'] = 300
    return df

def load_model(model_path):
    model = joblib.load(model_path)
    return model

def load_statistics(output_dir):
    hourly_stats = pd.read_csv(os.path.join(output_dir, 'hourly_statistics.csv'), index_col=0)
    daily_stats = pd.read_csv(os.path.join(output_dir, 'daily_statistics.csv'), index_col=0)
    monthly_stats = pd.read_csv(os.path.join(output_dir, 'monthly_statistics.csv'), index_col=0)
    return hourly_stats, daily_stats, monthly_stats

def refine_prediction(features_df, model, hourly_stats, daily_stats, monthly_stats, granularity):
    X = features_df[['Current', 'Voltage', 'Load_1', 'Load_2']]
    predictions = model.predict(X)
    
    if granularity == 'hourly':
        features_df['Hour'] = features_df['Datetime'].dt.hour
        avg_adjustment = hourly_stats.loc[features_df['Hour'], 'Actual_mean'].values
    elif granularity == 'daily':
        features_df['DayOfWeek'] = features_df['Datetime'].dt.dayofweek
        avg_adjustment = daily_stats.loc[features_df['DayOfWeek'], 'Actual_mean'].values
    else:
        features_df['Month'] = features_df['Datetime'].dt.month
        avg_adjustment = monthly_stats.loc[features_df['Month'], 'Actual_mean'].values
    
    predictions = predictions + avg_adjustment
    return predictions

def calculate_cost(predicted_power, price_per_kwh):
    cost = predicted_power * price_per_kwh
    return cost

def aggregate_results(features_df, predicted_power, estimated_cost):
    total_consumption = predicted_power.sum()
    total_cost = estimated_cost.sum()
    return total_consumption, total_cost

def output_results(features_df, predicted_power, estimated_cost, output_path):
    features_df['Predicted_Power'] = predicted_power
    features_df['Estimated_Cost'] = estimated_cost
    features_df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")

def print_summary(total_consumption, total_cost):
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
    start_date, end_date, granularity, price_per_kwh = get_user_input()
    features_df = generate_features(start_date, end_date, granularity)
    
    model_path = r'C:\Users\BezylMophatOtieno\source\repos\household_power_consumption_predictive_analysis\scripts\LINEAR\training\outputs\model\linear_regression_model.pkl'
    model = load_model(model_path)
    
    output_dir = r'C:\Users\BezylMophatOtieno\source\repos\household_power_consumption_predictive_analysis\scripts\LINEAR\training\outputs\plots'
    hourly_stats, daily_stats, monthly_stats = load_statistics(output_dir)
    
    predicted_power = refine_prediction(features_df, model, hourly_stats, daily_stats, monthly_stats, granularity)
    estimated_cost = calculate_cost(predicted_power, price_per_kwh)
    
    # Aggregate results
    total_consumption, total_cost = aggregate_results(features_df, predicted_power, estimated_cost)
    
    # Print summary to the console with enhanced design
    print_summary(total_consumption, total_cost)
    
    # Output results to a CSV file
    output_path = r'C:\Users\BezylMophatOtieno\source\repos\household_power_consumption_predictive_analysis\scripts\LINEAR\testing\predictions\predicted_costs_with_statistics.csv'
    output_results(features_df, predicted_power, estimated_cost, output_path)

if __name__ == "__main__":
    main()
