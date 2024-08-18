Sure! Here's a detailed README for your project:

---

# **Household Power Consumption Predictive Analysis**

## **Project Overview**

This project involves developing a predictive model for forecasting global active power consumption in households based on historical data. The model uses Long Short-Term Memory (LSTM) networks to predict energy usage patterns and estimate associated costs based on ongoing power rates. The project includes data preprocessing, model training, evaluation, and visualization of predictions.

## **Project Goals**

- **Predict Energy Consumption:** Forecast global active power usage using historical data.
- **Estimate Costs:** Calculate the total price of consumed power based on ongoing power rates.
- **Analyze Patterns:** Understand energy consumption patterns and cost implications across different days of the week and hours of the day.

## **Features**

- **Data Preprocessing:**
  - Scaled and transformed features and target variables.
  - Engineered additional features including daily, weekly, and monthly averages, time-related features, and lag features.

- **Model Development:**
  - Utilized an LSTM model for time-series forecasting.
  - Trained the model on preprocessed data with a sequence length of 24 hours.

- **Model Evaluation:**
  - Assessed model performance using Mean Squared Error (MSE), Mean Absolute Error (MAE), and Root Mean Squared Error (RMSE).
  - Analyzed prediction errors and their implications.

- **Predictions and Visualization:**
  - Generated predictions for global active power.
  - Estimated power costs using ongoing rates.
  - Created line plots, residual plots, and histograms to visualize prediction accuracy and errors.

## **Installation**

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/household_power_consumption_predictive_analysis.git
   ```

2. **Navigate to the Project Directory:**

   ```bash
   cd household_power_consumption_predictive_analysis
   ```

3. **Install Dependencies:**

   Ensure you have Python installed. Install required packages using `pip`:

   ```bash
   pip install -r requirements.txt
   ```

   **`requirements.txt`** includes:
   - pandas
   - numpy
   - tensorflow
   - joblib
   - scikit-learn
   - matplotlib
   - seaborn

## **Usage**

1. **Preprocess Data:**
   - Ensure you have the scaled test data and saved scalers as specified.
   - Adjust paths in the script as needed.

2. **Run the Model Evaluation Script:**

   ```bash
   python scripts/model_evaluation.py
   ```

   This script will:
   - Load the scaled test data and saved scalers.
   - Reshape the data for LSTM input.
   - Load the trained model and make predictions.
   - Evaluate the model and save predictions.
   - Generate and save visualizations of actual vs. predicted values, residuals, and residual distributions.

3. **View Results:**
   - Predictions will be saved to `datasets/predictions.csv`.
   - Visualizations will be saved to `outputs/plots`.

## **File Structure**

```
household_power_consumption_predictive_analysis/
│
```
└── 📁scripts
    └── 📁backup
        └── household_power_consumption.txt
        └── model_predictions.csv
    └── 📁LINEAR
        └── 📁datasets
            └── household_power_consumption.txt
        └── 📁testing
            └── 📁predictions
                └── predicted_costs_with_statistics.csv
            └── index.py
        └── 📁training
            └── 📁data_collection
                └── data_preprocessing.py
                └── exploratory_data_analysis.py
                └── load_data.py
            └── 📁data_preparation
                └── data_splitting.py
            └── 📁outputs
                └── 📁cost_statistics
                    └── cost_statistics.png
                    └── daily_cost_statistics.csv
                    └── hourly_cost_statistics.csv
                    └── monthly_cost_statistics.csv
                └── 📁data
                    └── evaluation_metrics.txt
                    └── manipulated_smart_meter_data.csv
                    └── model_predictions.csv
                    └── preprocessed_household_power_consumption.csv
                    └── raw_household_power_consumption.csv
                    └── X_test.csv
                    └── X_train.csv
                    └── y_test.csv
                    └── y_train.csv
                └── 📁model
                    └── linear_regression_model.pkl
                └── 📁plots
                    └── daily_statistics.csv
                    └── energy_consumption_statistics.png
                    └── hourly_statistics.csv
                    └── monthly_statistics.csv
                └── 📁power_statistics
                    └── daily_power_statistics.csv
                    └── hourly_power_statistics.csv
                    └── monthly_power_statistics.csv
                    └── power_statistics.png
                └── 📁price_statistics
                    └── daily_price_statistics.csv
                    └── hourly_price_statistics.csv
                    └── monthly_price_statistics.csv
                    └── price_statistics.png
            └── build_model.py
            └── evaluate_model.py
            └── predicted_data_analysis.py
            └── predicted_power_stats.py
            └── predicted_price_stats.py
    └── 📁LTSM
        └── 📁data_collection
            └── data_exploration.py
            └── data_inspection.py
            └── modifying_dataset.py
            └── reading_dataset.py
            └── test-data-splitting.py
        └── 📁datasets
            └── household_power_consumption_cleaned.csv
            └── household_power_consumption_features_test_data.csv
            └── household_power_consumption_features.csv
            └── household_power_consumption_original.csv
            └── household_power_consumption_scaled_features.csv
            └── household_power_consumption_scaled_target.csv
            └── household_power_consumption_scaled_test_features.csv
            └── household_power_consumption_scaled_test_target.csv
            └── household_power_consumption_sequences.npz
            └── household_power_consumption.txt
            └── predictions_with_time.csv
            └── predictions.csv
        └── 📁feature-engineering
            └── data-scalling-test-data.py
            └── data-scalling.py
            └── feature-selection.py
        └── 📁models
            └── lstm_model.keras
            └── lstm_model1.h5
            └── lstm_new_model.h5
        └── 📁outputs
            └── 📁data
                └── manipulated_smart_meter_data.csv
                └── predictions.csv
                └── test_features.csv
                └── test_scaled_features.csv
                └── test_scaled_target.csv
                └── test_target.csv
                └── train_features.csv
                └── train_scaled_features.csv
                └── train_scaled_target.csv
                └── train_target.csv
            └── 📁images
                └── correlation_matrix.png
                └── global_active_power_over_time.png
                └── histograms.png
            └── 📁plots
                └── actual_vs_predicted_over_time.png
                └── actual_vs_predicted.png
                └── daily_consumption.png
                └── daily_trend.png
                └── hourly_consumption.png
                └── hourly_trend.png
                └── residual_plot.png
                └── residuals_histogram.png
                └── weekly_trend.png
            └── 📁sequences
                └── test_sequences.npz
                └── train_sequences.npz
        └── 📁scalers
            └── scaler_X.pkl
            └── scaler_y.pkl
            └── test-scaler_X.pkl
            └── test-scaler_y.pkl
        └── 📁training
            └── model-building.py
            └── model-evaluation.py
            └── sequence-preparation.py
```
└── README.md
```

## **Future Work**

- **Further Model Optimization:**
  - Tune hyperparameters for improved model performance.
  - Explore additional features or sequence lengths for better predictions.

- **Deployment Preparation:**
  - Integrate the model into a user-facing application.
  - Develop a UI for users to input new data and view predictions and cost estimates.

## **Contributing**

If you'd like to contribute to this project, please fork the repository and submit a pull request with your changes.
