# **Household Power Consumption Predictive Analysis**

## **Project Overview**

This project aims to develop predictive models for forecasting global active power consumption in households using historical data. It involves the application of both Linear Regression and LSTM (Long Short-Term Memory) networks for prediction, along with an analysis of power costs based on ongoing rates. The project includes data preprocessing, model training, evaluation, and visualization of results.

## **Folder Structure**

The project is organized into several directories, each containing scripts, datasets, models, and outputs relevant to different aspects of the analysis:

```
📁 scripts
├── 📁 backup
│   ├── household_power_consumption.txt
│   └── model_predictions.csv
├── 📁 LINEAR
│   ├── 📁 datasets
│   │   └── household_power_consumption.txt
│   ├── 📁 testing
│   │   └── 📁 predictions
│   │       └── predicted_costs_with_statistics.csv
│   │   └── index.py
│   ├── 📁 training
│   │   ├── 📁 data_collection
│   │   │   ├── data_preprocessing.py
│   │   │   ├── exploratory_data_analysis.py
│   │   │   └── load_data.py
│   │   ├── 📁 data_preparation
│   │   │   └── data_splitting.py
│   │   ├── 📁 outputs
│   │   │   ├── 📁 cost_statistics
│   │   │   │   ├── cost_statistics.png
│   │   │   │   ├── daily_cost_statistics.csv
│   │   │   │   ├── hourly_cost_statistics.csv
│   │   │   │   └── monthly_cost_statistics.csv
│   │   │   ├── 📁 data
│   │   │   │   ├── evaluation_metrics.txt
│   │   │   │   ├── manipulated_smart_meter_data.csv
│   │   │   │   ├── model_predictions.csv
│   │   │   │   ├── preprocessed_household_power_consumption.csv
│   │   │   │   ├── raw_household_power_consumption.csv
│   │   │   │   ├── X_test.csv
│   │   │   │   ├── X_train.csv
│   │   │   │   ├── y_test.csv
│   │   │   │   └── y_train.csv
│   │   │   ├── 📁 model
│   │   │   │   └── linear_regression_model.pkl
│   │   │   ├── 📁 plots
│   │   │   │   ├── daily_statistics.csv
│   │   │   │   ├── energy_consumption_statistics.png
│   │   │   │   ├── hourly_statistics.csv
│   │   │   │   ├── monthly_statistics.csv
│   │   │   ├── 📁 power_statistics
│   │   │   │   ├── daily_power_statistics.csv
│   │   │   │   ├── hourly_power_statistics.csv
│   │   │   │   ├── monthly_power_statistics.csv
│   │   │   │   └── power_statistics.png
│   │   │   └── 📁 price_statistics
│   │   │       ├── daily_price_statistics.csv
│   │   │       ├── hourly_price_statistics.csv
│   │   │       ├── monthly_price_statistics.csv
│   │   │       └── price_statistics.png
│   │   ├── build_model.py
│   │   ├── evaluate_model.py
│   │   ├── predicted_data_analysis.py
│   │   ├── predicted_power_stats.py
│   │   └── predicted_price_stats.py
├── 📁 LTSM
│   ├── 📁 data_collection
│   │   ├── data_exploration.py
│   │   ├── data_inspection.py
│   │   ├── modifying_dataset.py
│   │   ├── reading_dataset.py
│   │   └── test-data-splitting.py
│   ├── 📁 datasets
│   │   ├── household_power_consumption_cleaned.csv
│   │   ├── household_power_consumption_features_test_data.csv
│   │   ├── household_power_consumption_features.csv
│   │   ├── household_power_consumption_original.csv
│   │   ├── household_power_consumption_scaled_features.csv
│   │   ├── household_power_consumption_scaled_target.csv
│   │   ├── household_power_consumption_scaled_test_features.csv
│   │   ├── household_power_consumption_scaled_test_target.csv
│   │   ├── household_power_consumption_sequences.npz
│   │   └── predictions_with_time.csv
│   ├── 📁 feature-engineering
│   │   ├── data-scalling-test-data.py
│   │   ├── data-scalling.py
│   │   └── feature-selection.py
│   ├── 📁 models
│   │   ├── lstm_model.keras
│   │   ├── lstm_model1.h5
│   │   └── lstm_new_model.h5
│   ├── 📁 outputs
│   │   ├── 📁 data
│   │   │   ├── manipulated_smart_meter_data.csv
│   │   │   ├── predictions.csv
│   │   │   ├── test_features.csv
│   │   │   ├── test_scaled_features.csv
│   │   │   ├── test_scaled_target.csv
│   │   │   ├── test_target.csv
│   │   │   ├── train_features.csv
│   │   │   ├── train_scaled_features.csv
│   │   │   ├── train_scaled_target.csv
│   │   │   └── train_target.csv
│   │   ├── 📁 images
│   │   │   ├── correlation_matrix.png
│   │   │   ├── global_active_power_over_time.png
│   │   │   └── histograms.png
│   │   ├── 📁 plots
│   │   │   ├── actual_vs_predicted_over_time.png
│   │   │   ├── actual_vs_predicted.png
│   │   │   ├── daily_consumption.png
│   │   │   ├── daily_trend.png
│   │   │   ├── hourly_consumption.png
│   │   │   ├── hourly_trend.png
│   │   │   ├── residual_plot.png
│   │   │   ├── residuals_histogram.png
│   │   │   └── weekly_trend.png
│   │   └── 📁 sequences
│   │       ├── test_sequences.npz
│   │       └── train_sequences.npz
│   └── 📁 scalers
│       ├── scaler_X.pkl
│       ├── scaler_y.pkl
│       ├── test-scaler_X.pkl
│       └── test-scaler_y.pkl
│   └── 📁 training
│       ├── model-building.py
│       ├── model-evaluation.py
│       └── sequence-preparation.py
```

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

   - Ensure you have the correct paths for datasets and scalers in your scripts.
   - Run the preprocessing scripts to prepare the data for model training.

2. **Train and Evaluate Models:**

   - For Linear Regression:
     ```bash
     python scripts/LINEAR/build_model.py
     python scripts/LINEAR/evaluate_model.py
     ```

   - For LSTM:
     ```bash
     python scripts/LTSM/model-building.py
     python scripts/LTSM/model-evaluation.py
     ```

3. **Generate Predictions and Visualizations:**

   - Run the respective scripts to analyze predicted data and generate visualizations.
   - Example for LSTM:
     ```bash
     python scripts/LTSM/predicted_data_analysis.py
     ```

## **Results**

- **Predictions:**
  - Predictions and evaluation metrics are saved in `datasets/predictions.csv` for further analysis.
