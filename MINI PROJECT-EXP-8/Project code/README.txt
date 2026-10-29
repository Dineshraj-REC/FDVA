PUBLIC TRANSPORT USAGE ANALYTICS - RUN GUIDE

1. Open a terminal in this folder (the folder containing transport_analytics.py).
2. Install dependencies:
   python -m pip install -r requirements.txt
3. Run:
   python transport_analytics.py

The script reads passenger_data.csv from the same folder and creates an outputs folder.
Charts are saved as PNG files and summary tables are saved as CSV files.

To use your own data, replace passenger_data.csv while preserving these column names:
Date, Route, Time_Period, Passengers, Day_Type

Important: the included data is simulated sample data for demonstration.
