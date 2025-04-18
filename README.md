# Email Marketing Campaign

This project is designed to streamline and enhance email marketing campaigns using data-driven insights and machine learning. It provides tools for data exploration, feature engineering, predictive modeling, and an interactive frontend for campaign optimization.

---

## Features

- **Exploratory Data Analysis (EDA):**  
  Visualize and understand email engagement patterns, including open and click rates by time, day, region, and content type.

- **Feature Engineering:**  
  Clean and merge raw data, handle anomalies, and create meaningful features such as engagement status.

- **Predictive Modeling:**  
  Train and evaluate machine learning models (Logistic Regression, XGBoost) to predict user engagement (Not Opened, Opened but Not Clicked, Clicked and Opened).

- **Optimization Metrics:**  
  Automatically compute and display the best-performing hours, days, and regions for email campaigns using model-based probability analysis.

- **Interactive Streamlit Frontend:**  
  User-friendly web app to input campaign parameters and receive real-time engagement predictions, insights, and optimization tips.

- **Campaign Analytics Dashboard:**  
  Visualize historical performance metrics and compare them with model-driven recommendations.

---

## Project Structure

```
email_marketing_campaign/
│
├── code.ipynb                  # Jupyter notebook for EDA, feature engineering, and model training
├── frontend.py                 # Streamlit app for predictions and insights
├── email_table.csv             # Raw email data
├── email_opened_table.csv      # Opened email records
├── link_clicked_table.csv      # Clicked email records
├── cleaned_data                # Cleaned dataset for modeling
├── email_engagement_model.joblib      # Trained ML model (Logistic Regression/XGBoost)
├── email_engagement_metrics.joblib    # Model-driven engagement metrics and optimization scores
└── README.md                   # Project documentation
```

---

## Data Pipeline

1. **Data Loading & Cleaning**
    - Load raw email, opened, and clicked data.
    - Merge datasets on `email_id`.
    - Fill missing values and remove anomalies (e.g., clicked but not opened).

2. **Feature Engineering**
    - Create `engagement_status` with three categories:
        - Not Opened
        - Opened but Not Clicked
        - Clicked and Opened
    - Drop low-importance features based on correlation and statistical tests.

3. **Exploratory Data Analysis**
    - Visualize open/click rates by hour, day, region, email text, and version.
    - Use scatter plots, bar plots, pair plots, and heatmaps.
    - Perform statistical tests (chi-squared) to assess feature importance.

4. **Model Training**
    - Encode categorical features and target variable.
    - Train/test split with stratification.
    - Train a Logistic Regression model (with class balancing) and optionally XGBoost.
    - Evaluate with classification report and confusion matrix.
    - Save the trained pipeline as `email_engagement_model.joblib`.

5. **Optimization Metrics**
    - Use the trained model to predict engagement probabilities for all combinations of hour, day, and region.
    - Identify and store the best-performing hours, days, and regions.
    - Save these metrics as `email_engagement_metrics.joblib`.

---

## Streamlit Frontend

- **User Inputs:**  
  Select hour, weekday, region, email text type, and version.

- **Prediction:**  
  Get real-time engagement status prediction and probability.

- **Insights:**  
  See how your choices compare to model-optimized recommendations for time, day, and region.

- **Optimization Tips:**  
  Receive actionable suggestions to maximize engagement.

- **Performance Metrics:**  
  View historical open rate, click rate, and engagement score.

---

## Installation

1. **Clone the repository:**
    ```bash
    git clone https://github.com/your-repo/email-marketing-campaign.git
    cd email-marketing-campaign
    ```

2. **Install Python dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *Required packages include: pandas, numpy, matplotlib, seaborn, scikit-learn, xgboost, streamlit, joblib.*

3. **Prepare Data:**
    - Place your raw CSV files (`email_table.csv`, `email_opened_table.csv`, `link_clicked_table.csv`) in the project directory.

4. **Run the Notebook:**
    - Open `code.ipynb` and run all cells to generate the cleaned data, train the model, and produce the metrics files.

5. **Start the Streamlit App:**
    ```bash
    streamlit run frontend.py
    ```
    - Access the app at [http://localhost:8501](http://localhost:8501).

---

## Usage

- Use the notebook for data exploration and model retraining.
- Use the Streamlit app for campaign planning and optimization.

---

## Contributions

Contributions are welcome! Please follow these steps:
1. Fork the repository.
2. Create a new branch:
    ```bash
    git checkout -b feature-name
    ```
3. Commit your changes:
    ```bash
    git commit -m "Add feature-name"
    ```
4. Push to your branch:
    ```bash
    git push origin feature-name
    ```
5. Open a pull request.

---

## Author

This project is maintained by **Gaurav**. For any inquiries, feel free to reach out.

---
