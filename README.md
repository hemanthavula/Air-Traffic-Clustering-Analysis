# ✈️ Air Traffic Passenger Clustering Analysis

## 📌 Project Overview
This project applies unsupervised machine learning to segment commercial airlines based on their operational scale. By utilizing K-Means clustering on air traffic passenger statistics, the analysis identifies distinct tiers of airline operations, providing data-driven insights into market dominance and air traffic distribution.

## 🛠️ Technical Stack
* **Language:** Python
* **Libraries:** `pandas`, `matplotlib`, `scikit-learn`, `sqlalchemy`, `pymysql`, `kneed`, `pickle`
* **Database:** MySQL
* **Algorithms:** K-Means Clustering

---

## 🚀 Step-by-Step Project Workflow

### Step 1: Data Ingestion & Database Integration
Instead of just reading from a static CSV, this project simulates a real-world data pipeline:
* Reads raw data from `Air_Traffic_Passenger_Statistics.csv`.
* Uses `SQLAlchemy` and `pymysql` to establish a connection to a local MySQL database (`air_routes_db`).
* Pushes the dataset into a SQL table named `airline_tbl`.
* Queries the database using `pd.read_sql_query` to pull the data back into a Pandas DataFrame for analysis.

### Step 2: Exploratory Data Analysis (EDA) & Preprocessing
* Extracted the key features needed for clustering: `Operating Airline`, `GEO Region`, and `Passenger Count`.
* Grouped the data to calculate the total number of flights held and the total passenger count for each airline.
* **Outlier Detection:** Created a scatter plot (`matplotlib`) of Flights vs. Passengers to visualize the distribution.
* **Outlier Removal:** Identified that 'United Airlines' and 'United Airlines - Pre 07/01/2013' were massive outliers skewing the scale. These were dropped from the dataset to ensure the clustering algorithm could accurately segment the rest of the market.

### Step 3: Determining the Optimal Clusters
K-Means requires specifying the number of clusters (k) beforehand. I used multiple techniques to find the best mathematical fit:
* **Elbow Method:** Calculated the Total Within-Cluster Sum of Squares (TWSS/Inertia) for `k=2` through `k=8`. Used the `kneed` library to programmatically detect the "elbow" point in the scree plot.
* **Metric Evaluation:** Because the elbow plot was mostly linear, I evaluated the clusters using advanced mathematical metrics:
  * **Silhouette Score:** To measure how similar an object is to its own cluster compared to others.
  * **Calinski-Harabasz Score:** To evaluate cluster separation.
  * **Davies-Bouldin Index:** To measure the similarity between clusters (lower is better).
* **Conclusion:** Iterating through Silhouette coefficients for different values of `k` proved that **`k=2`** was the optimal number of clusters for this specific dataset.

### Step 4: Final Model Building & Serialization
* Trained the final K-Means model using `n_clusters = 2`.
* Appended the generated `cluster_id` labels back onto the cleaned dataframe.
* **Serialization:** Saved the trained machine learning model as `Clust_.pkl` using the `pickle` library, allowing the model to be deployed or reused without retraining.
* **Data Export:** Exported the final, segmented dataset to `Air.csv` for use in BI dashboards or further reporting.

---

## 📂 Repository Files
* `Kmeans.py`: The main Python script containing the database pipeline, EDA, and clustering logic.
* `Air_Traffic_Passenger_Statistics.csv`: The raw input dataset.
* `Air.csv`: The final output dataset containing the assigned cluster IDs.
* `Clust_.pkl`: The serialized K-Means machine learning model.

## ⚙️ How to Run
1. Ensure you have MySQL installed and running locally.
2. Update the database credentials (`user`, `pw`, `db`) in the script to match your local environment.
3. Install the required dependencies: `pip install pandas matplotlib scikit-learn sqlalchemy pymysql kneed`
4. Run the Python script.
