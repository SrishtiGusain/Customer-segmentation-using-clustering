# Customer-segmentation-using-clustering
📌 Project Overview

This project uses Unsupervised Machine Learning to group mall customers based on their annual income and spending score.

The goal is to identify customer segments that can help businesses understand customer behavior and develop better marketing strategies.

This project explores two clustering algorithms:

K-Means Clustering: Divides customers into a predefined number of clusters.

DBSCAN Clustering: Groups customers based on data density and identifies noise points.

🎯 Objectives

Analyze mall customer data using Pandas.

Explore customer income and spending patterns.

Apply feature scaling using StandardScaler.

Group customers using clustering algorithms.

Visualize customer segments using Matplotlib.

Evaluate clustering quality using the Silhouette Score.

Predict the cluster of a new customer using the trained K-Means model.

📂 Project Structure

Mall-Customers-Clustering/
│
├── mall_customers.csv       # Dataset containing mall customer details
├── k.py                     # K-Means clustering implementation
├── db.py                    # DBSCAN clustering implementation
├── requirements.txt         # Required Python libraries
└── README.md                # Project documentation

📊 Dataset Description

The project uses the Mall Customers dataset.

Column

Description

CustomerID

Unique customer identifier

Gender

Customer's gender

Age

Customer's age

Annual Income (k$)

Annual income in thousands of dollars

Spending Score (1-100)

Spending score assigned to the customer

The clustering scripts use these two features:

Annual Income (k$)

Spending Score (1-100)

🛠️ Technologies Used

Python

Pandas

NumPy

Matplotlib

Scikit-learn

⚙️ Algorithms Used

1. K-Means Clustering (k.py)

K-Means groups customers into a predefined number of clusters.

Example configuration:

model = KMeans(
    n_clusters=5,
    random_state=42,
    n_init=10
)

The model assigns each customer to the nearest cluster centroid. The script can also scale a new customer's features and predict its cluster using the trained model.

2. DBSCAN Clustering (db.py)

DBSCAN discovers groups based on the density of nearby data points.

Example configuration:

model = DBSCAN(
    eps=0.5,
    min_samples=5
)

eps: Neighborhood distance used to find nearby points.

min_samples: Minimum number of points in an eps neighborhood for a point to qualify as a core point.

-1: Label used for noise points.

Unlike K-Means, DBSCAN does not require the number of clusters to be specified beforehand.

📈 Project Workflow

Load the dataset using Pandas.

Explore the data using head(), describe(), and isnull().sum().

Select annual income and spending score.

Visualize customers before clustering.

Scale the selected features using StandardScaler.

Train K-Means or DBSCAN.

Assign cluster labels to customers.

Visualize the resulting customer segments.

Evaluate clustering quality using the Silhouette Score when applicable.

Save the clustered dataset if implemented in the script.

Use k.py to predict the cluster of a new customer with K-Means.

📉 Evaluation: Silhouette Score

The Silhouette Score measures how well-separated the clusters are.

Close to +1: Clusters are generally well separated.

Close to 0: Clusters overlap or are close together.

Below 0: Some points may be assigned to unsuitable clusters.

For DBSCAN, noise points can be excluded from the score, provided at least two clusters remain and the score is otherwise valid.

🚀 How to Run the Project

Step 1: Open the project folder

Open Mall-Customers-Clustering in your terminal or VS Code.

Step 2: Install the required libraries

pip install -r requirements.txt

Step 3: Run K-Means

python k.py

Step 4: Run DBSCAN

python db.py

Make sure mall_customers.csv, k.py, and db.py are in the same folder, or update the CSV path in the scripts.

📦 Requirements

Your requirements.txt file should contain:

pandas
numpy
matplotlib
scikit-learn

Install the dependencies using:

pip install -r requirements.txt

💡 Key Learnings

Understanding unsupervised machine learning.

Comparing K-Means and DBSCAN.

Data preprocessing and feature scaling.

Cluster visualization and interpretation.

Silhouette Score evaluation.

Predicting cluster membership with K-Means.

🔮 Future Improvements

Compare different values of n_clusters in K-Means.

Experiment with DBSCAN's eps and min_samples.

Explore additional customer features.

Build an interactive dashboard using Streamlit.

Add more detailed customer segment descriptions.
