import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# 1. Load data
df = pd.read_csv(
    r"C:\Users\srish\OneDrive\Desktop\AiML\k_clustering\Mall_Customers.csv"
)
print(df.head())
print(df.describe())
print(df.isnull().sum())

# 2. Select features
X = df[["Annual Income (k$)", "Spending Score (1-100)"]]

# 3. Before clustering
plt.scatter(X.iloc[:, 0], X.iloc[:, 1])
plt.title("Before Clustering")
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.show()

# 4. Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train the K-Means model
model = KMeans(n_clusters=5, random_state=42, n_init=10)

# Assign each customer to a cluster
labels = model.fit_predict(X_scaled)

# Store cluster assignments in the dataset
df["Cluster"] = labels

print(df["Cluster"].value_counts().sort_index())
# 6. Silhouette score
print("Silhouette Score:", silhouette_score(X_scaled, labels))

# 7. After clustering

centers = scaler.inverse_transform(model.cluster_centers_)

plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=labels, cmap="viridis")
plt.scatter(centers[:, 0], centers[:, 1],
            color="red", marker="X", s=200, label="Centroids")
plt.title("After Clustering")
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.legend()
plt.show()

# 8. Save results
df.to_csv("Mall_Customers_clustered.csv", index=False)
# 8. Enter a new customer
income = float(input("Enter Annual Income (k$): "))
score = float(input("Enter Spending Score (1-100): "))

# 9. Create a DataFrame for the new customer
new_customer = pd.DataFrame({
    "Annual Income (k$)": [income],
    "Spending Score (1-100)": [score]
})

# 10. Scale the new customer
new_customer_scaled = scaler.transform(new_customer)

# 11. Predict the cluster
cluster = model.predict(new_customer_scaled)

print("New customer belongs to Cluster:", cluster[0])