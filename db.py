
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# 1. Load dataset
df = pd.read_csv( r"C:\Users\srish\OneDrive\Desktop\AiML\k_clustering\Mall_Customers.csv")

# 2. Select features
X = df[["Annual Income (k$)", "Spending Score (1-100)"]]

# 3. Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Create and train DBSCAN model
model = DBSCAN(eps=0.5, min_samples=5)
df["Cluster"] = model.fit_predict(X_scaled)

# 5. Display results
print(df[["Annual Income (k$)",
          "Spending Score (1-100)",
          "Cluster"]])

# 6. Count customers in each cluster
print(df["Cluster"].value_counts())

# 7. Visualize clusters
plt.scatter(
    X["Annual Income (k$)"],
    X["Spending Score (1-100)"],
    c=df["Cluster"],
    cmap="rainbow"
)

plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("Customer Segmentation Using DBSCAN")
plt.show()


# Remove noise points (-1)
labels = df["Cluster"]
mask = labels != -1

# Calculate silhouette score
if len(set(labels[mask])) > 1:
    score = silhouette_score(
        X_scaled[mask],
        labels[mask]
    )
    print("Silhouette Score:", score)
else:
    print("Silhouette Score cannot be calculated")
