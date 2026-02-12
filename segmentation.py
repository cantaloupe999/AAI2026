import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
# Use dataset
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
# Preprocess data: Select numerical features and scale them
features = ["tenure", "MonthlyCharges", "TotalCharges"]
X = df[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# Determine optimal number of clusters using elbow method
inertia = []
K = range(1, 6)
for k in K_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)
# Plot elbow curve
plt.figure(figsize=(8, 5))
plt.plot(K, inertia, 'bo-')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal K')
plt.savefig("elbow_plot.png")
plt.show()
# Apply K-Means with optimal K (e.g., 3 based on elbow method)
optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)
# Analyze clusters
cluster_summary = df.groupby('cluster')[features].mean().round(2)
print("Cluster Characteristics:")
print(cluster_summary)
# Example of targeted strategies
for cluster in range(optimal_k):
print(f"\nCluster {cluster} Strategy:")
if cluster_summary.loc[cluster, 'annual_spending'] > 1000:
print("High-spending customers: Offer exclusive promotions or loyalty
rewards.")
elif cluster_summary.loc[cluster, 'purchase_frequency'] > 10:
print("Frequent buyers: Provide bulk discounts or subscription plans.")
else:
print("Low-engagement customers: Send personalized re-engagement
campaigns.")
# Save cluster assignments to CSV
df.to_csv('customer_segments.csv', index=False)
