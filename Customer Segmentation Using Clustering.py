#!/usr/bin/env python
# coding: utf-8

# In[7]:


# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os

# Set environment variable to avoid KMeans memory leak warning
os.environ["OMP_NUM_THREADS"] = "1"

# Step 1: Load the Dataset
data = pd.read_csv("Mall_Customers.csv")

# Step 2: Exploratory Data Analysis (EDA)
print("Dataset Head:")
print(data.head())

print("\nDataset Info:")
print(data.info())

print("\nMissing Values:")
print(data.isnull().sum())

print("\nSummary Statistics:")
print(data.describe())

# Visualize distributions
plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
sns.histplot(data['Age'], bins=20, kde=True, color='blue')
plt.title('Age Distribution')

plt.subplot(2, 2, 2)
sns.histplot(data['Annual Income (k$)'], bins=20, kde=True, color='green')
plt.title('Annual Income Distribution')

plt.subplot(2, 2, 3)
sns.histplot(data['Spending Score (1-100)'], bins=20, kde=True, color='red')
plt.title('Spending Score Distribution')

plt.subplot(2, 2, 4)
sns.countplot(x='Gender', data=data, hue='Gender', palette='viridis', legend=False)
plt.title('Gender Distribution')

plt.tight_layout()
plt.show()

# Step 3: Data Preprocessing
X = data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 4: Determine the Optimal Number of Clusters (Elbow Method)
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Plot the Elbow Method
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--', color='b')
plt.title('Elbow Method for Optimal Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

# Step 5: Apply K-Means Clustering
optimal_clusters = 5
kmeans = KMeans(n_clusters=optimal_clusters, init='k-means++', random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# Add cluster labels to the original dataset
data['Cluster'] = clusters

# Step 6: Visualize the Clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)', hue='Cluster', data=data, palette='viridis', s=100)
plt.title('Customer Segments: Annual Income vs Spending Score')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.show()

# Step 7: Interpret the Clusters
# Analyze cluster characteristics (exclude 'Gender' column)
cluster_summary = data.groupby('Cluster')[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].mean()

# Add count of customers in each cluster
cluster_summary['Count'] = data['Cluster'].value_counts().sort_index()

print("\nCluster Summary:")
print(cluster_summary)

# Step 8: Business Insights
# Example Interpretation:
# Cluster 0: High Income, Low Spending (Potential for targeted marketing)
# Cluster 1: Average Income, Average Spending (Loyal Customers)
# Cluster 2: High Income, High Spending (Premium Customers)
# Cluster 3: Low Income, High Spending (Budget-Conscious but High Spenders)
# Cluster 4: Low Income, Low Spending (Least Priority)

