Customer-Segmentation-Using-Clustering
Overview
This project segments customers using K-Means clustering on the Mall_Customers.csv dataset based on purchasing behavior, income, and age.
Requirements
Install required libraries:
pip install pandas numpy matplotlib seaborn scikit-learn
Steps
1.	Load Data: Read Mall_Customers.csv using pandas.
2.	EDA: Check data structure, missing values, and visualize distributions.
3.	Preprocessing: Select features (Age, Annual Income, Spending Score) and standardize them.
4.	Optimal Clusters: Use the Elbow Method to determine the best number of clusters.
5.	K-Means Clustering: Train the model (k=5) and assign cluster labels.
6.	Visualization: Plot customer segments using Seaborn.
7.	Interpretation: Analyze cluster characteristics and customer distribution.
8.	Insights:
•High-income, low-spending customers: Potential for marketing.
•High-income, high-spending: Premium customers.
•Low-income, high-spending: Budget-conscious high spenders. Running the Script Ensure Mall_Customers.csv is in the same directory and run: python customer_segmentation.py
Output
Clustered dataset with labels.
Visualizations of customer segments.
Business insights based on clusters.
License
This project is open-source and for educational purposes.


