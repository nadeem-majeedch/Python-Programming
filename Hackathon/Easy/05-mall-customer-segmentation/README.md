# Problem 05 – Mall Customer Segmentation

**Domain**: Clustering (Unsupervised Learning)  
**Goal**: Segment mall customers into distinct groups based on age, income, and spending behavior.

## Objectives
- Use the Elbow Method to find the optimal number of clusters.
- Apply K-Means clustering with the optimal k.
- Visualize clusters using PCA.
- Profile each cluster's characteristics.

## Success Criteria
- Identify the optimal k from the elbow curve.
- Achieve silhouette score > 0.35.
- Provide meaningful cluster descriptions.

## Dataset: `data/mall_customers.csv`
~300 rows with columns:
- CustomerID, Age, AnnualIncome (k$), SpendingScore (1–100)
