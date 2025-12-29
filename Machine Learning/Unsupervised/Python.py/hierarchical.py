import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

# Title
st.title("Agglomerative Clustering App")

# Side bar info
st.sidebar.header("About the Project")
st.sidebar.info(
    "This app uses a Agglomerative Clustering  model trained on the Bio-dataset. "
    "It predicts Cluster based on their Age and Salary."
)

# Generate dataset
np.random.seed(42)
df = pd.DataFrame({
    'Person': [f"A{i}" for i in range(1, 31)],
    'Age': np.random.randint(20, 60, 30),
    'Salary': np.random.randint(20, 90, 30)
})
st.subheader("Sample Dataset")
st.dataframe(df)

# Select features
X = df[['Age', 'Salary']].values

# Normalize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create linkage
linked = linkage(X_scaled, method='ward')

# Plot dendrogram
st.subheader("Dendrogram")
fig, ax = plt.subplots(figsize=(8, 5))
dendrogram(linked, labels=df['Person'].values, leaf_rotation=90, leaf_font_size=10)
st.pyplot(fig)

# Calculate silhouette scores
scores = []
for k in range(2, 11):
    model = AgglomerativeClustering(n_clusters=k, linkage='ward')
    labels = model.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    scores.append((k, score))

# Display scores
st.subheader("Silhouette Scores")
for k, score in scores:
    st.write(f"Clusters: {k}, Silhouette Score: {score:.3f}")

# Find best number of clusters
best_k = max(scores, key=lambda x: x[1])[0]
st.success(f"Best number of clusters: {best_k}")

# Fit final model
final_model = AgglomerativeClustering(n_clusters=best_k, linkage='ward')
df['Cluster'] = final_model.fit_predict(X_scaled)

st.subheader("Clustered Data")
st.dataframe(df)

# --- Prediction Section ---
st.subheader("Predict Cluster for New Data")

# User input
age_input = st.number_input("Enter Age:", min_value=18, max_value=80, value=30)
salary_input = st.number_input("Enter Salary:", min_value=10, max_value=200, value=50)

# Transform input using same scaler
input_data = np.array([[age_input, salary_input]])
input_scaled = scaler.transform(input_data)

# Predict button
if st.button("Predict"):
    # Scale input
    input_data = np.array([[age_input, salary_input]])
    input_scaled = scaler.transform(input_data)

    # Predict cluster using nearest cluster center (approximation)
# Agglomerative doesn't have predict(), so we assign to closest cluster mean
cluster_means = []
for cluster in np.unique(df['Cluster']):
    cluster_points = X_scaled[df['Cluster'] == cluster]
    cluster_means.append(cluster_points.mean(axis=0))

# Compute distances to cluster centers
distances = [np.linalg.norm(input_scaled - mean) for mean in cluster_means]
predicted_cluster = np.argmin(distances)

st.success(f"The predicted cluster for Age is  {age_input} and Salary is {salary_input} is: Cluster {predicted_cluster}")

# Plot clusters
st.subheader("Cluster Plot")
fig2, ax2 = plt.subplots()
for cluster in np.unique(df['Cluster']):
    cluster_data = df[df['Cluster'] == cluster]
    ax2.scatter(cluster_data['Age'], cluster_data['Salary'], label=f'Cluster {cluster}')
ax2.set_xlabel('Age')
ax2.set_ylabel('Salary')
ax2.legend()
plt.grid(True)
st.pyplot(fig2)
