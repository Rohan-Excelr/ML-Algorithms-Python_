import pandas as pd
data = pd.read_csv("E:/DS NOTES/Mall_Customers.csv")
print(data.info())
print(data.describe())  
print(data.head())
x = data[['Annual Income (k$)','Spending Score (1-100)']]
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)
print("Data scaled successfully")
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters =2,random_state=42)
data['cluster'] = kmeans.fit_predict(x_scaled)
print("KMeans clustering completed")
centroids = scaler.inverse_transform(kmeans.cluster_centers_)
for i , c in enumerate(centroids):
    print(f"centroid {i}: {c[0]},{c[1]}")
    print(data['cluster'].value_counts())
import matplotlib.pyplot as plt
plt.scatter(x_scaled[:,0],x_scaled[:,1],c=data['cluster'])
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],c='red',marker='x')
plt.xlabel('Income (k$)')
plt.ylabel('Spending Score')    
plt.title('KMeans Clustering')
plt.show()


