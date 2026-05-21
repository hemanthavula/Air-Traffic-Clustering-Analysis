import pandas as pd 
import sweetviz
import matplotlib.pyplot as plt


# from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import OneHotEncoder

from sklearn.cluster import KMeans
from sklearn import metrics

import pickle
# **Import the data**

from sqlalchemy import create_engine

from urllib.parse import quote
df = pd.read_csv(r"Air_Traffic_Passenger_Statistics.csv")

# Credentials to connect to Database
user = 'user1' # user name
pw = quote('user1') # password
db = 'air_routes_db' # database
# creating engine to connect MySQL database
engine = create_engine(f"mysql+pymysql://{user}:{pw}@localhost/{db}")

# to_sql() - function to push the dataframe onto a SQL table.
df.to_sql('airline_tbl', con = engine, if_exists = 'replace', chunksize = 1000, index = False)

sql = 'select * from airline_tbl;'
df = pd.read_sql_query(sql, engine)

# Data types
df.info()
df.isnull().sum()

# EXPLORATORY DATA ANALYSIS (EDA) / DESCRIPTIVE STATISTICS
# ***Descriptive Statistics and Data Distribution Function***

df.describe()

df.duplicated().sum()

# my_report = sweetviz.analyze([df, "df"])
# my_report.show_html('Report.html')

# As we can see there are multiple columns in our dataset, 
# but for cluster analysis we will use 
# Operating Airline, Geo Region, Passenger Count and Flights held by each airline.
df1 = df[["Operating Airline", "GEO Region", "Passenger Count"]]

airline_count = df1["Operating Airline"].value_counts()
airline_count.sort_index(inplace=True)

passenger_count = df1.groupby("Operating Airline").sum()["Passenger Count"]
passenger_count.sort_index(inplace=True)

'''So as this algorithms is working with distances it is very sensitive to outliers, 
that’s why before doing cluster analysis we have to identify outliers and remove them from the dataset. 
In order to find outliers more accurately, we will build the scatter plot.'''

df2 = pd.concat([airline_count, passenger_count], axis=1)
# x = airline_count.values
# y = passenger_count.values
plt.figure(figsize = (10,10))
plt.scatter(df2['count'], df2['Passenger Count'])
plt.xlabel("Flights held")
plt.ylabel("Passengers")
for i, txt in enumerate(airline_count.index.values):
    a = plt.gca()
    plt.annotate(txt, (df2['count'][i], df2['Passenger Count'][i]))
plt.show()

df2.index
# We can see that most of the airlines are grouped together in the bottom left part of the plot, 
# some are above them, and it has 2 outliers United Airlines and Unites Airlines — Pre 07/01/2013.
# So let’s get rid of them.

index_labels_to_drop = ['United Airlines', 'United Airlines - Pre 07/01/2013']
df3 = df2.drop(index_labels_to_drop)


# # CLUSTERING MODEL BUILDING

# ### KMeans Clustering
# Libraries for creating scree plot or elbow curve 
# from sklearn.cluster import KMeans
# import matplotlib.pyplot as plt

###### scree plot or elbow curve ############

TWSS = []
k = list(range(2, 9))

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(df3)
    TWSS.append(kmeans.inertia_)

TWSS

# ## Creating a scree plot to find out no.of cluster
plt.plot(k, TWSS, 'ro-'); plt.xlabel("No_of_Clusters"); plt.ylabel("total_within_SS")

List = []

for k in range(2, 9):
    kmeans = KMeans(n_clusters=k) 
    kmeans.fit(df3)
    List.append(kmeans.inertia_)

from kneed import KneeLocator
kl = KneeLocator(range(2, 9), List, curve='convex', direction='decreasing')  # Adjust curve type and direction if needed

import matplotlib.pyplot as plt
plt.plot(range(2, 9), List)
plt.xticks(range(2, 9))
plt.ylabel("Inertia")
plt.axvline(x=kl.elbow, color='r', label='Elbow', ls='--')
plt.show()

print("Elbow point:", kl.elbow)
# Not able to detect the best K value (knee/elbow) as the line is mostly linear

# Building KMeans clustering
model = KMeans(n_clusters = 5)
yy = model.fit(df3)

# Cluster labels
model.labels_

# ## Cluster Evaluation

# **Silhouette coefficient:**  
# Silhouette coefficient is a Metric, which is used for calculating 
# goodness of clustering technique and the value ranges between (-1 to +1).
# It tells how similar an object is to its own cluster (cohesion) compared to 
# other clusters (separation).
# A score of 1 denotes the best meaning that the data point is very compact 
# within the cluster to which it belongs and far away from the other clusters.
# Values near 0 denote overlapping clusters.

# from sklearn import metrics
metrics.silhouette_score(df3, model.labels_)

# **Calinski Harabasz:**
# Higher value of CH index means cluster are well separated.
# There is no thumb rule which is acceptable cut-off value.
metrics.calinski_harabasz_score(df3, model.labels_)

# **Davies-Bouldin Index:**
# Unlike the previous two metrics, this score measures the similarity of clusters. 
# The lower the score the better the separation between your clusters. 
# Vales can range from zero and infinity
metrics.davies_bouldin_score(df3, model.labels_)

# ### Evaluation of Number of Clusters using Silhouette Coefficient Technique
from sklearn.metrics import silhouette_score

silhouette_coefficients = []

for k in range (2, 9):
    kmeans = KMeans(n_clusters = k)
    kmeans.fit(df3)
    score = silhouette_score(df3, kmeans.labels_)
    k = k
    Sil_coff = score
    silhouette_coefficients.append([k, Sil_coff])

silhouette_coefficients

sorted(silhouette_coefficients, reverse = True, key = lambda x: x[1])


# silhouette coefficients shows the number of clusters 'k = 2' as the best value

# Building KMeans clustering
bestmodel = KMeans(n_clusters = 2)
result = bestmodel.fit(df3)

# ## Save the KMeans Clustering Model
# import pickle
pickle.dump(result, open('Clust_.pkl', 'wb'))

import os
os.getcwd()

# Cluster labels
bestmodel.labels_

mb = pd.Series(bestmodel.labels_) 
df3['cluster_id'] = mb.values
# Concate the Results with data

# Save the Results to a CSV file
df3.to_csv('Air.csv', encoding = 'utf-8', index = True)
 
import os
os.getcwd()


