import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import hdbscan

import geopandas as gpd
import contextily as ctx
from shapely.geometry import Point

import warnings
warnings.filterwarnings('ignore')

# Load the Museum Data
url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/r-maSj5Yegvw2sJraT15FA/ODCAF-v1-0.csv'
df = pd.read_csv(url, encoding="ISO-8859-1")

# ✅ Clean coordinate columns
df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')
df.dropna(subset=['Latitude', 'Longitude'], inplace=True)

# ✅ Preprocess Coordinates for HDBSCAN
coords = df[['Latitude', 'Longitude']].copy()
coords['Latitude'] *= 2  # scale to balance latitude-longitude

# ✅ Apply HDBSCAN Clustering
hdb = hdbscan.HDBSCAN(min_samples=None, min_cluster_size=3, metric='euclidean')
df['Cluster'] = hdb.fit_predict(coords)

# ✅ Display Cluster Sizes
print("\nCluster Sizes:")
print(df['Cluster'].value_counts())

# ✅ Plotting Function
def plot_clustered_locations(df, title='HDBSCAN: Museums Clustered by Proximity'):
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['Longitude'], df['Latitude']), crs="EPSG:4326")
    gdf = gdf.to_crs(epsg=3857)

    fig, ax = plt.subplots(figsize=(15, 10))
    non_noise = gdf[gdf['Cluster'] != -1]
    noise = gdf[gdf['Cluster'] == -1]

    noise.plot(ax=ax, color='k', markersize=30, ec='r', alpha=1, label='Noise')
    non_noise.plot(ax=ax, column='Cluster', cmap='tab10', markersize=30, ec='k', legend=False, alpha=0.6)

    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, zoom=4)

    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    plt.show()

# ✅ Show the Clustering on a Map
plot_clustered_locations(df)
