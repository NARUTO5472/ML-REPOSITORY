import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import hdbscan
from sklearn.preprocessing import StandardScaler

import geopandas as gpd
import contextily as ctx
from shapely.geometry import Point

import warnings
warnings.filterwarnings('ignore')

import requests
import zipfile
import io
import os

# Download and extract Canada basemap
zip_file_url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/YcUk-ytgrPkmvZAh5bf7zA/Canada.zip'
output_dir = './'
os.makedirs(output_dir, exist_ok=True)

response = requests.get(zip_file_url)
response.raise_for_status()

with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
    for file_name in zip_ref.namelist():
        if file_name.endswith('.tif'):
            zip_ref.extract(file_name, output_dir)
            print(f"Downloaded and extracted: {file_name}")

# Load the CSV containing museum data
url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/r-maSj5Yegvw2sJraT15FA/ODCAF-v1-0.csv'
df = pd.read_csv(url, encoding="ISO-8859-1")

# Clean coordinate columns
df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')
df.dropna(subset=['Latitude', 'Longitude'], inplace=True)

# Scale and cluster
coords = df[['Latitude', 'Longitude']].copy()
coords['Latitude'] *= 2
dbscan = DBSCAN(eps=1.0, min_samples=3, metric='euclidean')
df['Cluster'] = dbscan.fit_predict(coords)


# Scale coordinates appropriately
coords_scaled = df.copy()
coords_scaled["Latitude"] = 2 * coords_scaled["Latitude"]

# Run DBSCAN clustering
dbscan = DBSCAN(eps=1.0, min_samples=3, metric='euclidean')
df['Cluster'] = dbscan.fit_predict(coords_scaled[['Latitude', 'Longitude']])


# Display cluster sizes
print("\n Cluster Sizes:")
print(df['Cluster'].value_counts())

# Plotting function
def plot_clustered_locations(df, title='Museums Clustered by Proximity'):
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['Longitude'], df['Latitude']), crs="EPSG:4326")
    gdf = gdf.to_crs(epsg=3857)

    fig, ax = plt.subplots(figsize=(15, 10))
    non_noise = gdf[gdf['Cluster'] != -1]
    noise = gdf[gdf['Cluster'] == -1]

    noise.plot(ax=ax, color='k', markersize=30, ec='r', alpha=1, label='Noise')
    non_noise.plot(ax=ax, column='Cluster', cmap='tab10', markersize=30, ec='k', legend=False, alpha=0.6)

    ctx.add_basemap(ax, source='./Canada.tif', zoom=4)

    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    plt.show()

# Call the plotting function
plot_clustered_locations(df, title='Museums Clustered by Proximity')
