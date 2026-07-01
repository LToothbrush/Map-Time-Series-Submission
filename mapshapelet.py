import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyts.classification import LearningShapelets
from pyts.utils import windowed_view
from numpy.lib.stride_tricks import sliding_window_view


data1574 = pd.read_csv("1574 hub dense vert.csv")
data1798 = pd.read_csv("1798 hub dense vert.csv")
data1950 = pd.read_csv("1950 hub dense vert.csv")
data2026 = pd.read_csv("2026 hub dense vert.csv")

common_angles = np.linspace(0, 360, 100) 

def interpolate_series(df):
    angles = df.iloc[:, 6].values    
    distances = df.iloc[:, 5].values  

    sorted_idx = np.argsort(angles)
    angles = angles[sorted_idx]
    distances = distances[sorted_idx]

    angles, unique_idx = np.unique(angles, return_index=True)
    distances = distances[unique_idx]

    interp_dist = np.interp(common_angles, angles, distances)
    return interp_dist


series1574 = interpolate_series(data1574)
series1798 = interpolate_series(data1798)
series1950 = interpolate_series(data1950)
series2026 = interpolate_series(data2026)

X = np.array([
    series1574,
    series1798,
    series1950,
    series2026
])

y = np.array([0, 1, 2, 3])
print("Dataset shape:", X.shape)


clf = LearningShapelets(random_state=42, tol=0.01)
clf.fit(X, y)

shapelets = np.asarray([clf.shapelets_[0, -1]])

shapelet_size = shapelets.shape[1]
X_window = windowed_view(X, window_size=shapelet_size, window_step=1)

X_dist = np.mean(
    (X_window[:, :, None] - shapelets[None, :]) ** 2,
    axis=3
).min(axis=1)

window_size = len(shapelets[0])
X_windows = sliding_window_view(X, window_shape=window_size, axis=1)

best_match_indices = []

for i, series_windows in enumerate(X_windows):
    distances = np.mean((series_windows - shapelets[0])**2, axis=1)
    best_idx = np.argmin(distances)
    best_match_indices.append(best_idx)
    print(f"Series {i} best matching segment: indices {best_idx} to {best_idx+window_size}")


plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
"""
series_labels = ["1574", "1798", "1950", "2026"]
colors = ['r', 'b', 'g', 'purple']

for i, series in enumerate([series1574, series1798 , series1950, series2026]):
    plt.plot(common_angles, series, label=series_labels[i], color=colors[i])
    
    
    start = best_match_indices[i]
    end = start + window_size
    plt.plot(common_angles[start:end], series[start:end], color='k', linewidth=3, alpha=0.7)

plt.title("Aligned Distance vs Angle with Shapelet Highlighted")
plt.xlabel("Angle")
plt.ylabel("Distance")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(shapelets[0], label="Shapelet 1", color='m')
plt.title("Learned Shapelet")
plt.xlabel("Index")
plt.ylabel("Distance")
plt.legend()

plt.tight_layout()
plt.show()
"""
series_labels = ["1574", "1798", "1950", "2026"]
colors = ['r', 'b', 'g', 'purple']


plt.figure(figsize=(8, 5))

for i, series in enumerate([series1574, series1798, series1950, series2026]):
    plt.plot(common_angles, series,
             label=series_labels[i],
             color=colors[i])

    start = best_match_indices[i]
    end = start + window_size
    plt.plot(common_angles[start:end],
             series[start:end],
             color='k',
             linewidth=3,
             alpha=0.7)

plt.title("Aligned Distance vs Angle with Shapelet Highlighted")
plt.xlabel("Angle (degrees)")
plt.ylabel("Distance")
plt.legend()
plt.tight_layout()
plt.show()


plt.figure(figsize=(6, 4))

plt.plot(shapelets[0], color='m', linewidth=2)
plt.title("Learned Shapelet")
plt.xlabel("Shapelet Index")
plt.ylabel("Distance")

plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 4))
for color, label in zip(colors, y):
    plt.scatter(
        X_dist[y == label],
        np.zeros_like(X_dist[y == label]),
        c=color,
        label=f'Class {label}'
    )
plt.title("Distances to Shapelet")
plt.xlabel("Distance")
plt.yticks([])
plt.legend()
plt.show()