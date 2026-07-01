import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view

data1574 = pd.read_csv("1574 hub dense vert.csv")
data1798 = pd.read_csv("1798 hub dense vert.csv")
data1950 = pd.read_csv("1950 hub dense vert.csv")
data2026 = pd.read_csv("2026 hub dense vert.csv")

common_angles = np.linspace(0, 360, 100)

def interpolate(df):
    angles = df.iloc[:, 6].values
    dist = df.iloc[:, 5].values

    idx = np.argsort(angles)
    angles = angles[idx]
    dist = dist[idx]

    angles, uidx = np.unique(angles, return_index=True)
    dist = dist[uidx]

    return np.interp(common_angles, angles, dist)

series1574 = interpolate(data1574)
series1798 = interpolate(data1798)
series1950 = interpolate(data1950)
series2026 = interpolate(data2026)

series_list = [series1574, series1798, series1950, series2026]
labels = ["1574", "1798", "1950", "2026"]
colors = ["r", "b", "g", "purple"]

window_size = 20

windows_2026 = sliding_window_view(series2026, window_size)
shapelet = windows_2026[len(windows_2026) // 2]


def find_best_match(series, shapelet):
    windows = sliding_window_view(series, window_size)
    dists = np.mean((windows - shapelet) ** 2, axis=1)
    return np.argmin(dists)

match_indices = [
    find_best_match(series1574, shapelet),
    find_best_match(series1798, shapelet),
    find_best_match(series1950, shapelet),
    find_best_match(series2026, shapelet),
]

def min_distance(series, shapelet):
    windows = sliding_window_view(series, window_size)
    dists = np.mean((windows - shapelet) ** 2, axis=1)
    return np.min(dists)

distances = np.array([
    min_distance(series1574, shapelet),
    min_distance(series1798, shapelet),
    min_distance(series1950, shapelet),
    min_distance(series2026, shapelet),
])

sim_01 = 1 - (distances / distances.max())
sim_scaled = 2 * sim_01 - 1
sim_scaled[-1] = 1.0 

plt.figure(figsize=(10, 5))

for i, (s, l, c) in enumerate(zip(series_list, labels, colors)):
    plt.plot(common_angles, s, label=l, color=c)

    start = match_indices[i]
    end = start + window_size

    plt.plot(
        common_angles[start:end],
        s[start:end],
        color="black",
        linewidth=3
    )

plt.title("Time Series with 2026 Shapelet Match Highlighted")
plt.legend()
plt.show()


plt.figure()
plt.plot(shapelet, color="black")
plt.title("2026 Reference Shapelet")
plt.show()

plt.figure()
plt.bar(labels, sim_scaled)
plt.ylim(-1, 1)
plt.title("Similarity to 2026 Shapelet (-1 to 1)")
plt.ylabel("Similarity")
plt.show()

print("\nSimilarity Scores (-1 to 1):")
for l, s in zip(labels, sim_scaled):
    print(l, ":", s)