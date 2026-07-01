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

series = {
    "1574": interpolate(data1574),
    "1798": interpolate(data1798),
    "1950": interpolate(data1950),
    "2026": interpolate(data2026),
}


window_size = 20
step = 1

series_2026 = series["2026"]

windows_2026 = sliding_window_view(series_2026, window_size)

def min_dist(series, shapelet):
    windows = sliding_window_view(series, window_size)
    return np.min(np.mean((windows - shapelet) ** 2, axis=1))

best_score = -np.inf
best_shapelet = None
best_index = 0

for i in range(0, len(windows_2026), step):

    candidate = windows_2026[i]

    d_2026 = min_dist(series["2026"], candidate)

    d_others = (
        min_dist(series["1574"], candidate) +
        min_dist(series["1798"], candidate) +
        min_dist(series["1950"], candidate)
    ) / 3

    score = d_others - d_2026

    if score > best_score:
        best_score = score
        best_shapelet = candidate
        best_index = i

shapelet = best_shapelet

print("Best shapelet index in 2026:", best_index)
print("Best score:", best_score)

def find_best_match(series, shapelet):
    windows = sliding_window_view(series, window_size)
    dists = np.mean((windows - shapelet) ** 2, axis=1)
    return np.argmin(dists)

match_indices = [
    find_best_match(series["1574"], shapelet),
    find_best_match(series["1798"], shapelet),
    find_best_match(series["1950"], shapelet),
    find_best_match(series["2026"], shapelet),
]

labels = ["1574", "1798", "1950", "2026"]
colors = ["r", "b", "g", "purple"]

plt.figure(figsize=(10, 5))

for i, name in enumerate(labels):
    s = series[name]

    plt.plot(common_angles, s, label=name, color=colors[i])

    start = match_indices[i]
    end = start + window_size

    plt.plot(
        common_angles[start:end],
        s[start:end],
        color="black",
        linewidth=3
    )

plt.title("True Optimized Shapelet Matches")
plt.legend()
plt.show()


plt.figure()
plt.plot(shapelet, color="black")
plt.title("Optimized Shapelet (Most 2026-Specific Pattern)")
plt.show()

distances = [
    min_dist(series["1574"], shapelet),
    min_dist(series["1798"], shapelet),
    min_dist(series["1950"], shapelet),
    min_dist(series["2026"], shapelet),
]

distances = np.array(distances)

sim = 1 - distances / distances.max()
sim = 2 * sim - 1
sim[-1] = 1.0

plt.figure()
plt.bar(labels, sim)
plt.ylim(-1, 1)
plt.title("Similarity to Optimized Shapelet (-1 to 1)")
plt.show()

print("\nFinal similarity:")
for l, s in zip(labels, sim):
    print(l, ":", s)