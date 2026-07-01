import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view

data1574 = pd.read_csv("1574 hub dense vert.csv")
data1798 = pd.read_csv("1798 hub dense vert.csv")
data1950 = pd.read_csv("1950 hub dense vert.csv")
data2026 = pd.read_csv("2026 hub dense vert.csv")

Harvard = pd.read_csv("Harvard Hub dense vert.csv")
Amsterdam = pd.read_csv("Amsterdam Hub dense vert.csv")

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


external_series = {
    "Harvard": interpolate(Harvard),
    "Amsterdam": interpolate(Amsterdam),
}

window_size = 20
step = 1

series_2026 = series["2026"]
windows_2026 = sliding_window_view(series_2026, window_size)

def znorm(x, eps=1e-8):
    return (x - x.mean(axis=-1, keepdims=True)) / (x.std(axis=-1, keepdims=True) + eps)

def min_dist(series, shapelet):
    windows = sliding_window_view(series, window_size)
    windows_z = znorm(windows)
    shapelet_z = znorm(shapelet[None, :])[0]
    return np.min(np.mean((windows_z - shapelet_z) ** 2, axis=1))


best_score = -np.inf
best_shapelet = None
best_index = 0

for i in range(0, len(windows_2026), step):
    candidate = windows_2026[i]

    if candidate.std() < 1e-6:
        continue

    d_external = (
        min_dist(external_series["Harvard"], candidate) +
        min_dist(external_series["Amsterdam"], candidate)
    ) / 2

    score = d_external

    if score > best_score:
        best_score = score
        best_shapelet = candidate
        best_index = i

shapelet = best_shapelet
print("Best shapelet index in 2026:", best_index)
print("Best score (avg distance from Harvard/Amsterdam):", best_score)

def find_best_match(series, shapelet):
    windows = sliding_window_view(series, window_size)
    windows_z = znorm(windows)
    shapelet_z = znorm(shapelet[None, :])[0]
    dists = np.mean((windows_z - shapelet_z) ** 2, axis=1)
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
    plt.plot(common_angles[start:end], s[start:end], color="black", linewidth=3)
plt.title("Cambridge Time Series Graphs with Selected Shapelet")
plt.legend()
plt.show()

plt.figure()
plt.plot(shapelet, color="purple")
plt.title("Selected Shapelet")
plt.show()

all_labels = ["1574", "1798", "1950", "2026", "Harvard", "Amsterdam"]
all_series = {**series, **external_series}

distances = np.array([min_dist(all_series[name], shapelet) for name in all_labels])

target_distances = distances[:4]
external_distances = distances[4:]

boundary = (target_distances.max() + external_distances.min()) / 2

sim = 1 - distances / boundary
sim = np.clip(sim, -1, 1)

print("\nRaw distances (all, for reference):")
for l, d in zip(all_labels, distances):
    print(f"  {l:10s}: {d:.6f}")

print(f"\nDecision boundary distance: {boundary:.6f}")

print("\nFinal similarity (all, for reference):")
for l, s in zip(all_labels, sim):
    print(f"  {l:10s}: {s:.5f}")


plot3_labels = ["1574", "1798", "1950", "2026"]
plot3_sim = sim[:4]
plot3_colors = ["r", "b", "g", "purple"]

plt.figure(figsize=(10, 4))
plt.bar(plot3_labels, plot3_sim, color=plot3_colors)
plt.ylim(-1, 1)
plt.axhline(0, color="black", linewidth=0.8)
plt.title("Similarity Score to Historical Maps")
plt.show()

plot4_labels = ["2026", "Harvard", "Amsterdam"]
plot4_display_labels = ["Cambridge", "Harvard", "Amsterdam"]
plot4_indices = [all_labels.index(l) for l in plot4_labels]
plot4_sim = sim[plot4_indices]
plot4_colors = ["purple", "brown", "teal"]

plt.figure(figsize=(10, 4))
plt.bar(plot4_display_labels, plot4_sim, color=plot4_colors)
plt.ylim(-1, 1)
plt.axhline(0, color="black", linewidth=0.8)
plt.title("Similarity Score to Other Maps")
plt.show()