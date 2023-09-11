import numpy as np
from sklearn.cluster import DBSCAN


def reduce_data_points(data: np.ndarray, rank: int) -> np.ndarray:
    """
    Perform Density-Based Spatial Clustering of Applications with Noise (DBSCAN) on the data
    to reduce the number of data points
    """
    dbscan = DBSCAN(eps=0.05, min_samples=2)
    dbscan.fit(data)

    unique_labels = np.unique(dbscan.labels_)
    merged_points = []
    outliers = []

    for label in unique_labels:
        if label == -1:
            outliers.append(data[dbscan.labels_ == label])
        else:
            merged_points.append(np.mean(data[dbscan.labels_ == label], axis=0))

    # reshape outliers
    outliers = np.array(outliers).reshape(-1, 2)

    merged_points = np.array(merged_points)
    print(merged_points.shape)
    outliers = np.array(outliers)
    print(outliers.shape)

    print(f"Process {rank} found {len(merged_points)} merged points")
    print(f"Process {rank} found {len(outliers)} outliers")

    if len(merged_points) == 0:
        return outliers

    if len(outliers) == 0:
        return merged_points

    return np.concatenate((merged_points, outliers))
