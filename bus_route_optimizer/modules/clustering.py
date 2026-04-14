"""
Clustering Module: Group nearby pickup points into clusters

Uses K-Means and DBSCAN algorithms to partition pickup locations
into efficient geographical clusters for route optimization.
"""

import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')


class Clustering:
    """Cluster pickup locations using K-Means or DBSCAN algorithms."""
    
    def __init__(self):
        """Initialize clustering module."""
        self.scaler = StandardScaler()
        self.clusters = None
        self.centers = None
        self.labels = None
    
    def haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calculate great-circle distance between two points on Earth using Haversine formula.
        
        Args:
            lat1, lon1: Latitude and longitude of first point (degrees)
            lat2, lon2: Latitude and longitude of second point (degrees)
        
        Returns:
            Distance in kilometers
        """
        # Convert to radians
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        r = 6371  # Earth's radius in kilometers
        
        return c * r
    
    def euclidean_distance(self, points1: np.ndarray, points2: np.ndarray) -> np.ndarray:
        """
        Calculate Euclidean distances between points.
        
        Args:
            points1: Array of shape (n, 2)
            points2: Array of shape (m, 2)
        
        Returns:
            Distance matrix of shape (n, m)
        """
        # Convert to km units for approximately 111km per degree
        points1 = points1 * 111  # km per degree latitude/longitude
        points2 = points2 * 111
        
        distances = np.zeros((len(points1), len(points2)))
        for i, p1 in enumerate(points1):
            for j, p2 in enumerate(points2):
                distances[i, j] = np.linalg.norm(p1 - p2)
        
        return distances
    
    def kmeans_clustering(
        self,
        pickup_locations: np.ndarray,
        num_clusters: Optional[int] = None,
        num_buses: Optional[int] = None,
        **kwargs
    ) -> Dict:
        """
        Cluster pickup locations using K-Means algorithm.
        
        Args:
            pickup_locations: Array of shape (n_points, 2) with [lat, lon]
            num_clusters: Number of clusters (if None, uses num_buses or suggests optimal)
            num_buses: Number of buses (used to determine clusters if not specified)
            **kwargs: Additional arguments for KMeans
        
        Returns:
            Dictionary containing:
                - labels: Cluster assignment for each point
                - centers: Cluster centers
                - inertia: Sum of squared distances to nearest cluster center
                - num_clusters: Number of clusters used
        """
        if num_clusters is None:
            num_clusters = num_buses or self._find_optimal_clusters(pickup_locations)
        
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, **kwargs)
        self.labels = kmeans.fit_predict(pickup_locations)
        self.centers = kmeans.cluster_centers_
        self.clusters = self._organize_by_cluster(pickup_locations, self.labels)
        
        result = {
            'labels': self.labels,
            'centers': self.centers,
            'inertia': kmeans.inertia_,
            'num_clusters': num_clusters,
            'clusters': self.clusters
        }
        
        return result
    
    def dbscan_clustering(
        self,
        pickup_locations: np.ndarray,
        eps: float = 0.1,  # ~11km at equator
        min_samples: int = 3,
        **kwargs
    ) -> Dict:
        """
        Cluster pickup locations using DBSCAN algorithm.
        
        DBSCAN is density-based and can identify outliers. Better for
        non-uniform cluster shapes.
        
        Args:
            pickup_locations: Array of shape (n_points, 2) with [lat, lon]
            eps: Maximum distance between points in a cluster (in scaled units)
            min_samples: Minimum number of samples in a cluster
            **kwargs: Additional arguments for DBSCAN
        
        Returns:
            Dictionary containing:
                - labels: Cluster assignment (-1 for noise points)
                - num_clusters: Number of clusters identified
                - num_noise_points: Number of noise points
                - clusters: Dictionary mapping cluster ID to point indices
        """
        # Scale coordinates for distance calculation
        scaler = StandardScaler()
        scaled_locations = scaler.fit_transform(pickup_locations)
        
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, **kwargs)
        self.labels = dbscan.fit_predict(scaled_locations)
        
        num_clusters = len(set(self.labels)) - (1 if -1 in self.labels else 0)
        num_noise = list(self.labels).count(-1)
        
        self.clusters = self._organize_by_cluster(pickup_locations, self.labels)
        
        result = {
            'labels': self.labels,
            'num_clusters': num_clusters,
            'num_noise_points': num_noise,
            'clusters': self.clusters
        }
        
        return result
    
    def _find_optimal_clusters(self, pickup_locations: np.ndarray) -> int:
        """
        Find optimal number of clusters using elbow method.
        
        Args:
            pickup_locations: Array of pickup coordinates
        
        Returns:
            Optimal number of clusters
        """
        inertias = []
        K_range = range(1, min(10, len(pickup_locations) // 3))
        
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(pickup_locations)
            inertias.append(kmeans.inertia_)
        
        # Simple elbow detection: find where second derivative is maximum
        if len(inertias) > 2:
            second_derivative = np.diff(np.diff(inertias))
            optimal_k = int(K_range[np.argmax(second_derivative) + 1])
        else:
            optimal_k = 2
        
        return max(2, optimal_k)
    
    def _organize_by_cluster(
        self,
        pickup_locations: np.ndarray,
        labels: np.ndarray
    ) -> Dict:
        """
        Organize pickup points by cluster assignment.
        
        Args:
            pickup_locations: Array of pickup coordinates
            labels: Cluster labels for each point
        
        Returns:
            Dictionary mapping cluster_id -> list of point indices
        """
        clusters = {}
        for idx, label in enumerate(labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(idx)
        
        return clusters
    
    def get_cluster_statistics(
        self,
        pickup_locations: np.ndarray,
        labels: np.ndarray
    ) -> Dict:
        """
        Calculate statistics for each cluster.
        
        Args:
            pickup_locations: Array of pickup coordinates
            labels: Cluster labels
        
        Returns:
            Dictionary with cluster statistics
        """
        stats = {}
        unique_labels = set(labels)
        
        for label in unique_labels:
            if label == -1:  # Skip noise points in DBSCAN
                continue
            
            mask = labels == label
            cluster_points = pickup_locations[mask]
            
            stats[label] = {
                'num_points': len(cluster_points),
                'center': np.mean(cluster_points, axis=0),
                'min_lat': np.min(cluster_points[:, 0]),
                'max_lat': np.max(cluster_points[:, 0]),
                'min_lon': np.min(cluster_points[:, 1]),
                'max_lon': np.max(cluster_points[:, 1]),
                'radius_km': self._calculate_cluster_radius(cluster_points)
            }
        
        return stats
    
    def _calculate_cluster_radius(self, cluster_points: np.ndarray) -> float:
        """
        Calculate the radius of a cluster (max distance from center).
        
        Args:
            cluster_points: Array of points in cluster
        
        Returns:
            Radius in kilometers
        """
        center = np.mean(cluster_points, axis=0)
        max_distance = 0
        
        for point in cluster_points:
            dist = self.haversine_distance(
                center[0], center[1], point[0], point[1]
            )
            max_distance = max(max_distance, dist)
        
        return max_distance


if __name__ == "__main__":
    # Example usage
    from data_generator import DataGenerator
    
    generator = DataGenerator()
    dataset = generator.generate_dataset(num_pickup_points=30, num_buses=3)
    
    clustering = Clustering()
    
    # K-Means clustering
    kmeans_result = clustering.kmeans_clustering(
        dataset['pickup_locations'],
        num_buses=dataset['num_buses']
    )
    print(f"K-Means: {kmeans_result['num_clusters']} clusters formed")
    print(f"Inertia: {kmeans_result['inertia']:.2f}")
    
    # Get statistics
    stats = clustering.get_cluster_statistics(
        dataset['pickup_locations'],
        kmeans_result['labels']
    )
    for cluster_id, stat in stats.items():
        print(f"\nCluster {cluster_id}:")
        print(f"  Points: {stat['num_points']}")
        print(f"  Radius: {stat['radius_km']:.2f} km")
