"""
Visualization Module: Visualize routes and clusters on maps

Provides functions to create interactive maps using folium
and static plots using matplotlib.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import folium
from folium import plugins
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


class RouteVisualizer:
    """Visualize bus routes and clusters on maps."""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize visualizer.
        
        Args:
            figsize: Figure size for matplotlib plots (width, height)
        """
        self.figsize = figsize
    
    def plot_clusters(
        self,
        pickup_locations: np.ndarray,
        clusters: Dict[int, List[int]],
        centers: Optional[np.ndarray] = None,
        depot: Optional[np.ndarray] = None,
        title: str = "Cluster Visualization",
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot clusters on a 2D map using matplotlib.
        
        Args:
            pickup_locations: Array of shape (n, 2) with [lat, lon]
            clusters: Dictionary mapping cluster_id -> list of point indices
            centers: Cluster centers (optional)
            depot: Depot location (optional)
            title: Plot title
            save_path: Path to save plot (if None, displays plot)
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Color map for clusters
        colors = plt.cm.tab20(np.linspace(0, 1, max(len(clusters), 10)))
        
        # Plot clusters
        for cluster_id, indices in clusters.items():
            if cluster_id == -1:  # Skip noise points
                color = 'gray'
                marker = 'x'
                label = 'Noise'
            else:
                color = colors[cluster_id % len(colors)]
                marker = 'o'
                label = f'Cluster {cluster_id}'
            
            cluster_points = pickup_locations[indices]
            ax.scatter(
                cluster_points[:, 1], cluster_points[:, 0],
                c=[color], marker=marker, s=100, label=label,
                alpha=0.7, edgecolors='black', linewidth=0.5
            )
        
        # Plot cluster centers
        if centers is not None:
            ax.scatter(
                centers[:, 1], centers[:, 0],
                marker='*', s=500, c='red', edgecolors='black',
                linewidth=1, label='Cluster Centers', zorder=5
            )
        
        # Plot depot
        if depot is not None:
            ax.scatter(
                depot[1], depot[0],
                marker='s', s=300, c='green', edgecolors='black',
                linewidth=1.5, label='Depot', zorder=5
            )
        
        ax.set_xlabel('Longitude', fontsize=11)
        ax.set_ylabel('Latitude', fontsize=11)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Cluster plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_route(
        self,
        pickup_locations: np.ndarray,
        route: List[int],
        depot: np.ndarray,
        destination: Optional[np.ndarray] = None,
        title: str = "Bus Route Visualization",
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot a single bus route on a 2D map.
        
        Args:
            pickup_locations: Array of shape (n, 2) with [lat, lon]
            route: Route (list of indices)
            depot: Starting location
            destination: Destination location (if None, same as depot)
            title: Plot title
            save_path: Path to save plot (if None, displays plot)
        """
        if destination is None:
            destination = depot
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot pickup points
        ax.scatter(
            pickup_locations[:, 1], pickup_locations[:, 0],
            c='blue', marker='o', s=100, alpha=0.6,
            edgecolors='black', linewidth=0.5, label='Pickup Points'
        )
        
        # Plot route path
        route_path = [depot] + pickup_locations[route[:-1]] + [destination]
        for i in range(len(route_path) - 1):
            start = route_path[i]
            end = route_path[i + 1]
            ax.arrow(
                start[1], start[0], end[1] - start[1], end[0] - start[0],
                head_width=0.0005, head_length=0.0003, fc='red', ec='red',
                alpha=0.6, linewidth=1.5
            )
        
        # Plot depot and destination
        ax.scatter(
            depot[1], depot[0],
            marker='s', s=300, c='green', edgecolors='black',
            linewidth=1.5, label='Depot', zorder=5
        )
        
        if not np.allclose(destination, depot):
            ax.scatter(
                destination[1], destination[0],
                marker='^', s=300, c='purple', edgecolors='black',
                linewidth=1.5, label='Destination', zorder=5
            )
        
        # Add stop numbers
        for idx, point in enumerate(pickup_locations[route[:-1]]):
            ax.annotate(
                str(idx + 1), (point[1], point[0]),
                fontsize=8, ha='center', va='center',
                bbox=dict(boxstyle='circle', facecolor='yellow', alpha=0.7)
            )
        
        ax.set_xlabel('Longitude', fontsize=11)
        ax.set_ylabel('Latitude', fontsize=11)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Route plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_multiple_routes(
        self,
        pickup_locations: np.ndarray,
        routes: List[List[int]],
        depot: np.ndarray,
        destination: Optional[np.ndarray] = None,
        title: str = "Multiple Bus Routes",
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot multiple bus routes on one map.
        
        Args:
            pickup_locations: Array of shape (n, 2) with [lat, lon]
            routes: List of routes (each route is list of indices)
            depot: Starting location
            destination: Destination location
            title: Plot title
            save_path: Path to save plot
        """
        if destination is None:
            destination = depot
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Color map for routes
        colors = plt.cm.tab10(np.linspace(0, 1, max(len(routes), 2)))
        
        # Plot pickup points
        ax.scatter(
            pickup_locations[:, 1], pickup_locations[:, 0],
            c='lightblue', marker='o', s=100, alpha=0.6,
            edgecolors='black', linewidth=0.5, label='Pickup Points', zorder=2
        )
        
        # Plot each route
        for route_id, route in enumerate(routes):
            color = colors[route_id]
            route_path = [depot] + pickup_locations[route[:-1]] + [destination]
            
            # Plot path
            for i in range(len(route_path) - 1):
                start = route_path[i]
                end = route_path[i + 1]
                ax.plot(
                    [start[1], end[1]], [start[0], end[0]],
                    color=color, linewidth=2, alpha=0.7,
                    label=f'Bus {route_id + 1}' if i == 0 else ''
                )
        
        # Plot depot and destination
        ax.scatter(
            depot[1], depot[0],
            marker='s', s=300, c='green', edgecolors='black',
            linewidth=1.5, label='Depot', zorder=5
        )
        
        if not np.allclose(destination, depot):
            ax.scatter(
                destination[1], destination[0],
                marker='^', s=300, c='purple', edgecolors='black',
                linewidth=1.5, label='Destination', zorder=5
            )
        
        ax.set_xlabel('Longitude', fontsize=11)
        ax.set_ylabel('Latitude', fontsize=11)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Multiple routes plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def create_interactive_map(
        self,
        pickup_locations: np.ndarray,
        route: List[int],
        depot: np.ndarray,
        destination: Optional[np.ndarray] = None,
        title: str = "Bus Route Map",
        zoom_start: int = 13,
        save_path: Optional[str] = None
    ) -> folium.Map:
        """
        Create an interactive map using folium.
        
        Args:
            pickup_locations: Array of shape (n, 2) with [lat, lon]
            route: Route (list of indices)
            depot: Starting location
            destination: Destination location
            title: Map title
            zoom_start: Initial zoom level
            save_path: Path to save HTML map
        
        Returns:
            folium Map object
        """
        if destination is None:
            destination = depot
        
        # Calculate map center
        center_lat = np.mean(pickup_locations[:, 0])
        center_lon = np.mean(pickup_locations[:, 1])
        
        # Create map
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=zoom_start,
            tiles='OpenStreetMap'
        )
        
        # Add route path
        route_path = [depot.tolist()] + pickup_locations[route[:-1]].tolist() + [destination.tolist()]
        
        # Draw route
        folium.PolyLine(
            route_path,
            color='red',
            weight=2,
            opacity=0.8,
            popup=title
        ).add_to(m)
        
        # Add markers for pickup points
        for idx, point in enumerate(pickup_locations[route[:-1]]):
            folium.CircleMarker(
                location=[point[0], point[1]],
                radius=8,
                popup=f"Stop {idx + 1}",
                color='blue',
                fill=True,
                fillColor='blue',
                fillOpacity=0.7
            ).add_to(m)
        
        # Add depot marker
        folium.Marker(
            location=depot.tolist(),
            popup="Depot (Start)",
            icon=folium.Icon(color='green', icon='home')
        ).add_to(m)
        
        # Add destination marker
        if not np.allclose(destination, depot):
            folium.Marker(
                location=destination.tolist(),
                popup="Destination (End)",
                icon=folium.Icon(color='purple', icon='flag')
            ).add_to(m)
        
        # Add route info
        m.add_child(folium.LatLngPopup())
        
        if save_path:
            m.save(save_path)
            print(f"Interactive map saved to {save_path}")
        
        return m
    
    def plot_route_statistics(
        self,
        routes: List[Dict],
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot statistics about routes (distance, time, load).
        
        Args:
            routes: List of route dictionaries with metrics
            save_path: Path to save plot
        """
        if not routes:
            print("No routes to plot")
            return
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        num_buses = len(routes)
        bus_ids = [f"Bus {i+1}" for i in range(num_buses)]
        
        # Distance plot
        distances = [r.get('distance', 0) for r in routes]
        axes[0].bar(bus_ids, distances, color='skyblue', edgecolor='black')
        axes[0].set_ylabel('Distance (km)', fontsize=10)
        axes[0].set_title('Route Distance', fontweight='bold')
        axes[0].grid(axis='y', alpha=0.3)
        
        # Time plot
        times = [r.get('time_minutes', 0) for r in routes]
        axes[1].bar(bus_ids, times, color='lightcoral', edgecolor='black')
        axes[1].set_ylabel('Time (minutes)', fontsize=10)
        axes[1].set_title('Estimated Travel Time', fontweight='bold')
        axes[1].grid(axis='y', alpha=0.3)
        
        # Load plot
        loads = [r.get('total_load', 0) for r in routes]
        axes[2].bar(bus_ids, loads, color='lightgreen', edgecolor='black')
        axes[2].set_ylabel('Load (passengers)', fontsize=10)
        axes[2].set_title('Total Load', fontweight='bold')
        axes[2].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Statistics plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()


if __name__ == "__main__":
    # Example usage
    from data_generator import DataGenerator
    from route_optimizer import RouteOptimizer
    
    generator = DataGenerator()
    dataset = generator.generate_dataset(num_pickup_points=20)
    
    optimizer = RouteOptimizer()
    result = optimizer.solve_tsp_with_depot(
        dataset['pickup_locations'],
        dataset['depot'],
        dataset['destination'],
        method='hybrid'
    )
    
    visualizer = RouteVisualizer()
    visualizer.plot_route(
        dataset['pickup_locations'],
        result['route'],
        dataset['depot'],
        dataset['destination'],
        title="Optimized Bus Route"
    )
