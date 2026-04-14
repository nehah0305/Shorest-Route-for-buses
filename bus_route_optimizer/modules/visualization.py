"""
Visualization Module: Static plots of routes and clusters on the custom 2-D map.

Uses matplotlib for offline/CLI rendering.  The Streamlit UI renders its own
interactive Plotly map — this module is used only by main.py when
visualize=True.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


def _orthogonal_polyline(points: List[np.ndarray]) -> List[np.ndarray]:
    """Insert corners so each segment moves horizontally or vertically."""
    if not points:
        return []

    polyline = [np.array(points[0], dtype=float)]
    for start, end in zip(points, points[1:]):
        start = np.array(start, dtype=float)
        end = np.array(end, dtype=float)
        if np.allclose(start, end):
            continue
        corner = np.array([end[0], start[1]], dtype=float)
        if not np.allclose(polyline[-1], corner):
            polyline.append(corner)
        if not np.allclose(polyline[-1], end):
            polyline.append(end)
    return polyline


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
                cluster_points[:, 0], cluster_points[:, 1],   # x, y
                c=[color], marker=marker, s=100, label=label,
                alpha=0.7, edgecolors='black', linewidth=0.5
            )

        # Plot cluster centers
        if centers is not None:
            ax.scatter(
                centers[:, 0], centers[:, 1],   # x, y
                marker='*', s=500, c='red', edgecolors='black',
                linewidth=1, label='Cluster Centers', zorder=5
            )

        # Plot depot
        if depot is not None:
            ax.scatter(
                depot[0], depot[1],   # x, y
                marker='s', s=300, c='green', edgecolors='black',
                linewidth=1.5, label='Depot', zorder=5
            )

        ax.set_xlabel('X', fontsize=11)
        ax.set_ylabel('Y', fontsize=11)
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
            pickup_locations[:, 0], pickup_locations[:, 1],   # x, y
            c='blue', marker='o', s=100, alpha=0.6,
            edgecolors='black', linewidth=0.5, label='Pickup Points'
        )

        # Plot route path
        route_path = _orthogonal_polyline([depot] + pickup_locations[route[:-1]].tolist() + [destination])
        for i in range(len(route_path) - 1):
            start = route_path[i]
            end = route_path[i + 1]
            ax.annotate("", xy=(end[0], end[1]), xytext=(start[0], start[1]),
                        arrowprops=dict(arrowstyle="->", color="red", lw=1.5))

        # Plot depot and destination
        ax.scatter(
            depot[0], depot[1],   # x, y
            marker='s', s=300, c='green', edgecolors='black',
            linewidth=1.5, label='Depot', zorder=5
        )

        if not np.allclose(destination, depot):
            ax.scatter(
                destination[0], destination[1],   # x, y
                marker='^', s=300, c='purple', edgecolors='black',
                linewidth=1.5, label='Destination', zorder=5
            )

        # Add stop numbers
        for idx, point in enumerate(pickup_locations[route[:-1]]):
            ax.annotate(
                str(idx + 1), (point[0], point[1]),   # x, y
                fontsize=8, ha='center', va='center',
                bbox=dict(boxstyle='circle', facecolor='yellow', alpha=0.7)
            )

        ax.set_xlabel('X', fontsize=11)
        ax.set_ylabel('Y', fontsize=11)
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
            pickup_locations[:, 0], pickup_locations[:, 1],   # x, y
            c='lightblue', marker='o', s=100, alpha=0.6,
            edgecolors='black', linewidth=0.5, label='Pickup Points', zorder=2
        )

        # Plot each route
        for route_id, route in enumerate(routes):
            color = colors[route_id]
            route_path = _orthogonal_polyline([depot] + pickup_locations[route[:-1]].tolist() + [destination])

            # Plot path
            for i in range(len(route_path) - 1):
                start = route_path[i]
                end = route_path[i + 1]
                ax.plot(
                    [start[0], end[0]], [start[1], end[1]],   # x, y
                    color=color, linewidth=2, alpha=0.7,
                    label=f'Bus {route_id + 1}' if i == 0 else ''
                )

        # Plot depot and destination
        ax.scatter(
            depot[0], depot[1],   # x, y
            marker='s', s=300, c='green', edgecolors='black',
            linewidth=1.5, label='Depot', zorder=5
        )

        if not np.allclose(destination, depot):
            ax.scatter(
                destination[0], destination[1],   # x, y
                marker='^', s=300, c='purple', edgecolors='black',
                linewidth=1.5, label='Destination', zorder=5
            )

        ax.set_xlabel('X', fontsize=11)
        ax.set_ylabel('Y', fontsize=11)
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
        save_path: Optional[str] = None,
    ) -> None:
        """
        Stub: interactive map is now rendered in the Streamlit UI via Plotly.
        Saves a simple matplotlib PNG when save_path is supplied.
        """
        self.plot_route(pickup_locations, route, depot, destination,
                        title=title, save_path=save_path)
    
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
