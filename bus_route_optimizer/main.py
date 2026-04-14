"""
Main Orchestrator: Coordinate all components for complete bus route optimization

This module orchestrates clustering, route optimization, and reinforcement learning
to provide end-to-end bus route optimization with dynamic improvements.
"""

import numpy as np
from typing import Dict, List, Optional
import time
from modules.data_generator import DataGenerator
from modules.clustering import Clustering
from modules.route_optimizer import RouteOptimizer
from modules.visualization import RouteVisualizer
from modules.reinforcement_learning import RLOptimizer


class BusRouteOptimization:
    """Complete bus route optimization system."""
    
    def __init__(self, use_rl: bool = True):
        """
        Initialize optimizer system.
        
        Args:
            use_rl: Whether to use reinforcement learning for dynamic optimization
        """
        self.clustering = Clustering()
        self.route_optimizer = RouteOptimizer()
        self.visualizer = RouteVisualizer()
        self.rl_optimizer = RLOptimizer() if use_rl else None
        self.use_rl = use_rl
    
    def optimize_routes(
        self,
        dataset: Dict,
        clustering_method: str = 'kmeans',
        routing_method: str = 'hybrid',
        visualize: bool = True
    ) -> Dict:
        """
        Complete pipeline: cluster locations and optimize routes.
        
        Args:
            dataset: Dataset dictionary from DataGenerator
            clustering_method: 'kmeans' or 'dbscan'
            routing_method: 'nn' (nearest neighbor), 'hybrid', or 'dp'
            visualize: Whether to generate visualizations
        
        Returns:
            Dictionary containing:
                - routes: Optimized routes for each bus
                - total_distance: Sum of all route distances
                - clusters: Cluster assignments
                - statistics: Route statistics
        """
        print("=" * 60)
        print("BUS ROUTE OPTIMIZATION SYSTEM")
        print("=" * 60)
        
        # Step 1: Clustering
        print("\n[1/4] CLUSTERING NEARBY PICKUP POINTS...")
        pickup_locations = dataset['pickup_locations']
        
        if clustering_method == 'kmeans':
            clustering_result = self.clustering.kmeans_clustering(
                pickup_locations,
                num_buses=dataset['num_buses']
            )
        else:
            clustering_result = self.clustering.dbscan_clustering(pickup_locations)
        
        num_clusters = clustering_result['num_clusters']
        print(f"  ✓ Created {num_clusters} clusters")
        
        # Step 2: Route optimization for each cluster
        print("\n[2/4] OPTIMIZING ROUTES FOR EACH CLUSTER...")
        routes = []
        route_details = []
        total_distance = 0
        total_time = 0
        
        for cluster_id, point_indices in clustering_result['clusters'].items():
            if cluster_id == -1:  # Skip noise points
                continue
            
            cluster_points = pickup_locations[point_indices]
            
            # Solve TSP for this cluster
            result = self.route_optimizer.solve_tsp_with_depot(
                cluster_points,
                dataset['depot'],
                dataset['destination'],
                method=routing_method
            )
            
            # Map back to original indices
            original_route = [point_indices[i] for i in result['route']]
            
            # Calculate additional metrics
            distance = result['distance']
            time_minutes = self.route_optimizer.estimate_travel_time(distance)
            fuel_liters = self.route_optimizer.estimate_fuel_consumption(distance)
            total_load = np.sum(dataset['demands'][original_route])
            
            routes.append(original_route)
            route_details.append({
                'cluster_id': cluster_id,
                'route': original_route,
                'distance': distance,
                'time_minutes': time_minutes,
                'fuel_liters': fuel_liters,
                'total_load': total_load,
                'num_stops': len(original_route),
                'waypoints': result['waypoints']
            })
            
            total_distance += distance
            total_time += time_minutes
            
            print(f"  ✓ Cluster {cluster_id}: {len(point_indices)} stops, "
                  f"{distance:.1f} km, {time_minutes:.0f} min")
        
        # Step 3: ReinformentLearning optimization (if enabled)
        if self.use_rl:
            print("\n[3/4] REINFORCEMENT LEARNING OPTIMIZATION...")
            # Prepare data for RL
            rl_routes_data = []
            for detail in route_details:
                rl_routes_data.append({
                    'route': detail['route'],
                    'estimated_distance': detail['distance'],
                    'actual_distance': detail['distance'] * np.random.uniform(0.95, 1.05),  # Simulate variation
                    'estimated_time': detail['time_minutes'],
                    'actual_time': detail['time_minutes'] * np.random.uniform(0.95, 1.05),
                    'is_on_time': np.random.random() > 0.1  # 90% on-time
                })
            
            rl_reward = self.rl_optimizer.learn_from_episode(rl_routes_data, time_of_day=8)
            print(f"  ✓ RL Episode reward: {rl_reward:.2f}")
            rl_stats = self.rl_optimizer.get_learning_statistics()
            print(f"  ✓ Episodes trained: {rl_stats.get('episodes', 0)}")
        else:
            print("\n[3/4] SKIPPING REINFORCEMENT LEARNING...")
        
        # Step 4: Visualization
        if visualize:
            print("\n[4/4] GENERATING VISUALIZATIONS...")
            self._create_visualizations(
                pickup_locations,
                routes,
                route_details,
                clustering_result,
                dataset
            )
            print("  ✓ Visualizations generated")
        else:
            print("\n[4/4] SKIPPING VISUALIZATION...")
        
        # Summary
        print("\n" + "=" * 60)
        print("OPTIMIZATION RESULTS")
        print("=" * 60)
        print(f"Total Distance: {total_distance:.1f} km")
        print(f"Total Time: {total_time:.0f} minutes ({total_time/60:.1f} hours)")
        print(f"Estimated Fuel: {self.route_optimizer.estimate_fuel_consumption(total_distance):.1f} liters")
        print(f"Number of Buses Used: {len(routes)}")
        
        return {
            'routes': routes,
            'route_details': route_details,
            'total_distance': total_distance,
            'total_time': total_time,
            'clusters': clustering_result,
            'num_buses': len(routes)
        }
    
    def _create_visualizations(
        self,
        pickup_locations: np.ndarray,
        routes: List[List[int]],
        route_details: List[Dict],
        clustering_result: Dict,
        dataset: Dict
    ) -> None:
        """
        Create all visualizations for optimization results.
        
        Args:
            pickup_locations: Array of pickup locations
            routes: List of optimized routes
            route_details: Details for each route
            clustering_result: Clustering results
            dataset: Original dataset
        """
        # Cluster visualization
        self.visualizer.plot_clusters(
            pickup_locations,
            clustering_result['clusters'],
            centers=clustering_result.get('centers'),
            depot=dataset['depot'],
            title="Pickup Location Clusters",
            save_path="visualizations/01_clusters.png"
        )
        
        # Individual route visualizations
        for i, (route, details) in enumerate(zip(routes, route_details)):
            self.visualizer.plot_route(
                pickup_locations,
                route,
                dataset['depot'],
                dataset['destination'],
                title=f"Bus {i+1} Route (Distance: {details['distance']:.1f} km)",
                save_path=f"visualizations/02_route_bus_{i+1}.png"
            )
        
        # All routes on one map
        self.visualizer.plot_multiple_routes(
            pickup_locations,
            routes,
            dataset['depot'],
            dataset['destination'],
            title="All Bus Routes",
            save_path="visualizations/03_all_routes.png"
        )
        
        # Statistics plot
        self.visualizer.plot_route_statistics(
            route_details,
            save_path="visualizations/04_statistics.png"
        )
        
        # Interactive map for first route (if available)
        if routes:
            try:
                self.visualizer.create_interactive_map(
                    pickup_locations,
                    routes[0],
                    dataset['depot'],
                    dataset['destination'],
                    title="Interactive Route Map - Bus 1",
                    save_path="visualizations/05_interactive_map.html"
                )
            except Exception as e:
                print(f"  Note: Could not create interactive map: {e}")
    
    def handle_absence(
        self,
        original_routes: Dict,
        absent_location_index: int,
        dataset: Dict
    ) -> Dict:
        """
        Dynamic rerouting when a student/employee is absent.
        
        Args:
            original_routes: Original optimization results
            absent_location_index: Index of absent location
            dataset: Original dataset
        
        Returns:
            Updated optimization results
        """
        print(f"\n[DYNAMIC REROUTING] Location {absent_location_index} is absent")
        
        # Remove absent location from routes
        pickup_locations_new = np.delete(
            dataset['pickup_locations'],
            absent_location_index,
            axis=0
        )
        
        # Adjust route indices
        new_routes = []
        for route in original_routes['routes']:
            new_route = [
                i - 1 if i > absent_location_index else i
                for i in route
                if i != absent_location_index
            ]
            if new_route:
                new_routes.append(new_route)
        
        # Re-optimize
        dataset_new = dataset.copy()
        dataset_new['pickup_locations'] = pickup_locations_new
        dataset_new['num_pickup_points'] = len(pickup_locations_new)
        
        return self.optimize_routes(dataset_new, visualize=False)
    
    def handle_traffic_delay(
        self,
        route_id: int,
        delay_minutes: float,
        route_details: List[Dict]
    ) -> Dict:
        """
        Adjust estimates based on real-time traffic delays.
        
        Args:
            route_id: ID of delayed route (0-indexed)
            delay_minutes: Additional delay in minutes
            route_details: Current route details
        
        Returns:
            Updated route details with adjusted time windows
        """
        if route_id >= len(route_details):
            print(f"Route {route_id} not found")
            return {}
        
        route = route_details[route_id]
        route['actual_delay_minutes'] = delay_minutes
        route['updated_time_minutes'] = route['time_minutes'] + delay_minutes
        
        print(f"\n[TRAFFIC DELAY] Route {route_id}: +{delay_minutes:.0f} minutes delay")
        print(f"  Updated ETA: {route['updated_time_minutes']:.0f} minutes")
        
        # If using RL, learn from this delay
        if self.use_rl:
            self.rl_optimizer.learn_from_episode(
                [{
                    'route': route['route'],
                    'estimated_distance': route['distance'],
                    'actual_distance': route['distance'],
                    'estimated_time': route['time_minutes'],
                    'actual_time': route['updated_time_minutes'],
                    'is_on_time': False
                }],
                time_of_day=8
            )
        
        return route


def main():
    """Main execution example."""
    
    # Generate sample data
    print("Generating sample dataset...")
    generator = DataGenerator(seed=42)
    dataset = generator.generate_dataset(
        num_pickup_points=30,
        num_buses=2,
        bus_capacity=50,
        city_center=(40.7128, -74.0060),
        radius_km=10.0
    )
    print(f"✓ Generated dataset with {dataset['num_pickup_points']} pickup points")
    
    # Create and run optimizer
    optimizer = BusRouteOptimization(use_rl=True)
    
    # Optimize routes
    start_time = time.time()
    results = optimizer.optimize_routes(
        dataset,
        clustering_method='kmeans',
        routing_method='hybrid',
        visualize=True
    )
    elapsed = time.time() - start_time
    print(f"\n⏱ Optimization took {elapsed:.2f} seconds")
    
    # Example: Handle absence
    print("\n" + "=" * 60)
    print("TESTING DYNAMIC REROUTING (Student Absence)")
    print("=" * 60)
    updated_results = optimizer.handle_absence(results, absent_location_index=5, dataset=dataset)
    print(f"✓ Rerouting complete: {updated_results['total_distance']:.1f} km")
    
    # Example: Handle traffic delay
    print("\n" + "=" * 60)
    print("TESTING TRAFFIC DELAY HANDLING")
    print("=" * 60)
    delayed_route = optimizer.handle_traffic_delay(
        route_id=0,
        delay_minutes=15.0,
        route_details=results['route_details']
    )
    
    print("\n" + "=" * 60)
    print("✓ SYSTEM READY FOR PRODUCTION")
    print("=" * 60)


if __name__ == "__main__":
    main()
