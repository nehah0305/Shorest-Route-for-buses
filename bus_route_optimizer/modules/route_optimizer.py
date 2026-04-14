"""
Route Optimization Module: Solve TSP and find optimal paths within clusters

Implements multiple algorithms for finding near-optimal routes:
- Nearest Neighbor (greedy heuristic)
- 2-opt local search optimization
- Dynamic Programming (for small clusters)
- A* pathfinding for modified TSP
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
import heapq
from itertools import permutations


class RouteOptimizer:
    """Optimize routes within clusters using various algorithms."""
    
    def __init__(self):
        """Initialize route optimizer."""
        self.distance_matrix = None
        self.best_route = None
        self.best_distance = float('inf')
    
    def calculate_distance_matrix(
        self,
        locations: np.ndarray,
        depot: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Calculate pairwise Euclidean distances between all locations (vectorised).

        Args:
            locations: Array of shape (n, 2) with [x, y] in abstract map units.
            depot:     Ignored (kept for API compatibility).

        Returns:
            Distance matrix of shape (n, n).
        """
        diff = locations[:, None, :] - locations[None, :, :]   # (n, n, 2)
        dist_matrix = np.sqrt((diff ** 2).sum(axis=2))
        np.fill_diagonal(dist_matrix, 0.0)
        self.distance_matrix = dist_matrix
        return dist_matrix

    @staticmethod
    def euclidean_distance(x1: float, y1: float, x2: float, y2: float) -> float:
        """Euclidean distance between two points in abstract map units."""
        return float(np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))

    # Keep old name as an alias so any external callers don't break.
    @staticmethod
    def haversine_distance(x1: float, y1: float, x2: float, y2: float) -> float:
        """Alias for euclidean_distance (replaces Haversine for custom maps)."""
        return float(np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))
    
    def nearest_neighbor(
        self,
        locations: np.ndarray,
        start_idx: int = 0
    ) -> Tuple[List[int], float]:
        """
        Greedy nearest neighbor algorithm for TSP.
        
        Fast heuristic that builds solution by always visiting nearest unvisited location.
        Time complexity: O(n^2)
        
        Args:
            locations: Array of shape (n, 2) with [lat, lon]
            start_idx: Starting location index
        
        Returns:
            Tuple of (route, total_distance)
                - route: List of location indices in order
                - total_distance: Total distance of route in km
        """
        if self.distance_matrix is None:
            self.calculate_distance_matrix(locations)
        
        n = len(locations)
        unvisited = set(range(n))
        current = start_idx
        route = [current]
        unvisited.remove(current)
        
        total_distance = 0
        
        # Greedily visit nearest unvisited location
        while unvisited:
            nearest = min(unvisited, key=lambda x: self.distance_matrix[current, x])
            total_distance += self.distance_matrix[current, nearest]
            route.append(nearest)
            unvisited.remove(nearest)
            current = nearest
        
        # Return to start
        total_distance += self.distance_matrix[current, start_idx]
        route.append(start_idx)
        
        return route, total_distance
    
    def two_opt_optimization(
        self,
        route: List[int],
        max_iterations: int = 1000,
        improvement_threshold: float = 0.001
    ) -> Tuple[List[int], float]:
        """
        2-opt local search optimization for TSP.
        
        Iteratively improves route by removing crossing edges and reconnecting optimally.
        Time complexity: O(n^2) per iteration
        
        Args:
            route: Initial route (list of indices)
            max_iterations: Maximum number of iterations
            improvement_threshold: Stop if improvement less than this
        
        Returns:
            Tuple of (optimized_route, total_distance)
        """
        best_route = route[:]
        best_distance = self._calculate_route_distance(best_route)
        improved = True
        iteration = 0
        
        while improved and iteration < max_iterations:
            improved = False
            iteration += 1
            
            for i in range(1, len(route) - 2):
                for j in range(i + 1, len(route)):
                    # Create new route by reversing segment between i and j
                    new_route = route[:i] + route[i:j][::-1] + route[j:]
                    new_distance = self._calculate_route_distance(new_route)
                    
                    # Check if improvement exceeds threshold
                    if new_distance < best_distance - improvement_threshold:
                        best_route = new_route
                        best_distance = new_distance
                        route = new_route
                        improved = True
                        break
                
                if improved:
                    break
        
        return best_route, best_distance
    
    def dynamic_programming_tsp(
        self,
        locations: np.ndarray
    ) -> Tuple[List[int], float]:
        """
        Solve TSP using Dynamic Programming (Held-Karp algorithm).
        
        Optimal solution but exponential time complexity O(n^2 * 2^n).
        Only practical for small instances (n <= 20).
        
        Args:
            locations: Array of shape (n, 2) with [lat, lon]
        
        Returns:
            Tuple of (optimal_route, total_distance)
        """
        if self.distance_matrix is None:
            self.calculate_distance_matrix(locations)
        
        n = len(locations)
        
        # Only use for small instances
        if n > 20:
            print(f"Warning: DP-TSP slow for n={n}. Using nearest neighbor instead.")
            return self.nearest_neighbor(locations)
        
        # Memoization table: dp[mask][i] = (min_cost, next_city)
        dp = {}
        
        def tsp_dp(mask: int, pos: int) -> float:
            """Recursive DP solver with memoization."""
            if mask == (1 << n) - 1:  # All cities visited
                return self.distance_matrix[pos, 0]  # Return to start
            
            if (mask, pos) in dp:
                return dp[(mask, pos)]
            
            ans = float('inf')
            for city in range(n):
                if (mask & (1 << city)) == 0:  # City not visited
                    new_ans = self.distance_matrix[pos, city] + tsp_dp(
                        mask | (1 << city), city
                    )
                    ans = min(ans, new_ans)
            
            dp[(mask, pos)] = ans
            return ans
        
        # Calculate minimum distance
        min_distance = tsp_dp(1, 0)  # Start at city 0
        
        # Reconstruct route
        route = self._reconstruct_dp_route(n, min_distance, dp)
        
        return route, min_distance
    
    def _reconstruct_dp_route(self, n: int, min_distance: float, dp: Dict) -> List[int]:
        """Reconstruct TSP route from DP solution."""
        route = [0]
        mask = 1
        pos = 0
        
        while mask != (1 << n) - 1:
            for city in range(n):
                if (mask & (1 << city)) == 0:
                    new_mask = mask | (1 << city)
                    new_cost = self.distance_matrix[pos, city] + dp.get(
                        (new_mask, city), float('inf')
                    )
                    
                    if new_cost == dp.get((mask, pos), float('inf')):
                        route.append(city)
                        mask = new_mask
                        pos = city
                        break
        
        route.append(0)  # Return to start
        return route
    
    def hybrid_optimization(
        self,
        locations: np.ndarray,
        start_idx: int = 0
    ) -> Tuple[List[int], float]:
        """
        Hybrid approach: Start with nearest neighbor, then improve with 2-opt.
        
        Balances speed and solution quality.
        
        Args:
            locations: Array of shape (n, 2) with [lat, lon]
            start_idx: Starting location index
        
        Returns:
            Tuple of (optimized_route, total_distance)
        """
        # Start with nearest neighbor
        route, _ = self.nearest_neighbor(locations, start_idx)
        
        # Improve with 2-opt
        optimized_route, distance = self.two_opt_optimization(route)
        
        return optimized_route, distance
    
    def solve_tsp_with_depot(
        self,
        pickup_locations: np.ndarray,
        depot: np.ndarray,
        destination: Optional[np.ndarray] = None,
        method: str = 'hybrid'
    ) -> Dict:
        """
        Solve TSP for pickup locations, starting and ending at depot.

        The distance matrix is built from pickup_locations only so that
        route indices map 1-to-1 to pickup_locations rows (no off-by-one).
        The depot→first-stop and last-stop→destination legs are appended
        to the total distance afterwards.

        Args:
            pickup_locations: Array of shape (n, 2) with [lat, lon]
            depot: Starting location
            destination: Final destination (defaults to depot)
            method: 'nn', 'dp', or 'hybrid'

        Returns:
            Dict with keys: route, distance, waypoints, num_stops
        """
        if destination is None:
            destination = depot

        if len(pickup_locations) == 0:
            return {'route': [], 'distance': 0.0, 'waypoints': [depot, destination], 'num_stops': 0}

        # Build distance matrix for pickup points only (correct indices)
        self.calculate_distance_matrix(pickup_locations)

        if method == 'nn':
            route, inner_distance = self.nearest_neighbor(pickup_locations, 0)
        elif method == 'dp':
            route, inner_distance = self.dynamic_programming_tsp(pickup_locations)
        elif method == 'hybrid':
            route, inner_distance = self.hybrid_optimization(pickup_locations, 0)
        else:
            raise ValueError(f"Unknown method: {method}")

        # The NN/DP routes include a return-to-start leg; remove it for an
        # open path (depot → pickups → destination).
        visit_order = route[:-1]  # drop the closing "return to 0" index

        # Add depot→first and last→destination distances
        first_stop = pickup_locations[visit_order[0]]
        last_stop  = pickup_locations[visit_order[-1]]
        depot_to_first = self.euclidean_distance(
            depot[0], depot[1], first_stop[0], first_stop[1]
        )
        last_to_dest = self.euclidean_distance(
            last_stop[0], last_stop[1], destination[0], destination[1]
        )
        # Remove the artificial return-to-start leg that NN added
        return_leg = self.distance_matrix[visit_order[-1], visit_order[0]]
        total_distance = inner_distance - return_leg + depot_to_first + last_to_dest

        waypoints = [depot] + [pickup_locations[i] for i in visit_order] + [destination]

        return {
            'route': visit_order,
            'distance': float(total_distance),
            'waypoints': waypoints,
            'num_stops': len(pickup_locations)
        }
    
    def _calculate_route_distance(self, route: List[int]) -> float:
        """Calculate total distance for a given route."""
        if self.distance_matrix is None:
            return 0
        
        distance = 0
        for i in range(len(route) - 1):
            distance += self.distance_matrix[route[i], route[i + 1]]
        
        return distance
    
    def estimate_travel_time(self, distance_km: float, avg_speed_kmh: float = 30) -> float:
        """
        Estimate travel time given distance.
        
        Args:
            distance_km: Total distance in kilometers
            avg_speed_kmh: Average speed in km/h (default: 30 km/h for city traffic)
        
        Returns:
            Estimated time in minutes
        """
        return (distance_km / avg_speed_kmh) * 60
    
    def estimate_fuel_consumption(
        self,
        distance_km: float,
        fuel_efficiency_kmpl: float = 5.0
    ) -> float:
        """
        Estimate fuel consumption.
        
        Args:
            distance_km: Total distance in kilometers
            fuel_efficiency_kmpl: Fuel efficiency in km per liter
        
        Returns:
            Estimated fuel in liters
        """
        return distance_km / fuel_efficiency_kmpl


if __name__ == "__main__":
    # Example usage
    from data_generator import DataGenerator
    from clustering import Clustering
    
    generator = DataGenerator()
    dataset = generator.generate_dataset(num_pickup_points=15)
    
    optimizer = RouteOptimizer()
    
    # Solve TSP
    result = optimizer.solve_tsp_with_depot(
        dataset['pickup_locations'],
        dataset['depot'],
        dataset['destination'],
        method='hybrid'
    )
    
    print(f"Optimized Route Distance: {result['distance']:.2f} km")
    print(f"Travel Time: {optimizer.estimate_travel_time(result['distance']):.1f} minutes")
    print(f"Fuel Consumption: {optimizer.estimate_fuel_consumption(result['distance']):.2f} liters")
    print(f"Route: {result['route']}")
