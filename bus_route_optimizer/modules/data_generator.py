"""
Data Generator Module: Create sample datasets for bus route optimization

Generates realistic pickup locations, depot, and destination coordinates
with optional time window and demand constraints.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Optional


class DataGenerator:
    """Generate sample bus route data for testing and prototyping."""
    
    def __init__(self, seed: int = 42):
        """
        Initialize data generator with random seed for reproducibility.
        
        Args:
            seed: Random seed for numpy
        """
        self.seed = seed
        np.random.seed(seed)
    
    def generate_pickup_locations(
        self,
        num_locations: int = 30,
        city_center: Tuple[float, float] = (40.7128, -74.0060),
        radius_km: float = 10.0
    ) -> np.ndarray:
        """
        Generate random pickup locations within a specified radius from city center.
        
        Uses Haversine distance approximation for realistic geographic distribution.
        
        Args:
            num_locations: Number of pickup points to generate
            city_center: (latitude, longitude) of reference point (default: NYC)
            radius_km: Maximum radius in kilometers
        
        Returns:
            Array of shape (num_locations, 2) with [latitude, longitude]
        """
        # Approximate km to degrees: 1km ≈ 0.009 degrees
        radius_degrees = radius_km * 0.009
        
        lat_center, lon_center = city_center
        
        # Generate random points in circular distribution
        angles = np.random.uniform(0, 2 * np.pi, num_locations)
        distances = np.random.uniform(0, radius_degrees, num_locations)
        
        latitudes = lat_center + distances * np.cos(angles)
        longitudes = lon_center + distances * np.sin(angles)
        
        pickup_locations = np.column_stack([latitudes, longitudes])
        return pickup_locations
    
    def generate_dataset(
        self,
        num_pickup_points: int = 30,
        num_buses: int = 2,
        bus_capacity: int = 50,
        depot: Optional[Tuple[float, float]] = None,
        destination: Optional[Tuple[float, float]] = None,
        city_center: Tuple[float, float] = (40.7128, -74.0060),
        radius_km: float = 10.0
    ) -> Dict:
        """
        Generate complete dataset for bus route optimization.
        
        Args:
            num_pickup_points: Number of pickup locations
            num_buses: Number of buses available
            bus_capacity: Capacity of each bus
            depot: Starting location (default: city center)
            destination: Final destination (default: city center + offset)
            city_center: Reference city center
            radius_km: Radius for pickup location generation
        
        Returns:
            Dictionary containing:
                - pickup_locations: Array of pickup coordinates
                - depot: Start location
                - destination: End location
                - num_buses: Number of buses
                - bus_capacity: Capacity per bus
                - demands: Demand at each pickup location
                - time_windows: Optional time windows for pickups
        """
        if depot is None:
            depot = city_center
        
        if destination is None:
            # Offset destination slightly from depot
            destination = (city_center[0] + 0.05, city_center[1] + 0.05)
        
        # Generate pickup locations
        pickup_locations = self.generate_pickup_locations(
            num_pickup_points, city_center, radius_km
        )
        
        # Generate demands (number of passengers) for each pickup
        demands = np.random.randint(1, bus_capacity // 2 + 1, num_pickup_points)
        
        # Generate time windows (early morning to late afternoon)
        time_windows = self._generate_time_windows(num_pickup_points)
        
        dataset = {
            'pickup_locations': pickup_locations,
            'depot': np.array(depot),
            'destination': np.array(destination),
            'num_buses': num_buses,
            'bus_capacity': bus_capacity,
            'demands': demands,
            'time_windows': time_windows,
            'num_pickup_points': num_pickup_points
        }
        
        return dataset
    
    def _generate_time_windows(self, num_locations: int) -> np.ndarray:
        """
        Generate time windows for pickup locations.
        
        Time windows represent acceptable pickup times in minutes from start.
        
        Args:
            num_locations: Number of locations
        
        Returns:
            Array of shape (num_locations, 2) with [earliest_time, latest_time]
        """
        # Window duration: 15-45 minutes
        window_durations = np.random.randint(15, 46, num_locations)
        
        # Earliest times distributed throughout morning (0-120 minutes)
        earliest_times = np.random.randint(0, 121, num_locations)
        latest_times = earliest_times + window_durations
        
        time_windows = np.column_stack([earliest_times, latest_times])
        return time_windows
    
    def save_dataset(self, dataset: Dict, filepath: str) -> None:
        """
        Save dataset to CSV files for later use.
        
        Args:
            dataset: Dictionary containing dataset information
            filepath: Base filepath (without extension)
        """
        # Save pickup locations
        df_pickup = pd.DataFrame(
            dataset['pickup_locations'],
            columns=['latitude', 'longitude']
        )
        df_pickup.to_csv(f"{filepath}_pickup_locations.csv", index=False)
        
        # Save metadata
        metadata = {
            'depot_lat': [dataset['depot'][0]],
            'depot_lon': [dataset['depot'][1]],
            'destination_lat': [dataset['destination'][0]],
            'destination_lon': [dataset['destination'][1]],
            'num_buses': [dataset['num_buses']],
            'bus_capacity': [dataset['bus_capacity']],
            'num_pickup_points': [dataset['num_pickup_points']]
        }
        df_metadata = pd.DataFrame(metadata)
        df_metadata.to_csv(f"{filepath}_metadata.csv", index=False)
        
        # Save demands and time windows
        df_demands = pd.DataFrame({
            'demand': dataset['demands'],
            'earliest_time': dataset['time_windows'][:, 0],
            'latest_time': dataset['time_windows'][:, 1]
        })
        df_demands.to_csv(f"{filepath}_demands.csv", index=False)
        
        print(f"Dataset saved to {filepath}_*.csv")
    
    def load_dataset(self, filepath: str) -> Dict:
        """
        Load dataset from saved CSV files.
        
        Args:
            filepath: Base filepath (without extension)
        
        Returns:
            Dictionary containing dataset information
        """
        df_pickup = pd.read_csv(f"{filepath}_pickup_locations.csv")
        df_metadata = pd.read_csv(f"{filepath}_metadata.csv")
        df_demands = pd.read_csv(f"{filepath}_demands.csv")
        
        dataset = {
            'pickup_locations': df_pickup[['latitude', 'longitude']].values,
            'depot': np.array([
                df_metadata['depot_lat'].values[0],
                df_metadata['depot_lon'].values[0]
            ]),
            'destination': np.array([
                df_metadata['destination_lat'].values[0],
                df_metadata['destination_lon'].values[0]
            ]),
            'num_buses': int(df_metadata['num_buses'].values[0]),
            'bus_capacity': int(df_metadata['bus_capacity'].values[0]),
            'demands': df_demands['demand'].values,
            'time_windows': df_demands[['earliest_time', 'latest_time']].values,
            'num_pickup_points': len(df_pickup)
        }
        
        return dataset


if __name__ == "__main__":
    # Example usage
    generator = DataGenerator(seed=42)
    dataset = generator.generate_dataset(num_pickup_points=30, num_buses=2)
    generator.save_dataset(dataset, "sample_dataset")
    print("Sample dataset generated successfully!")
    print(f"Number of pickup points: {dataset['num_pickup_points']}")
    print(f"Number of buses: {dataset['num_buses']}")
    print(f"Depot: {dataset['depot']}")
    print(f"Destination: {dataset['destination']}")
