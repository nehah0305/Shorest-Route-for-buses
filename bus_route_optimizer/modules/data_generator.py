"""
Data Generator Module: Create sample datasets for bus route optimization

Generates pickup locations on an abstract 2-D map using x/y coordinates
(no real-world geography — pure custom map).

Coordinate system:
  • (0, 0)               = bottom-left corner
  • (map_width, map_height) = top-right corner
  • 1 unit  ≈  1 abstract "block" (used as-is for distance/time/fuel maths)
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Optional


class DataGenerator:
    """Generate sample bus route data on a custom abstract 2-D map."""

    def __init__(self, seed: int = 42):
        """
        Initialize data generator with a random seed for reproducibility.

        Args:
            seed: NumPy random seed
        """
        self.seed = seed
        np.random.seed(seed)

    def generate_pickup_locations(
        self,
        num_locations: int = 30,
        map_width: float = 100.0,
        map_height: float = 100.0,
        margin: float = 8.0,
        custom_locations: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Generate random pickup locations uniformly on the custom map.

        Args:
            num_locations: Number of pickup points to generate
            map_width:     Width  of the map in abstract units
            map_height:    Height of the map in abstract units
            margin:        Keep stops this many units away from the edges
            custom_locations: If provided, return these directly (shape (n,2))

        Returns:
            Array of shape (num_locations, 2) with [x, y]
        """
        if custom_locations is not None:
            return np.asarray(custom_locations, dtype=float)

        xs = np.random.uniform(margin, map_width  - margin, num_locations)
        ys = np.random.uniform(margin, map_height - margin, num_locations)
        return np.column_stack([xs, ys])

    def generate_dataset(
        self,
        num_pickup_points: int = 30,
        num_buses: int = 2,
        bus_capacity: int = 50,
        map_width: float = 100.0,
        map_height: float = 100.0,
        margin: float = 8.0,
        depot: Optional[Tuple[float, float]] = None,
        destination: Optional[Tuple[float, float]] = None,
        custom_locations: Optional[np.ndarray] = None,
    ) -> Dict:
        """
        Generate a complete dataset for bus route optimisation.

        Args:
            num_pickup_points: Number of pickup stops
            num_buses:         Number of buses available
            bus_capacity:      Capacity per bus (passengers)
            map_width:         Map width  in abstract units
            map_height:        Map height in abstract units
            margin:            Edge margin used by the random generator
            depot:             (x, y) of start location  [default: near bottom-left]
            destination:       (x, y) of final destination [default: near top-right]
            custom_locations:  Pre-defined stop positions, shape (n, 2)

        Returns:
            Dict with keys:
                pickup_locations, depot, destination,
                num_buses, bus_capacity, demands, time_windows,
                num_pickup_points, map_width, map_height
        """
        if depot is None:
            depot = (margin / 2, margin / 2)

        if destination is None:
            destination = (map_width - margin / 2, map_height - margin / 2)

        pickup_locations = self.generate_pickup_locations(
            num_pickup_points, map_width, map_height, margin, custom_locations
        )
        # Actual count may differ when custom_locations is supplied
        n = len(pickup_locations)

        demands      = np.random.randint(1, bus_capacity // 2 + 1, n)
        time_windows = self._generate_time_windows(n)

        return {
            'pickup_locations':  pickup_locations,
            'depot':             np.array(depot, dtype=float),
            'destination':       np.array(destination, dtype=float),
            'num_buses':         num_buses,
            'bus_capacity':      bus_capacity,
            'demands':           demands,
            'time_windows':      time_windows,
            'num_pickup_points': n,
            'map_width':         map_width,
            'map_height':        map_height,
        }
    
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
        """Save dataset to CSV files for later use."""
        pd.DataFrame(
            dataset['pickup_locations'], columns=['x', 'y']
        ).to_csv(f"{filepath}_pickup_locations.csv", index=False)

        pd.DataFrame({
            'depot_x':       [dataset['depot'][0]],
            'depot_y':       [dataset['depot'][1]],
            'destination_x': [dataset['destination'][0]],
            'destination_y': [dataset['destination'][1]],
            'num_buses':     [dataset['num_buses']],
            'bus_capacity':  [dataset['bus_capacity']],
            'map_width':     [dataset.get('map_width', 100)],
            'map_height':    [dataset.get('map_height', 100)],
            'num_pickup_points': [dataset['num_pickup_points']],
        }).to_csv(f"{filepath}_metadata.csv", index=False)

        pd.DataFrame({
            'demand':         dataset['demands'],
            'earliest_time':  dataset['time_windows'][:, 0],
            'latest_time':    dataset['time_windows'][:, 1],
        }).to_csv(f"{filepath}_demands.csv", index=False)

        print(f"Dataset saved to {filepath}_*.csv")

    def load_dataset(self, filepath: str) -> Dict:
        """Load dataset from previously saved CSV files."""
        df_p = pd.read_csv(f"{filepath}_pickup_locations.csv")
        df_m = pd.read_csv(f"{filepath}_metadata.csv")
        df_d = pd.read_csv(f"{filepath}_demands.csv")

        return {
            'pickup_locations':  df_p[['x', 'y']].values,
            'depot':             np.array([df_m['depot_x'].iloc[0],       df_m['depot_y'].iloc[0]]),
            'destination':       np.array([df_m['destination_x'].iloc[0], df_m['destination_y'].iloc[0]]),
            'num_buses':         int(df_m['num_buses'].iloc[0]),
            'bus_capacity':      int(df_m['bus_capacity'].iloc[0]),
            'demands':           df_d['demand'].values,
            'time_windows':      df_d[['earliest_time', 'latest_time']].values,
            'num_pickup_points': len(df_p),
            'map_width':         float(df_m.get('map_width',  100).iloc[0]),
            'map_height':        float(df_m.get('map_height', 100).iloc[0]),
        }


if __name__ == "__main__":
    gen = DataGenerator(seed=42)
    ds  = gen.generate_dataset(num_pickup_points=30, num_buses=2)
    gen.save_dataset(ds, "sample_dataset")
    print(f"Pickup points: {ds['num_pickup_points']}")
    print(f"Depot: {ds['depot']},  Destination: {ds['destination']}")
