"""
Utility Functions: Helper functions for bus route optimization system

Provides utility functions for data validation, format conversion,
and system diagnostics.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import json


class DataValidator:
    """Validate input data and constraints."""
    
    @staticmethod
    def validate_locations(locations: np.ndarray) -> bool:
        """Check if locations are valid coordinates."""
        if locations.shape[1] != 2:
            raise ValueError("Locations must have shape (n, 2) with [lat, lon]")
        
        if np.any((locations[:, 0] < -90) | (locations[:, 0] > 90)):
            raise ValueError("Latitude must be between -90 and 90 degrees")
        
        if np.any((locations[:, 1] < -180) | (locations[:, 1] > 180)):
            raise ValueError("Longitude must be between -180 and 180 degrees")
        
        return True
    
    @staticmethod
    def validate_demands(demands: np.ndarray, bus_capacity: int) -> bool:
        """Check if demands are reasonable."""
        if np.any(demands < 0):
            raise ValueError("Demands cannot be negative")
        
        if np.any(demands > bus_capacity):
            print(f"Warning: Some demands exceed bus capacity ({bus_capacity})")
        
        return True
    
    @staticmethod
    def validate_routes(routes: List[List[int]], num_locations: int) -> bool:
        """Check if routes are valid."""
        all_indices = set()
        
        for route in routes:
            for idx in route:
                if idx < 0 or idx >= num_locations:
                    raise ValueError(f"Invalid location index: {idx}")
                all_indices.add(idx)
        
        if len(all_indices) != num_locations:
            print(f"Warning: Not all locations covered in routes")
        
        return True


class MetricsCalculator:
    """Calculate and aggregate metrics."""
    
    @staticmethod
    def calculate_fleet_metrics(routes: List[Dict]) -> Dict:
        """Calculate aggregate fleet metrics."""
        total_distance = sum(r.get('distance', 0) for r in routes)
        total_time = sum(r.get('time_minutes', 0) for r in routes)
        total_load = sum(r.get('total_load', 0) for r in routes)
        total_fuel = sum(r.get('fuel_liters', 0) for r in routes)
        
        return {
            'total_distance': total_distance,
            'total_time': total_time,
            'total_load': total_load,
            'total_fuel': total_fuel,
            'avg_distance_per_route': total_distance / len(routes) if routes else 0,
            'avg_time_per_route': total_time / len(routes) if routes else 0,
            'num_routes': len(routes)
        }
    
    @staticmethod
    def calculate_efficiency_score(metrics: Dict, baseline_metrics: Dict = None) -> float:
        """Calculate efficiency score (0-100)."""
        if baseline_metrics is None:
            # Use simple distance-based metric
            return min(100, max(0, 100 - (metrics['total_distance'] / 10)))
        
        # Compare to baseline
        distance_ratio = metrics['total_distance'] / baseline_metrics['total_distance']
        time_ratio = metrics['total_time'] / baseline_metrics['total_time']
        
        avg_ratio = (distance_ratio + time_ratio) / 2
        efficiency = max(0, min(100, (1 - avg_ratio) * 100))
        
        return efficiency


def get_pixel_road_axes(map_width: float, map_height: float, rows: int = 96, cols: int = 96) -> Tuple[np.ndarray, np.ndarray]:
    """Return the row and column coordinates used for the pixel-map road grid."""
    # Keep these fractions aligned with app.py so snapping and drawn lanes match.
    row_fracs = (0.06, 0.14, 0.22, 0.30, 0.38, 0.46, 0.54, 0.62, 0.72, 0.82, 0.90)
    col_fracs = (0.05, 0.13, 0.21, 0.29, 0.37, 0.45, 0.54, 0.63, 0.72, 0.82, 0.91)
    road_rows = np.array([int(rows * frac) for frac in row_fracs], dtype=float)
    road_cols = np.array([int(cols * frac) for frac in col_fracs], dtype=float)
    row_step = float(map_height) / max(rows - 1, 1)
    col_step = float(map_width) / max(cols - 1, 1)
    road_y = np.clip(road_rows * row_step, 0.0, float(map_height))
    road_x = np.clip(road_cols * col_step, 0.0, float(map_width))
    return road_y, road_x


def snap_point_to_pixel_road(
    point: np.ndarray,
    map_width: float,
    map_height: float,
    rows: int = 96,
    cols: int = 96,
) -> np.ndarray:
    """Snap a point to the nearest pixel-road intersection."""
    road_y, road_x = get_pixel_road_axes(map_width, map_height, rows=rows, cols=cols)
    snapped = np.array(point, dtype=float).copy()
    snapped[0] = road_x[np.argmin(np.abs(road_x - snapped[0]))]
    snapped[1] = road_y[np.argmin(np.abs(road_y - snapped[1]))]
    return snapped


def build_pixel_road_path(
    points: List[np.ndarray],
    map_width: float,
    map_height: float,
    rows: int = 96,
    cols: int = 96,
) -> List[np.ndarray]:
    """Build an orthogonal path that stays on the pixel-map brown roads."""
    if not points:
        return []

    snapped_points = [snap_point_to_pixel_road(point, map_width, map_height, rows=rows, cols=cols) for point in points]
    path: List[np.ndarray] = [snapped_points[0]]

    for start, end in zip(snapped_points, snapped_points[1:]):
        if np.allclose(start, end):
            continue
        corner = np.array([end[0], start[1]], dtype=float)
        if not np.allclose(path[-1], corner):
            path.append(corner)
        if not np.allclose(path[-1], end):
            path.append(end)

    return path


class ReportGenerator:
    """Generate reports and summaries."""
    
    @staticmethod
    def generate_summary_report(results: Dict, filepath: str = None) -> str:
        """Generate text summary report."""
        report = []
        report.append("=" * 80)
        report.append("BUS ROUTE OPTIMIZATION REPORT")
        report.append("=" * 80)
        
        report.append(f"\nDataset Summary:")
        report.append(f"  Total Pickup Points: {results.get('num_pickup_points', 'N/A')}")
        report.append(f"  Number of Buses: {results.get('num_buses', 'N/A')}")
        report.append(f"  Number of Clusters: {results.get('num_clusters', 'N/A')}")
        
        report.append(f"\nOptimization Results:")
        report.append(f"  Total Distance: {results.get('total_distance', 0):.2f} km")
        report.append(f"  Total Time: {results.get('total_time', 0):.0f} minutes")
        report.append(f"  Total Fuel: {results.get('total_fuel', 0):.2f} liters")
        
        report.append(f"\nRoute Details:")
        for i, route in enumerate(results.get('route_details', [])):
            report.append(f"  Bus {i+1}:")
            report.append(f"    Distance: {route.get('distance', 0):.2f} km")
            report.append(f"    Time: {route.get('time_minutes', 0):.0f} min")
            report.append(f"    Stops: {route.get('num_stops', 0)}")
            report.append(f"    Load: {route.get('total_load', 0)} passengers")
        
        report.append("\n" + "=" * 80)
        
        report_text = "\n".join(report)
        
        if filepath:
            with open(filepath, 'w') as f:
                f.write(report_text)
        
        return report_text
    
    @staticmethod
    def generate_json_report(results: Dict, filepath: str = None) -> str:
        """Generate JSON format report."""
        report = {
            'summary': {
                'total_distance': float(results.get('total_distance', 0)),
                'total_time': float(results.get('total_time', 0)),
                'num_buses': results.get('num_buses', 0),
                'num_clusters': results.get('num_clusters', 0)
            },
            'routes': results.get('route_details', [])
        }
        
        # Convert numpy types to native Python types
        def convert_to_native(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_native(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native(item) for item in obj]
            return obj
        
        report = convert_to_native(report)
        report_json = json.dumps(report, indent=2)
        
        if filepath:
            with open(filepath, 'w') as f:
                f.write(report_json)
        
        return report_json


class PerformanceAnalyzer:
    """Analyze algorithm and system performance."""
    
    @staticmethod
    def compare_algorithms(method1_result: Dict, method2_result: Dict) -> Dict:
        """Compare two optimization methods."""
        dist1 = method1_result.get('distance', 0)
        dist2 = method2_result.get('distance', 0)
        
        distance_improvement = ((dist1 - dist2) / dist1 * 100) if dist1 > 0 else 0
        
        return {
            'method1_distance': dist1,
            'method2_distance': dist2,
            'distance_improvement': distance_improvement,
            'better_method': 'method2' if distance_improvement > 0 else 'method1',
            'improvement_magnitude': abs(distance_improvement)
        }
    
    @staticmethod
    def scalability_analysis(num_locations_list: List[int],
                            execution_times: List[float]) -> Dict:
        """Analyze scalability with increasing problem size."""
        if len(num_locations_list) != len(execution_times):
            raise ValueError("Lists must have same length")
        
        # Calculate growth rate
        growth_rates = []
        for i in range(1, len(num_locations_list)):
            size_ratio = num_locations_list[i] / num_locations_list[i-1]
            time_ratio = execution_times[i] / execution_times[i-1]
            growth_rates.append(time_ratio)
        
        avg_growth = np.mean(growth_rates) if growth_rates else 0
        
        # Estimate complexity (polynomial degree)
        if avg_growth > 0:
            complexity_estimate = np.log(avg_growth) / np.log(size_ratio)
        else:
            complexity_estimate = 0
        
        return {
            'num_locations': num_locations_list,
            'execution_times': execution_times,
            'growth_rates': growth_rates,
            'avg_growth_rate': avg_growth,
            'complexity_estimate': complexity_estimate,
            'complexity_class': 'O(n²)' if 1.8 < complexity_estimate < 2.2 else 'Other'
        }


# Example usage and diagnostics
if __name__ == "__main__":
    # Test validator
    print("Testing Data Validator...")
    locations = np.array([[40.7128, -74.0060], [40.7580, -73.9855]])
    demands = np.array([10, 15])
    
    try:
        DataValidator.validate_locations(locations)
        DataValidator.validate_demands(demands, bus_capacity=50)
        print("✓ Data validation passed")
    except Exception as e:
        print(f"✗ Validation error: {e}")
    
    # Test metrics calculator
    print("\nTesting Metrics Calculator...")
    sample_routes = [
        {'distance': 25.5, 'time_minutes': 45, 'total_load': 35, 'fuel_liters': 5.1},
        {'distance': 22.3, 'time_minutes': 40, 'total_load': 38, 'fuel_liters': 4.5}
    ]
    
    fleet_metrics = MetricsCalculator.calculate_fleet_metrics(sample_routes)
    print(f"Fleet Metrics: {fleet_metrics}")
    
    # Test report generator
    print("\nGenerating Sample Report...")
    sample_results = {
        'num_pickup_points': 30,
        'num_buses': 2,
        'num_clusters': 2,
        'total_distance': 47.8,
        'total_time': 85,
        'total_fuel': 9.6,
        'route_details': sample_routes
    }
    
    report = ReportGenerator.generate_summary_report(sample_results)
    print(report)
    
    print("\n✓ All utilities working correctly!")
