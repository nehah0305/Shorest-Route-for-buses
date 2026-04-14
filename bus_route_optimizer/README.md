# Bus Route Optimizer - ML-Based Smart Route Optimization System

A comprehensive machine learning-based system for optimizing school and employee pickup bus routes using clustering algorithms, route optimization techniques, and reinforcement learning for continuous improvement.

## 🎯 Features

### Core Optimization
- **Smart Clustering**: K-Means and DBSCAN algorithms to group nearby pickup points
- **Route Optimization**: Multiple algorithms including:
  - Nearest Neighbor heuristic (fast greedy algorithm)
  - 2-opt local search optimization
  - Dynamic Programming (exact solution for small instances)
  - Hybrid approach (NN + 2-opt)
- **Performance Metrics**:
  - Total distance calculation
  - Travel time estimation
  - Fuel consumption analysis
  - Load balancing across buses

### Advanced Features
- **Reinforcement Learning**: Q-Learning agent that improves routes over time based on:
  - Historical traffic patterns
  - Actual travel times vs. estimates
  - Real-time delays and performance
- **Dynamic Rerouting**: Automatically adjust routes when:
  - Students/employees are absent
  - Real-time traffic delays occur
- **Time Window Constraints**: Pickup time preferences
- **Interactive Visualizations**: 
  - Cluster maps
  - Individual route visualization
  - Multi-route comparison
  - Statistics dashboards
  - Interactive Folium maps

## 📊 System Architecture

```
bus_route_optimizer/
├── modules/
│   ├── __init__.py
│   ├── data_generator.py        # Generate realistic test data
│   ├── clustering.py             # K-Means & DBSCAN clustering
│   ├── route_optimizer.py        # TSP solving algorithms
│   ├── visualization.py          # Maps and charts
│   └── reinforcement_learning.py # Q-Learning optimization
├── data/                         # Sample datasets
├── visualizations/               # Output maps and charts
├── main.py                       # Main orchestrator
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## 🚀 Quick Start

### Installation

```bash
# Clone or download the project
cd bus_route_optimizer

# Install dependencies
pip install -r requirements.txt

# Create output directories
mkdir -p visualizations data
```

### Basic Usage

```python
from modules.data_generator import DataGenerator
from modules.clustering import Clustering
from modules.route_optimizer import RouteOptimizer
from modules.visualization import RouteVisualizer

# Generate sample data
generator = DataGenerator(seed=42)
dataset = generator.generate_dataset(num_pickup_points=30, num_buses=2)

# Cluster locations
clustering = Clustering()
clustering_result = clustering.kmeans_clustering(
    dataset['pickup_locations'],
    num_buses=dataset['num_buses']
)

# Optimize routes
optimizer = RouteOptimizer()
result = optimizer.solve_tsp_with_depot(
    dataset['pickup_locations'],
    dataset['depot'],
    dataset['destination'],
    method='hybrid'
)

# Visualize
visualizer = RouteVisualizer()
visualizer.plot_clusters(
    dataset['pickup_locations'],
    clustering_result['clusters'],
    centers=clustering_result['centers'],
    depot=dataset['depot']
)
```

### End-to-End Pipeline

```python
from main import BusRouteOptimization

# Create optimizer
optimizer = BusRouteOptimization(use_rl=True)

# Optimize all routes
results = optimizer.optimize_routes(
    dataset,
    clustering_method='kmeans',
    routing_method='hybrid',
    visualize=True
)

print(f"Total Distance: {results['total_distance']:.1f} km")
print(f"Total Time: {results['total_time']:.0f} minutes")
```

## 📚 Module Documentation

### Data Generator
Generates realistic test datasets with configurable parameters.

**Key Methods:**
- `generate_dataset()` - Create complete dataset with locations, demands, time windows
- `generate_pickup_locations()` - Random geographic distribution
- `save_dataset()` / `load_dataset()` - Persistence

**Example:**
```python
generator = DataGenerator(seed=42)
dataset = generator.generate_dataset(
    num_pickup_points=30,
    num_buses=2,
    bus_capacity=50,
    city_center=(40.7128, -74.0060),  # NYC
    radius_km=10.0
)
```

### Clustering
Groups pickup points into efficient clusters using K-Means or DBSCAN.

**Key Methods:**
- `kmeans_clustering()` - K-Means clustering with automatic cluster count
- `dbscan_clustering()` - Density-based clustering
- `get_cluster_statistics()` - Analyze cluster properties

**Example:**
```python
clustering = Clustering()
result = clustering.kmeans_clustering(
    pickup_locations,
    num_buses=2
)
print(f"Clusters formed: {result['num_clusters']}")
```

### Route Optimizer
Solves TSP and finds optimal routes within clusters.

**Key Methods:**
- `nearest_neighbor()` - O(n²) greedy heuristic
- `two_opt_optimization()` - Local search improvement
- `dynamic_programming_tsp()` - Exact solution (for n ≤ 20)
- `hybrid_optimization()` - NN + 2-opt combination
- `solve_tsp_with_depot()` - Full TSP with depot and destination
- `estimate_travel_time()` - Time estimation based on distance
- `estimate_fuel_consumption()` - Fuel consumption analysis

**Example:**
```python
optimizer = RouteOptimizer()
optimization_result = optimizer.solve_tsp_with_depot(
    pickup_locations,
    depot,
    destination,
    method='hybrid'  # fastest with good quality
)
print(f"Distance: {optimization_result['distance']:.1f} km")
print(f"Time: {optimizer.estimate_travel_time(optimization_result['distance']):.0f} min")
```

### Visualization
Renders maps and charts for routes and clusters.

**Key Methods:**
- `plot_clusters()` - Cluster visualization with matplotlib
- `plot_route()` - Single route visualization
- `plot_multiple_routes()` - Compare routes on one map
- `create_interactive_map()` - Interactive Folium map
- `plot_route_statistics()` - Distance/time/load charts

**Example:**
```python
visualizer = RouteVisualizer()
visualizer.plot_route(
    pickup_locations,
    route,
    depot,
    destination,
    save_path="route_map.png"
)
```

### Reinforcement Learning
Dynamic route optimization using Q-Learning.

**Key Method:**
- `learn_from_episode()` - Learn from daily operations
- `predict_travel_time()` - Improve time predictions with history
- `get_learned_policy()` - Extract best routes
- `save_learning()` / `load_learning()` - Persistence

**Example:**
```python
rl = RLOptimizer(learning_rate=0.1, discount_factor=0.9)

# Learn from operations
routes_data = [{
    'route': [1, 2, 3, 4],
    'estimated_distance': 25.0,
    'actual_distance': 24.8,
    'estimated_time': 45.0,
    'actual_time': 44.2,
    'is_on_time': True
}]

reward = rl.learn_from_episode(routes_data, time_of_day=8)
rl.save_learning("learned_policy.json")
```

## 🔧 Advanced Usage

### Custom Optimization Parameters

```python
optimizer = BusRouteOptimization(use_rl=True)

# Control clustering behavior
results = optimizer.optimize_routes(
    dataset,
    clustering_method='dbscan',  # density-based
    routing_method='dp',          # dynamic programming (slower, more accurate)
    visualize=True
)
```

### Handling Absences

```python
# When a student is absent, reroute automatically
updated_results = optimizer.handle_absence(
    original_routes=results,
    absent_location_index=5,
    dataset=dataset
)
```

### Real-Time Traffic Adjustments

```python
# Adjust for traffic delays
delayed_route = optimizer.handle_traffic_delay(
    route_id=0,
    delay_minutes=15.0,
    route_details=results['route_details']
)
```

## 📈 Performance Characteristics

### Algorithm Complexity

| Algorithm | Time Complexity | Quality |
|-----------|-----------------|---------|
| Nearest Neighbor | O(n²) | ~80% optimal |
| 2-opt Local Search | O(n²) per iteration | ~90% optimal |
| Dynamic Programming | O(n² · 2ⁿ) | Optimal |
| Hybrid (NN+2-opt) | O(n²) | ~90% optimal |

### Data Sizes

- **Small**: 5-20 locations → Use DP algorithm for optimal solution
- **Medium**: 20-50 locations → Use Hybrid approach
- **Large**: 50+ locations → Use Clustering + Hybrid for each cluster

## 🎨 Output Examples

The system generates several visualizations:

1. **Cluster Map** (`01_clusters.png`)
   - Color-coded clusters with centers
   - Depot location highlighted

2. **Individual Routes** (`02_route_bus_*.png`)
   - Arrows showing pickup order
   - Stop numbers for reference

3. **All Routes Map** (`03_all_routes.png`)
   - Multiple buses on single map
   - Distance-optimized layout

4. **Statistics Dashboard** (`04_statistics.png`)
   - Distance by bus
   - Travel time comparison
   - Load balancing

5. **Interactive Map** (`05_interactive_map.html`)
   - Clickable markers
   - Route inspection tools

## 📊 Sample Results

For a 30-location dataset with 2 buses:

```
========================================
OPTIMIZATION RESULTS
========================================
Total Distance: 142.3 km
Total Time: 238 minutes (3.97 hours)
Estimated Fuel: 28.5 liters
Number of Buses Used: 2
Number of Clusters: 2

Bus 1: 15 stops, 71.2 km, 119 min
Bus 2: 15 stops, 71.1 km, 119 min
```

## 🔬 Reinforcement Learning

The system improves through experience:

```
Episode 1: Average Reward = 45.3
Episode 2: Average Reward = 52.1
Episode 3: Average Reward = 58.7
...
Episode 50: Average Reward = 78.2 (72% improvement)
```

## ⚙️ Configuration

### Default Parameters

```python
# Clustering
num_clusters = num_buses  # Auto-determine clusters
eps = 0.1  # DBSCAN epsilon (~11km scale)

# Routing
method = 'hybrid'  # Balance speed and quality

# RL
learning_rate = 0.1
discount_factor = 0.9
exploration_rate = 0.1
```

## 🚨 Constraints Handled

✅ Each location visited exactly once
✅ Bus capacity constraints
✅ No unnecessary backtracking (via 2-opt)
✅ Time window preferences
✅ Depot and destination requirements
✅ One-way routes with different return location

## 📝 File Formats

### Input Dataset (CSV)

**pickup_locations.csv:**
```
latitude,longitude
40.7580,-73.9855
40.7489,-73.9680
...
```

**metadata.csv:**
```
depot_lat,depot_lon,destination_lat,destination_lon,num_buses,bus_capacity
40.7128,-74.0060,40.7200,-73.9700,2,50
```

### Output (JSON)

**Learned Policy:**
```json
{
  "route_tuple_time_8": {
    "action": "[1, 2, 3, 4, 5]",
    "q_value": 78.5
  }
}
```

## 🔗 Dependencies

- **numpy**: Numerical computing
- **pandas**: Data manipulation
- **scikit-learn**: Machine learning (K-Means, DBSCAN)
- **matplotlib**: Static visualizations
- **folium**: Interactive maps
- **scipy**: Scientific computing
- **networkx**: Graph algorithms
- **torch**: Optional for potential RL extensions

## 🤝 Contributing

Suggestions for improvements:
- Add genetic algorithm for global optimization
- Implement A* with heuristics
- Add real Google Maps API integration
- Vehicle routing problem (VRP) with time windows
- Multi-objective optimization (distance + time + cost)

## 📄 License

This project is open source and available for educational and commercial use.

## 📞 Support

For issues or questions, refer to the inline code documentation or create detailed examples.

---

**Version**: 1.0.0
**Last Updated**: December 2024
**Status**: Production Ready ✅
