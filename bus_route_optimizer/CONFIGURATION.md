# Configuration and Customization Guide

## Default Configuration

### System Parameters
```python
# Clustering Configuration
CLUSTERING_METHOD = 'kmeans'  # 'kmeans' or 'dbscan'
NUM_CLUSTERS = None           # None for auto-detection
DBSCAN_EPS = 0.1             # Geographic scale (0.1 ≈ 11km)
DBSCAN_MIN_SAMPLES = 3       # Minimum points per cluster

# Routing Configuration  
ROUTING_METHOD = 'hybrid'    # 'nn', 'dp', 'hybrid'
MAX_2OPT_ITERATIONS = 1000   # Max optimization iterations

# Distance and Time Parameters
AVG_SPEED_KMH = 30.0         # Average city driving speed
FUEL_EFFICIENCY_KMPL = 5.0   # Fuel efficiency in km/liter
```

### Reinforcement Learning Parameters
```python
# Q-Learning Configuration
LEARNING_RATE = 0.1          # α: How fast to update Q-values
DISCOUNT_FACTOR = 0.9        # γ: Future reward weight
EXPLORATION_RATE = 0.1       # ε: Exploration probability
EXPLORATION_DECAY = 0.995    # Decay rate per episode
```

## Customization Examples

### Example 1: Use DBSCAN Instead of K-Means
```python
from main import BusRouteOptimization

optimizer = BusRouteOptimization()
results = optimizer.optimize_routes(
    dataset,
    clustering_method='dbscan',  # Changed!
    routing_method='hybrid'
)
```

### Example 2: Faster Optimization (Sacrifice Quality)
```python
# Use only nearest neighbor (no 2-opt)
results = optimizer.optimize_routes(
    dataset,
    routing_method='nn'  # Faster but ~80% quality
)
```

### Example 3: Optimal Solution (Slower)
```python
# Use dynamic programming for small instances
results = optimizer.optimize_routes(
    dataset,
    routing_method='dp'  # Optimal but O(n² * 2^n)
)
```

### Example 4: Aggressive RL Learning
```python
from modules.reinforcement_learning import RLOptimizer

# More aggressive learning
rl = RLOptimizer(
    learning_rate=0.3,      # Higher learning rate
    discount_factor=0.95,   # Higher future weight
    exploration_rate=0.2    # More exploration
)
```

## Dataset Customization

### Create Custom Dataset
```python
import numpy as np

# Define your locations
pickup_locations = np.array([
    [40.7128, -74.0060],  # Location 1
    [40.7580, -73.9855],  # Location 2
    # ... more locations
])

# Define demands (passengers per location)
demands = np.array([10, 15, 20, ...])

# Create dataset dict
dataset = {
    'pickup_locations': pickup_locations,
    'depot': np.array([40.7128, -74.0060]),
    'destination': np.array([40.7200, -73.9700]),
    'num_buses': 2,
    'bus_capacity': 50,
    'demands': demands,
    'num_pickup_points': len(pickup_locations)
}
```

### Load from CSV
```python
import pandas as pd
import numpy as np

# Read locations
df = pd.read_csv('pickup_locations.csv')
pickup_locations = df[['latitude', 'longitude']].values

# Read demand data
demands = df['passengers'].values
```

## Performance Tuning

### For Large Datasets (50+ locations)
```python
# Use clustering to break into smaller problems
optimization_method = 'hybrid'  # Faster than DP
num_clusters = min(10, dataset['num_buses'] * 3)
```

### For Time-Critical Applications
```python
# Minimize computation time
optimization_method = 'nn'  # Fastest
use_rl = False              # Skip learning
visualize = False           # Skip plotting
```

### For Highest Quality Solutions
```python
# Maximize solution quality
optimization_method = 'hybrid'  # Best balance
use_rl = True                   # Learn patterns
max_iterations = 5000           # More 2-opt iterations
```

## Integration Guide

### Connect to Your Data Source
```python
class YourDataSource:
    def fetch_pickup_locations(self):
        """Fetch from your database/API."""
        # Your implementation here
        pass
    
    def fetch_demands(self):
        """Get passenger demands."""
        pass

# Use in optimizer
data_source = YourDataSource()
locations = data_source.fetch_pickup_locations()
demands = data_source.fetch_demands()
```

### Export Results
```python
from modules.utils import ReportGenerator

# Generate reports
ReportGenerator.generate_summary_report(results, 'report.txt')
ReportGenerator.generate_json_report(results, 'report.json')

# Save visualizations
visualizer.plot_clusters(..., save_path='clusters.png')
visualizer.create_interactive_map(..., save_path='map.html')
```

### Real-Time Updates
```python
# Update with actual travel times
actual_times = fetch_actual_times_from_gps()

rl_optimizer.learn_from_episode({
    'route': route,
    'estimated_time': estimated,
    'actual_time': actual_times[route],
    'on_time': actual_times[route] <= estimated * 1.1
})
```

## Monitoring and Logging

### Enable Detailed Logging
```python
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Now all print statements are logged to file
```

### Track Performance Metrics
```python
metrics_history = []

for day in range(30):
    results = optimizer.optimize_routes(dataset)
    metrics_history.append({
        'day': day,
        'distance': results['total_distance'],
        'time': results['total_time'],
        'timestamp': datetime.now()
    })

# Analyze trends
import pandas as pd
df = pd.DataFrame(metrics_history)
print(df.describe())
```

## Validation & Testing

### Validate Configuration
```python
from modules.utils import DataValidator

# Check input data
DataValidator.validate_locations(pickup_locations)
DataValidator.validate_demands(demands, bus_capacity)
DataValidator.validate_routes(routes, num_locations)
```

### Compare Algorithms
```python
from modules.utils import PerformanceAnalyzer

# Compare two methods
nn_result = optimizer.solve_tsp_with_depot(..., method='nn')
hybrid_result = optimizer.solve_tsp_with_depot(..., method='hybrid')

comparison = PerformanceAnalyzer.compare_algorithms(nn_result, hybrid_result)
print(f"Improvement: {comparison['improvement']:.1f}%")
```

### Scalability Testing
```python
import time

sizes = [10, 20, 30, 40, 50]
times = []

for size in sizes:
    data = generator.generate_dataset(num_pickup_points=size)
    start = time.time()
    results = optimizer.optimize_routes(data, visualize=False)
    times.append(time.time() - start)

analysis = PerformanceAnalyzer.scalability_analysis(sizes, times)
print(f"Complexity: {analysis['complexity_class']}")
```

---

**For more examples, see `Complete_Demo.ipynb` and `QUICKSTART.md`**
