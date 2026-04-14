# Quick Start Guide - Bus Route Optimizer

## 🚀 5-Minute Setup

### Step 1: Install Dependencies
```bash
cd bus_route_optimizer
pip install -r requirements.txt
```

### Step 2: Run the Main Demo
```bash
python main.py
```

This will:
- Generate a sample dataset (30 pickup locations, 2 buses)
- Cluster nearby locations  
- Optimize routes using TSP solver
- Train reinforcement learning model
- Generate visualizations
- Handle dynamic rerouting scenarios

### Step 3: Open the Interactive Notebook
```bash
jupyter notebook Complete_Demo.ipynb
```

Run all cells to see the complete system in action with visualizations.

---

## 📊 Quick Examples

### Example 1: Generate Dataset and Optimize
```python
from modules.data_generator import DataGenerator
from main import BusRouteOptimization

# Generate data
generator = DataGenerator(seed=42)
dataset = generator.generate_dataset(num_pickup_points=30, num_buses=2)

# Optimize
optimizer = BusRouteOptimization(use_rl=True)
results = optimizer.optimize_routes(dataset)

print(f"Total Distance: {results['total_distance']:.1f} km")
print(f"Total Time: {results['total_time']:.0f} minutes")
```

### Example 2: Cluster and Inspect
```python
from modules.clustering import Clustering

clustering = Clustering()
result = clustering.kmeans_clustering(dataset['pickup_locations'], num_buses=2)

for cluster_id, indices in result['clusters'].items():
    print(f"Cluster {cluster_id}: {len(indices)} locations")
```

### Example 3: Route Visualization
```python
from modules.visualization import RouteVisualizer

visualizer = RouteVisualizer()
visualizer.plot_clusters(
    dataset['pickup_locations'],
    result['clusters'],
    centers=result['centers'],
    depot=dataset['depot']
)
```

### Example 4: Dynamic Rerouting
```python
# Handle student absence
updated_results = optimizer.handle_absence(
    original_routes=results,
    absent_location_index=5,
    dataset=dataset
)

print(f"New distance: {updated_results['route_details'][0]['distance']:.1f} km")
```

### Example 5: RL Learning
```python
from modules.reinforcement_learning import RLOptimizer

rl = RLOptimizer(learning_rate=0.1)

# Simulate operations
for day in range(10):
    routes_data = [{...}]  # Actual route execution data
    reward = rl.learn_from_episode(routes_data, time_of_day=8)
    print(f"Day {day}: Reward = {reward:.2f}")

# Get learned policy
policy = rl.get_learned_policy()
```

---

## 📁 Project Structure

```
bus_route_optimizer/
├── modules/
│   ├── __init__.py              # Package init
│   ├── data_generator.py        # Dataset generation
│   ├── clustering.py            # K-Means & DBSCAN
│   ├── route_optimizer.py       # TSP solver
│   ├── visualization.py         # Plotting & mapping
│   ├── reinforcement_learning.py # Q-Learning
│   └── utils.py                 # Utilities & validators
├── data/                        # Sample datasets
├── visualizations/              # Output plots
├── Complete_Demo.ipynb          # Full interactive demo
├── main.py                      # Main orchestrator
├── requirements.txt             # Dependencies
├── README.md                    # Full documentation
└── QUICKSTART.md               # This file
```

---

## 🎯 Key Capabilities

✅ **Clustering**: Group 50+ locations into efficient clusters
✅ **Routing**: Optimize TSP within each cluster (<1s for 30 points)
✅ **Constraints**: Handle capacity limits, time windows  
✅ **Learning**: RL-based continuous improvement
✅ **Dynamics**: Real-time rerouting for changes
✅ **Visualization**: Interactive maps and analytics

---

## ⚙️ Configuration

### Clustering Methods
- **K-Means**: Fast, regular-shaped clusters (default)
- **DBSCAN**: Flexible, density-based clustering

### Routing Methods
- **Nearest Neighbor**: O(n²), ~80% quality
- **2-opt**: Local search, ~90% quality  
- **Hybrid**: NN + 2-opt (recommended)
- **DP**: Exact solution for n ≤ 20

### RL Parameters
```python
rl = RLOptimizer(
    learning_rate=0.1,      # How fast to learn
    discount_factor=0.9,    # Future reward importance
    exploration_rate=0.1    # Exploration vs exploitation
)
```

---

## 📈 Performance Expectations

For 30 locations with 2 buses:
- **Computation Time**: <5 seconds
- **2-opt Improvement**: 8-12% over nearest neighbor
- **Route Quality**: 88-92% of optimal
- **RL Learning**: 10-20% improvement over 50 episodes

---

## 🔧 Troubleshooting

### Import Error
```python
# Add to Python path
import sys
sys.path.insert(0, '/path/to/bus_route_optimizer')
```

### Visualization Not Showing
```python
# For Jupyter notebooks, add this:
%matplotlib inline
```

### Memory Issues with Large Datasets
```python
# Use DBSCAN instead of K-Means for large datasets
clustering.dbscan_clustering(locations, eps=0.1)
```

---

## 📚 Next Steps

1. **Explore the Code**: Read module docstrings
2. **Run the Notebook**: Execute `Complete_Demo.ipynb`
3. **Customize**: Modify parameters and test scenarios
4. **Integrate**: Connect to your data sources
5. **Deploy**: Use in production with your fleet

---

## 📞 Support

- Check `README.md` for detailed documentation
- Review code comments for implementation details
- Run `main.py` for working examples
- Examine `Complete_Demo.ipynb` for use cases

---

**Ready to optimize your routes? Run `python main.py` now!** 🚀
