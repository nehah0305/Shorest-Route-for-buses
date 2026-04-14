# Bus Route Optimizer Technical Reference

## 1. Purpose

Bus Route Optimizer is an abstract 2-D bus route planning system. It groups pickup points, computes efficient visit orders, estimates operational cost, and renders the result in a Streamlit interface. The project uses custom map coordinates rather than latitude/longitude and is designed around reproducible datasets and visual route analysis.

The current UI is locked to the Pixel Adventure map theme. In that mode, buses must move only on brown road lanes, with orthogonal movement only: up, down, left, and right. Routes are snapped to the road grid before optimization and rendering.

## 2. Coordinate Model

The project uses abstract map units.

- X is the horizontal axis.
- Y is the vertical axis.
- (0, 0) is the bottom-left corner.
- (map_width, map_height) is the top-right corner.

Distances are not geodesic. They are computed from the abstract coordinate space. In Pixel Adventure mode, route geometry is additionally constrained to the brown road network.

## 3. Project Layout

```text
bus_route_optimizer/
├── app.py
├── main.py
├── modules/
│   ├── clustering.py
│   ├── data_generator.py
│   ├── gamification.py
│   ├── reinforcement_learning.py
│   ├── route_optimizer.py
│   ├── utils.py
│   └── visualization.py
├── Complete_Demo.ipynb
├── README.md
├── QUICKSTART.md
├── CONFIGURATION.md
├── PROJECT_SUMMARY.md
└── INDEX.md
```

## 4. Execution Entry Points

### Streamlit app

`app.py` is the interactive front end. It is responsible for:

- generating a dataset,
- snapping points to the road grid,
- running the optimizer,
- displaying maps, charts, tables, and leaderboard data.

### Orchestrator

`main.py` coordinates clustering, route optimization, reinforcement learning, and optional visualization. It is the best entry point for command-line or scripted runs.

### Notebook demo

`Complete_Demo.ipynb` is the interactive notebook for step-by-step exploration.

## 5. Data Flow

1. `DataGenerator.generate_dataset()` creates pickup points, depot, destination, demands, and time windows.
2. In Pixel Adventure mode, the app snaps those points onto the road grid.
3. `Clustering` splits the pickup points into bus-sized groups.
4. `RouteOptimizer.solve_tsp_with_depot()` computes the visit order for each cluster.
5. `RLOptimizer` optionally learns from simulated route execution data.
6. `RouteVisualizer` and the Streamlit map render the final routes.

## 6. Data Generation

File: `modules/data_generator.py`

### What it does

- Generates pickup coordinates inside map bounds.
- Generates random demand values per pickup.
- Generates pickup time windows.
- Produces a dataset dictionary used by the rest of the system.

### Important dataset keys

- `pickup_locations`: array of pickup coordinates.
- `depot`: start location.
- `destination`: end location.
- `num_buses`: number of buses available.
- `bus_capacity`: capacity limit per bus.
- `demands`: passenger counts for each pickup.
- `time_windows`: earliest/latest service times.
- `map_width` and `map_height`: abstract map size.

### Change impact

- Changing `num_pickup_points` affects clustering size, route length, and runtime.
- Changing `map_width` or `map_height` changes the spatial scale of the entire system.
- Custom locations bypass random generation and are used directly.

## 7. Clustering

File: `modules/clustering.py`

### Responsibilities

- Groups pickup points into clusters that can be assigned to buses.
- Supports K-Means and DBSCAN.
- Handles sparse data fallback behavior.

### Why clustering matters

Clustering reduces the TSP problem size per bus. Instead of solving one large route across all stops, the project solves smaller routes inside each cluster.

### Change impact

- More clusters generally reduce per-route travel distance but may increase total fleet complexity.
- DBSCAN can fail on sparse datasets; the system falls back to K-Means when needed.

## 8. Route Optimization

File: `modules/route_optimizer.py`

### Algorithms

- Nearest Neighbor: greedy baseline route.
- 2-opt: local improvement of a route.
- Dynamic Programming: exact TSP for small problems.
- Hybrid: nearest neighbor followed by 2-opt.

### Road-grid mode

The optimizer now supports a road-grid travel mode:

- pickup points, depot, and destination are snapped to lane intersections,
- distance calculations use the same snapped road grid,
- route waypoints are returned as orthogonal segments only,
- total distance is based on the road path, not diagonal shortcuts.

### Practical effect of changes

- If you move a pickup point, the route may shift to a different lane intersection.
- If you add more lanes in the grid, route options become denser.
- If you reduce lane density, buses have fewer valid path choices and routes may become less flexible.

## 9. Road Snapping and Path Construction

File: `modules/utils.py`

### Road-grid helpers

- `get_pixel_road_axes()` defines the lane coordinates used for snapping.
- `snap_point_to_pixel_road()` moves a point to the nearest valid brown-road intersection.
- `build_pixel_road_path()` converts route points into an orthogonal path.

### Why the helper layer exists

The snapping logic is shared by the UI and the optimizer so the visible route, the animated bus marker, and the measured distance all use the same path geometry.

### Change impact

- Changing the road fractions changes the whole map geometry.
- Changing the grid density changes both route realism and route diversity.
- If the snapping grid and drawing grid disagree, the map will show paths that do not match the optimizer.

## 10. Visualization

File: `modules/visualization.py`

### Static plots

- `plot_clusters()` shows pickup groupings.
- `plot_route()` shows a single route.
- `plot_multiple_routes()` shows all routes together.
- `plot_route_statistics()` shows distance, time, and load charts.

### Route geometry behavior

Static plots use orthogonal polylines so the saved images reflect lane-following behavior instead of direct diagonal segments.

### Change impact

- Visualization changes are presentation-only unless they rely on route geometry helpers.
- If route geometry changes in the optimizer, the visualizer should use the same geometry model.

## 11. Main Orchestrator

File: `main.py`

### Responsibilities

- Runs clustering.
- Calls the route optimizer for each cluster.
- Aggregates route metrics.
- Optionally trains the reinforcement learning component.
- Optionally generates visualizations.

### Data passed through the pipeline

- route index order,
- route distance,
- travel time,
- fuel estimate,
- load per bus,
- route waypoints.

### Change impact

- A change to route geometry in the optimizer automatically affects metrics, charts, exports, and map output.
- Changing clustering changes which points are paired with each bus.

## 12. Streamlit Application

File: `app.py`

### UI sections

- Sidebar configuration.
- Score summary card.
- Map tab.
- Routes tab.
- Statistics tab.
- Leaderboard tab.
- Achievements tab.
- Compare tab.

### Current map behavior

The app is locked to Pixel Adventure only.

- No alternate map styles are exposed.
- The road map is denser than before.
- All generated points are snapped onto the brown lane network.
- Routes are rendered from the optimizer’s road-aware waypoints.

### Change impact while running

- Changing stop count changes the clustering input and therefore the final route order.
- Changing map size changes snapping positions and lane spacing.
- Changing routing method changes route order, distance, and time.
- Changing clustering method changes bus assignment and route grouping.

## 13. Running the Project

### Streamlit

```bash
cd bus_route_optimizer
streamlit run app.py
```

### Command-line demo

```bash
cd bus_route_optimizer
python main.py
```

### Notebook

Open `Complete_Demo.ipynb` in Jupyter and run the cells in order.

## 14. Runtime Changes and Their Effects

### Changing pickup coordinates

- Affects cluster membership.
- Affects route order.
- Affects distance, fuel, and time.

### Changing the number of buses

- Affects how many clusters the system tries to create.
- Affects route length per bus.
- Affects total fleet balance.

### Changing the routing method

- `nn` is fastest and simplest.
- `hybrid` gives better quality for most cases.
- `dp` is exact but only practical for small clusters.

### Changing the map size

- Affects the visual spread of the road grid.
- Affects snapping coordinates.
- Affects the apparent route spacing.

### Changing the road grid density

- More lanes means more possible valid paths.
- More lanes can make the map feel more city-like.
- Excessive density can make the map visually busy.

## 15. Outputs

The project may generate:

- route charts,
- cluster plots,
- interactive HTML maps,
- leaderboard data,
- cached Python bytecode,
- notebook outputs.

These are runtime artifacts and should not be committed.

## 16. Files That Should Stay Out of Git

Ignored by the repository:

- `.venv/`
- `__pycache__/`
- `*.pyc`
- `.ipynb_checkpoints/`
- `bus_route_optimizer/visualizations/`
- `bus_route_optimizer/.streamlit/`
- `bus_route_optimizer/data/leaderboard.json`

## 17. Development Workflow

1. Edit the module that owns the behavior.
2. Run the smallest relevant validation.
3. Check errors before expanding to other files.
4. Regenerate visual outputs only if needed.
5. Keep route geometry helpers and renderers in sync.

## 18. Extending the Project Safely

### If you change route logic

- Update the optimizer.
- Update the visualizer.
- Update the Streamlit map.
- Revalidate route orthogonality.

### If you change the road layout

- Update snapping helpers.
- Update tile generation.
- Update any coordinate assumptions in the UI.

### If you add a new output artifact

- Add it to `.gitignore` if it is generated at runtime.
- Document whether it is reproducible or persistent.

## 19. Verification Checklist

- The app launches with `streamlit run app.py`.
- Routes remain orthogonal in Pixel Adventure mode.
- Pickups, depot, and destination are snapped to the road grid.
- Generated outputs are ignored by Git.
- New changes preserve the shared snapping geometry.

## 20. Summary

This project is an abstract bus route optimizer with clustering, TSP solving, optional reinforcement learning, and custom map visualization. The current production behavior is road-grid locked, with dense brown lanes and orthogonal travel only.