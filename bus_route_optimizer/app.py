"""
🚌 Bus Route Optimizer — Gamified Streamlit UI  (custom map edition)
Run with:  streamlit run app.py   (from inside bus_route_optimizer/)
"""

import os, sys, time
from datetime import date
from typing import List, Tuple
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# ── make sure local modules are importable regardless of cwd ──────────────────
sys.path.insert(0, os.path.dirname(__file__))

from modules.gamification import GamificationEngine
from modules.utils import ReportGenerator
from main import BusRouteOptimization
from modules.data_generator import DataGenerator

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🚌 Bus Route Optimizer",
    page_icon="🚌",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .score-card   { background:#1e293b; border-radius:12px; padding:18px; text-align:center; }
  .score-number { font-size:3rem; font-weight:800; color:#38bdf8; }
  .star-row     { font-size:1.8rem; }
  .achievement  { background:#064e3b; color:#6ee7b7; border-radius:8px;
                  padding:8px 12px; margin:4px; display:inline-block; }
  h1 { color:#f1f5f9 !important; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
ALGO_INFO = {
    "hybrid": ("Hybrid (NN + 2-opt)", "Greedy nearest-neighbour seed refined with 2-opt edge swaps. Best quality/speed balance. Recommended for most runs."),
    "nn":     ("Nearest Neighbour",   "Pure greedy heuristic — fastest but lowest quality. Great for quick previews."),
    "dp":     ("Dynamic Programming", "Held-Karp exact algorithm (optimal). Exponential cost; only runs when ≤15 stops per cluster."),
}

MAP_STYLES = ["Grid City", "District Zones", "Open Field", "Pixel Adventure"]

BUS_PALETTE = ["#ef4444", "#3b82f6", "#22c55e", "#a855f7", "#f97316", "#06b6d4"]

# ── State ─────────────────────────────────────────────────────────────────────
if "gamification" not in st.session_state:
    st.session_state.gamification = GamificationEngine()
if "history" not in st.session_state:
    st.session_state.history = []
if "total_xp" not in st.session_state:
    st.session_state.total_xp = 0
if "earned_ids" not in st.session_state:
    st.session_state.earned_ids = set()
if "cmp_results" not in st.session_state:
    st.session_state.cmp_results = {}

engine: GamificationEngine = st.session_state.gamification


# ── Custom map renderer ───────────────────────────────────────────────────────
def _pixel_colorscale() -> list:
    """Discrete colors for pixel-art tile classes."""
    stops = [
        (0 / 4, "#9ed85f"),  # grass
        (1 / 4, "#42b8cf"),  # water
        (2 / 4, "#d3b34e"),  # path
        (3 / 4, "#2f8c54"),  # forest
        (4 / 4, "#d7ec9f"),  # meadow highlight
    ]
    scale = []
    for value, color in stops:
        scale.append([value, color])
        scale.append([value, color])
    return scale


def _build_pixel_tilemap(
    map_width: float,
    map_height: float,
    seed: int,
    cols: int = 72,
    rows: int = 48,
) -> np.ndarray:
    """Create a deterministic pixel-art map inspired by retro RPG tiles."""
    rng = np.random.default_rng(seed)
    tiles = np.zeros((rows, cols), dtype=np.int8)  # 0 -> grass

    # Meadow highlights
    meadow_noise = rng.random((rows, cols))
    tiles[meadow_noise < 0.05] = 4

    # Forest blocks
    for _ in range(16):
        x0 = int(rng.integers(0, max(cols - 8, 1)))
        y0 = int(rng.integers(0, max(rows - 8, 1)))
        w = int(rng.integers(4, 10))
        h = int(rng.integers(3, 8))
        tiles[y0:min(rows, y0 + h), x0:min(cols, x0 + w)] = 3

    # Main roads (horizontal + vertical)
    road_rows = [int(rows * 0.25), int(rows * 0.55), int(rows * 0.8)]
    road_cols = [int(cols * 0.2), int(cols * 0.55), int(cols * 0.82)]
    for rr in road_rows:
        tiles[max(0, rr - 1):min(rows, rr + 1), :] = 2
    for cc in road_cols:
        tiles[:, max(0, cc - 1):min(cols, cc + 1)] = 2

    # River strip with slight wobble
    for y in range(rows):
        center = int(cols * 0.35 + 3 * np.sin(y / 4.0))
        half = 2 + (y % 3 == 0)
        x0 = max(0, center - half)
        x1 = min(cols, center + half)
        tiles[y, x0:x1] = 1

    # Keep start and end corners path-friendly
    safe_margin_x = max(2, int(cols * 0.08))
    safe_margin_y = max(2, int(rows * 0.08))
    tiles[0:safe_margin_y, 0:safe_margin_x] = 2
    tiles[rows - safe_margin_y:rows, cols - safe_margin_x:cols] = 2

    return tiles


def _polyline_point(points: List[np.ndarray], progress: float) -> Tuple[float, float]:
    """Get interpolated x/y position at progress in [0, 1] along a polyline."""
    if not points:
        return 0.0, 0.0
    if len(points) == 1:
        return float(points[0][0]), float(points[0][1])

    progress = float(np.clip(progress, 0.0, 1.0))
    segment_lengths = []
    for i in range(len(points) - 1):
        p0, p1 = points[i], points[i + 1]
        segment_lengths.append(float(np.linalg.norm(p1 - p0)))
    total = sum(segment_lengths)
    if total <= 1e-9:
        last = points[-1]
        return float(last[0]), float(last[1])

    target = total * progress
    walked = 0.0
    for i, seg_len in enumerate(segment_lengths):
        if walked + seg_len >= target:
            ratio = (target - walked) / max(seg_len, 1e-9)
            p0, p1 = points[i], points[i + 1]
            xy = p0 + ratio * (p1 - p0)
            return float(xy[0]), float(xy[1])
        walked += seg_len

    last = points[-1]
    return float(last[0]), float(last[1])


def render_custom_map(
    dataset: dict,
    results: dict,
    style: str = "Grid City",
    height: int = 560,
    animate_buses: bool = False,
    animation_steps: int = 70,
    animation_seconds: int = 14,
) -> go.Figure:
    """
    Render a styled 2-D custom map of all bus routes using Plotly.

    coordinate convention: col-0 = X (horizontal), col-1 = Y (vertical).
    """
    pickup_locs = dataset["pickup_locations"]
    depot       = dataset["depot"]
    dest        = dataset["destination"]
    mw          = float(dataset.get("map_width",  100))
    mh          = float(dataset.get("map_height", 100))

    fig = go.Figure()
    is_pixel = style == "Pixel Adventure"

    # ── Background & style ───────────────────────────────────────────────────
    if is_pixel:
        tile_seed = int(dataset.get("num_pickup_points", 0) + dataset.get("num_buses", 0) * 97)
        tiles = _build_pixel_tilemap(mw, mh, tile_seed)
        rows, cols = tiles.shape
        fig.add_trace(go.Heatmap(
            z=tiles,
            x0=0,
            dx=mw / cols,
            y0=0,
            dy=mh / rows,
            colorscale=_pixel_colorscale(),
            zmin=0,
            zmax=4,
            showscale=False,
            opacity=0.96,
            zsmooth=False,
            hoverinfo="skip",
            name="Pixel Terrain",
        ))
    else:
        fig.add_shape(type="rect", x0=0, y0=0, x1=mw, y1=mh,
                      fillcolor="#0f172a", line=dict(color="#334155", width=1))

    if style == "Grid City":
        step = max(mw, mh) / 10
        for v in np.arange(0, mw + step, step):
            fig.add_shape(type="line", x0=v, y0=0, x1=v, y1=mh,
                          line=dict(color="#1e293b", width=1))
        for h in np.arange(0, mh + step, step):
            fig.add_shape(type="line", x0=0, y0=h, x1=mw, y1=h,
                          line=dict(color="#1e293b", width=1))

    elif style == "District Zones":
        zones = [
            (0,     0,     mw/2, mh/2, "#0c1e2d", "Residential"),
            (mw/2,  0,     mw,   mh/2, "#0c2d1a", "Commercial"),
            (0,     mh/2,  mw/2, mh,   "#2d0c1a", "Industrial"),
            (mw/2,  mh/2,  mw,   mh,   "#1a0c2d", "University"),
        ]
        zone_colors = ["#164e63", "#14532d", "#7f1d1d", "#3b0764"]
        for i, (x0, y0, x1, y1, fc, label) in enumerate(zones):
            fig.add_shape(type="rect", x0=x0, y0=y0, x1=x1, y1=y1,
                          fillcolor=fc, line=dict(color=zone_colors[i], width=1.5))
            fig.add_annotation(x=(x0+x1)/2, y=(y0+y1)/2, text=label,
                               font=dict(color="#475569", size=10), showarrow=False)

    # ── Route lines ──────────────────────────────────────────────────────────
    for bus_i, detail in enumerate(results["route_details"]):
        color = BUS_PALETTE[bus_i % len(BUS_PALETTE)]
        wpts  = detail["waypoints"]
        xs    = [float(p[0]) for p in wpts]
        ys    = [float(p[1]) for p in wpts]
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines",
            line=dict(color=color, width=3 if is_pixel else 2.5),
            name=f"🚌 Bus {bus_i+1}",
            hoverinfo="skip",
        ))

    # ── Stop markers (per bus, numbered) ────────────────────────────────────
    for bus_i, detail in enumerate(results["route_details"]):
        color = BUS_PALETTE[bus_i % len(BUS_PALETTE)]
        for stop_i, idx in enumerate(detail["route"]):
            pt = pickup_locs[idx]
            fig.add_trace(go.Scatter(
                x=[float(pt[0])], y=[float(pt[1])],
                mode="markers+text",
                marker=dict(size=13, color=color,
                            line=dict(color="white", width=1)),
                text=[str(stop_i + 1)],
                textposition="middle center",
                textfont=dict(size=7, color="white"),
                showlegend=False,
                hovertext=(f"Bus {bus_i+1} · Stop {stop_i+1}<br>"
                           f"x={pt[0]:.1f}, y={pt[1]:.1f}"),
                hoverinfo="text",
            ))

    # ── Depot & Destination ──────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=[float(depot[0])], y=[float(depot[1])],
        mode="markers+text",
        marker=dict(size=20, color="#22c55e", symbol="square",
                    line=dict(color="white", width=2)),
        text=["🏠"], textposition="top center",
        name="Depot", hovertext="Depot", hoverinfo="text",
    ))
    fig.add_trace(go.Scatter(
        x=[float(dest[0])], y=[float(dest[1])],
        mode="markers+text",
        marker=dict(size=20, color="#a855f7", symbol="diamond",
                    line=dict(color="white", width=2)),
        text=["🏫"], textposition="top center",
        name="Destination", hovertext="Destination", hoverinfo="text",
    ))

    # ── Animated bus movement ────────────────────────────────────────────────
    bus_trace_indices = []
    if animate_buses and results["route_details"]:
        for bus_i, detail in enumerate(results["route_details"]):
            color = BUS_PALETTE[bus_i % len(BUS_PALETTE)]
            points = [np.array(p, dtype=float) for p in detail["waypoints"]]
            x0, y0 = _polyline_point(points, 0.0)
            fig.add_trace(go.Scatter(
                x=[x0],
                y=[y0],
                mode="markers",
                marker=dict(size=15, color=color, symbol="square", line=dict(color="white", width=1.5)),
                name=f"Bus {bus_i+1} (Live)",
                hovertext=f"Bus {bus_i+1}",
                hoverinfo="text",
            ))
            bus_trace_indices.append(len(fig.data) - 1)

        frames = []
        total_steps = max(20, int(animation_steps))
        for step in range(total_steps + 1):
            progress = step / total_steps
            frame_data = []
            for detail in results["route_details"]:
                points = [np.array(p, dtype=float) for p in detail["waypoints"]]
                x, y = _polyline_point(points, progress)
                frame_data.append(go.Scatter(x=[x], y=[y]))
            frames.append(go.Frame(data=frame_data, traces=bus_trace_indices, name=f"f{step}"))
        fig.frames = frames

    # ── Layout ───────────────────────────────────────────────────────────────
    fig.update_layout(
        plot_bgcolor="#0f172a" if not is_pixel else "#0e2619",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e2e8f0" if is_pixel else "#94a3b8"),
        xaxis=dict(range=[-2, mw + 2], showgrid=False, zeroline=False,
                   title="X", color="#334155", constrain="domain"),
        yaxis=dict(range=[-2, mh + 2], showgrid=False, zeroline=False,
                   title="Y", color="#334155", scaleanchor="x", scaleratio=1),
        legend=dict(bgcolor="rgba(30,41,59,0.85)", bordercolor="#334155",
                    borderwidth=1, font=dict(size=11)),
        margin=dict(t=10, b=30, l=40, r=10),
        height=height,
        hovermode="closest",
    )

    if animate_buses and bus_trace_indices:
        frame_ms = int(max(40, (animation_seconds * 1000) / max(1, animation_steps)))
        fig.update_layout(
            updatemenus=[{
                "type": "buttons",
                "showactive": False,
                "direction": "left",
                "x": 0.01,
                "y": 1.08,
                "buttons": [
                    {
                        "label": "▶ Play",
                        "method": "animate",
                        "args": [None, {"frame": {"duration": frame_ms, "redraw": True},
                                        "transition": {"duration": 0}, "fromcurrent": True}],
                    },
                    {
                        "label": "⏸ Pause",
                        "method": "animate",
                        "args": [[None], {"frame": {"duration": 0, "redraw": False},
                                           "transition": {"duration": 0}, "mode": "immediate"}],
                    },
                ],
            }],
            sliders=[{
                "active": 0,
                "x": 0.14,
                "y": 1.08,
                "len": 0.84,
                "currentvalue": {"prefix": "Route progress: "},
                "pad": {"t": 0, "b": 0},
                "steps": [
                    {
                        "label": f"{int((i / max(1, animation_steps)) * 100)}%",
                        "method": "animate",
                        "args": [[f"f{i}"], {"frame": {"duration": 0, "redraw": True},
                                             "transition": {"duration": 0}, "mode": "immediate"}],
                    }
                    for i in range(animation_steps + 1)
                ],
            }],
        )
    return fig

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🎮 Game Settings")
    st.divider()

    player_name = st.text_input("👤 Player Name", value="Optimizer")

    st.divider()
    st.subheader("🗺️ Map Settings")
    map_style  = st.selectbox("Map Style", MAP_STYLES)
    map_width  = st.slider("Map Width  (units)", 50, 200, 100, 10)
    map_height = st.slider("Map Height (units)", 50, 200, 100, 10)

    st.subheader("📍 Stops")
    stop_mode = st.radio("Stop Placement", ["🎲 Random", "✏️ Custom"], horizontal=True)

    num_stops = st.slider("Pickup Stops", min_value=3, max_value=60, value=20, step=1,
                          disabled=(stop_mode == "✏️ Custom"))
    num_buses = st.slider("Number of Buses", min_value=1, max_value=6, value=2)
    seed      = st.number_input("Random Seed", min_value=0, max_value=9999, value=42, step=1,
                                disabled=(stop_mode == "✏️ Custom"))

    st.divider()
    st.subheader("⚙️ Algorithm")
    clustering_method = st.selectbox("Clustering",    ["kmeans", "dbscan"])
    routing_method    = st.selectbox("Routing (TSP)", ["hybrid", "nn", "dp"])
    use_rl            = st.checkbox("Enable RL Optimizer", value=True)

    algo_name, algo_desc = ALGO_INFO[routing_method]
    with st.expander(f"ℹ️ About: {algo_name}"):
        st.caption(algo_desc)

    st.divider()
    st.subheader("🎯 Challenge Mode")
    challenge_on = st.toggle("Daily Challenge", value=False,
                             help="Lock everything to today's seed (30 stops, 3 buses, 100×100 map). "
                                  "Compete with others on identical conditions!")
    if challenge_on:
        _today_seed = int(date.today().strftime("%Y%m%d")) % 10000
        num_stops, num_buses, seed = 30, 3, _today_seed
        map_width, map_height = 100, 100
        st.info(f"📅 Today's seed: **{_today_seed}** — 100×100 map, 30 stops, 3 buses")

    st.divider()
    run_btn = st.button("🚀 Run Optimisation", use_container_width=True, type="primary")

    st.divider()
    st.subheader("📊 Session Stats")
    col_a, col_b = st.columns(2)
    col_a.metric("Total Runs", len(st.session_state.history))
    col_b.metric("Total XP ✨", st.session_state.total_xp)
    if st.session_state.history:
        best = max(st.session_state.history, key=lambda x: x["score"])
        st.metric("🏆 Best Score", f"{best['score']:.1f}  {best['star_display']}")

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🚌 Bus Route Optimizer")
st.caption("Shortest-route planning with K-Means clustering, TSP solving, and Q-Learning — now gamified!")

# ── Run optimisation ──────────────────────────────────────────────────────────
# ── Custom stop editor (shown only in Custom mode) ────────────────────────────
custom_locs = None
if stop_mode == "✏️ Custom":
    st.subheader("✏️ Place Your Stops")
    st.caption(f"Enter x/y coordinates. Valid range: 0 – {map_width} (x),  0 – {map_height} (y).")
    default_df = pd.DataFrame({
        "x": np.round(np.random.uniform(10, map_width  - 10, 8), 1).tolist(),
        "y": np.round(np.random.uniform(10, map_height - 10, 8), 1).tolist(),
    })
    edited = st.data_editor(
        default_df, num_rows="dynamic", use_container_width=True,
        column_config={
            "x": st.column_config.NumberColumn("X", min_value=0, max_value=map_width,  step=0.5),
            "y": st.column_config.NumberColumn("Y", min_value=0, max_value=map_height, step=0.5),
        },
    )
    if len(edited) < 2:
        st.warning("Add at least 2 stops to run optimisation.")
    else:
        custom_locs = edited[["x", "y"]].dropna().values
        num_stops   = len(custom_locs)

# ── Run optimisation ──────────────────────────────────────────────────────────
if run_btn:
    with st.spinner("⚙️ Building map and optimising routes…"):
        t0 = time.time()
        generator = DataGenerator(seed=int(seed))
        dataset   = generator.generate_dataset(
            num_pickup_points=num_stops,
            num_buses=num_buses,
            map_width=float(map_width),
            map_height=float(map_height),
            custom_locations=custom_locs,
        )
        optimizer = BusRouteOptimization(use_rl=use_rl)
        results   = optimizer.optimize_routes(
            dataset,
            clustering_method=clustering_method,
            routing_method=routing_method,
            visualize=False,
        )
        elapsed = time.time() - t0

    total_fuel    = optimizer.route_optimizer.estimate_fuel_consumption(results["total_distance"])
    session_count = engine.get_session_count() + 1

    score_data   = engine.compute_score(
        results["total_distance"], results["total_time"],
        total_fuel, results["num_buses"], num_stops, elapsed,
    )
    achievements = engine.evaluate_achievements(
        score_data["score"], elapsed, total_fuel,
        results["num_buses"], num_stops, routing_method, session_count,
    )
    engine.add_to_leaderboard(player_name, score_data, {
        "stops": num_stops, "buses": num_buses,
        "clustering": clustering_method, "routing": routing_method,
        "map_style": map_style,
        "challenge": challenge_on,
    }, achievements)

    run_xp = sum(a["xp"] for a in achievements)
    st.session_state.total_xp += run_xp
    for a in achievements:
        st.session_state.earned_ids.add(a["id"])
    st.session_state.history.append({
        **score_data,
        "results":      results,
        "dataset":      dataset,
        "achievements": achievements,
        "elapsed":      elapsed,
        "fuel":         total_fuel,
        "map_style":    map_style,
        "map_width":    map_width,
        "map_height":   map_height,
        "clustering":   clustering_method,
        "routing":      routing_method,
        "xp":           run_xp,
        "num_stops":    num_stops,
        "challenge":    challenge_on,
    })
    st.success(f"✅ Done in {elapsed:.2f}s  |  +{run_xp} XP earned")

# ── Display ───────────────────────────────────────────────────────────────────
if not st.session_state.history:
    st.info("👈 Configure your settings in the sidebar and hit **Run Optimisation** to start!")
    st.stop()

latest  = st.session_state.history[-1]
results = latest["results"]
dataset = latest["dataset"]

# ── Score Card ────────────────────────────────────────────────────────────────
col_score, col_metrics = st.columns([1, 2])

with col_score:
    st.markdown(f"""
    <div class="score-card">
      <div style="font-size:1rem;color:#94a3b8">EFFICIENCY SCORE</div>
      <div class="score-number">{latest['score']:.1f}</div>
      <div class="star-row">{latest['star_display']}</div>
      <div style="margin-top:8px;color:#64748b">⏱ {latest['elapsed']:.2f}s</div>
    </div>
    """, unsafe_allow_html=True)

with col_metrics:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("🛣️ Total Distance", f"{results['total_distance']:.1f} km")
    m2.metric("⏰ Total Time",     f"{results['total_time']:.0f} min")
    m3.metric("⛽ Fuel Used",      f"{latest['fuel']:.1f} L")
    m4.metric("🚌 Buses Used",     results["num_buses"])

    # Score breakdown bar — each component is on 0-100 scale
    breakdown = {
        "Distance\n(40%)": latest["dist_score"],
        "Time\n(30%)":     latest["time_score"],
        "Fuel\n(20%)":     latest["fuel_score"],
        "Speed\n(10%)":    min(100.0, latest["speed_bonus"] * 10),
    }
    fig_bar = go.Figure(go.Bar(
        x=list(breakdown.keys()),
        y=list(breakdown.values()),
        marker_color=["#38bdf8","#818cf8","#34d399","#fbbf24"],
        text=[f"{v:.1f}" for v in breakdown.values()],
        textposition="outside",
    ))
    fig_bar.update_layout(
        margin=dict(t=10, b=0, l=0, r=0), height=160,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(range=[0, 115], showgrid=False, visible=False),
        xaxis=dict(color="#94a3b8", tickfont=dict(size=10)),
        font=dict(color="#94a3b8"),
        showlegend=False,
    )
    st.plotly_chart(fig_bar, use_container_width=True)

st.divider()

# ── Achievements ──────────────────────────────────────────────────────────────
if latest["achievements"]:
    st.subheader("🏅 Achievements Unlocked!")
    badges_html = " ".join(
        f'<span class="achievement">{a["name"]} <small>+{a["xp"]}xp</small></span>'
        for a in latest["achievements"]
    )
    st.markdown(badges_html, unsafe_allow_html=True)
    st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_map, tab_routes, tab_stats, tab_board, tab_ach, tab_cmp = st.tabs([
    "🗺️ Map", "📍 Routes", "📊 Charts", "🏆 Leaderboard", "🏅 Achievements", "⚖️ Compare",
])

# ── MAP TAB ───────────────────────────────────────────────────────────────────
with tab_map:
    style_col, anim_col1, anim_col2 = st.columns([2, 2, 3])
    display_style = style_col.selectbox(
        "Map view style",
        MAP_STYLES,
        index=MAP_STYLES.index(latest.get("map_style", "Grid City")) if latest.get("map_style", "Grid City") in MAP_STYLES else 0,
        key="display_map_style",
        help="Change map appearance here without re-running optimization.",
    )
    animate_buses = anim_col1.toggle("Animate bus movement", value=True)
    animation_steps = anim_col2.slider(
        "Animation smoothness",
        min_value=20,
        max_value=120,
        value=70,
        step=10,
        disabled=not animate_buses,
    )
    fig_map = render_custom_map(
        dataset, results,
        style=display_style,
        animate_buses=animate_buses,
        animation_steps=animation_steps,
    )
    st.plotly_chart(fig_map, use_container_width=True)
    if not results.get("route_details"):
        st.warning("No routes were generated for this run. Try KMeans clustering or add more stops.")

# ── ROUTES TAB ────────────────────────────────────────────────────────────────
with tab_routes:
    pickup_locs = dataset["pickup_locations"]
    demands     = dataset["demands"]
    tw          = dataset["time_windows"]
    depot       = dataset["depot"]
    dest        = dataset["destination"]

    # ── Export buttons ────────────────────────────────────────────────────────
    exp_col1, exp_col2, exp_col3 = st.columns([2, 2, 4])

    # 1. All-routes CSV
    all_rows = []
    for bus_i, detail in enumerate(results["route_details"]):
        for order, idx in enumerate(detail["route"]):
            pt = pickup_locs[idx]
            all_rows.append({
                "Bus":        bus_i + 1,
                "Stop":       order + 1,
                "Pickup_ID":  idx,
                "Latitude":   round(float(pt[0]), 6),
                "Longitude":  round(float(pt[1]), 6),
                "Demand":     int(demands[idx]),
                "Window_Start_min": int(tw[idx, 0]),
                "Window_End_min":   int(tw[idx, 1]),
            })
    csv_bytes = pd.DataFrame(all_rows).to_csv(index=False).encode()
    exp_col1.download_button(
        "📥 Download Routes CSV", data=csv_bytes,
        file_name="optimised_routes.csv", mime="text/csv",
    )

    # 2. JSON report  (use existing ReportGenerator)
    json_str = ReportGenerator.generate_json_report(results)
    exp_col2.download_button(
        "📥 Download JSON Report", data=json_str.encode(),
        file_name="route_report.json", mime="application/json",
    )

    st.divider()
    BUS_COLORS = ["#ef4444","#3b82f6","#22c55e","#a855f7","#f97316","#06b6d4"]

    for i, detail in enumerate(results["route_details"]):
        color = BUS_COLORS[i % len(BUS_COLORS)]
        header = (f"🚌 Bus {i+1}  —  {detail['num_stops']} stops  •  "
                  f"{detail['distance']:.1f} km  •  {detail['time_minutes']:.0f} min  •  "
                  f"{detail['fuel_liters']:.1f} L  •  {detail['total_load']} pax")
        with st.expander(header, expanded=(i == 0)):
            # Stop table
            rows = []
            for order, idx in enumerate(detail["route"]):
                pt = pickup_locs[idx]
                rows.append({
                    "Stop #":    order + 1,
                    "Pickup ID": idx,
                    "Latitude":  round(float(pt[0]), 5),
                    "Longitude": round(float(pt[1]), 5),
                    "Demand":    int(demands[idx]),
                    "Window Start (min)": int(tw[idx, 0]),
                    "Window End (min)":   int(tw[idx, 1]),
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            # Mini custom-map route plot (no real-world geo projection)
            route_pts = [depot.tolist()] + [pickup_locs[idx].tolist() for idx in detail["route"]] + [dest.tolist()]
            rx = [p[0] for p in route_pts]
            ry = [p[1] for p in route_pts]
            fig_r = go.Figure()
            fig_r.add_trace(go.Scatter(
                x=rx, y=ry,
                mode="lines+markers",
                line=dict(color=color, width=2),
                marker=dict(size=8, color=color),
                name=f"Bus {i+1}",
            ))
            # Depot & destination
            fig_r.add_trace(go.Scatter(
                x=[float(depot[0]), float(dest[0])],
                y=[float(depot[1]), float(dest[1])],
                mode="markers+text",
                marker=dict(size=14, color=["#22c55e", "#a855f7"], symbol="diamond"),
                text=["Depot","Destination"], textposition="top center",
                name="Key Points",
            ))
            fig_r.update_layout(
                xaxis=dict(
                    title="X",
                    range=[-2, float(dataset.get("map_width", 100)) + 2],
                    showgrid=False,
                    zeroline=False,
                    color="#94a3b8",
                ),
                yaxis=dict(
                    title="Y",
                    range=[-2, float(dataset.get("map_height", 100)) + 2],
                    showgrid=False,
                    zeroline=False,
                    scaleanchor="x",
                    scaleratio=1,
                    color="#94a3b8",
                ),
                margin=dict(t=0, b=0, l=0, r=0), height=280,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="#0f172a",
                showlegend=False,
            )
            st.plotly_chart(fig_r, use_container_width=True)

# ── STATS TAB ─────────────────────────────────────────────────────────────────
with tab_stats:
    bus_labels = [f"Bus {i+1}" for i in range(len(results["route_details"]))]
    dists = [d["distance"]     for d in results["route_details"]]
    times = [d["time_minutes"] for d in results["route_details"]]
    loads = [d["total_load"]   for d in results["route_details"]]

    def _bar(title, labels, vals, color):
        fig = px.bar(x=labels, y=vals, title=title, color_discrete_sequence=[color])
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          margin=dict(t=40, b=0, l=0, r=0), showlegend=False,
                          font=dict(color="#94a3b8"), yaxis=dict(gridcolor="#1e293b"))
        return fig

    c1, c2, c3 = st.columns(3)
    c1.plotly_chart(_bar("Distance (km)", bus_labels, dists, "#38bdf8"), use_container_width=True)
    c2.plotly_chart(_bar("Time (min)",    bus_labels, times, "#818cf8"), use_container_width=True)
    c3.plotly_chart(_bar("Passengers",    bus_labels, loads, "#34d399"), use_container_width=True)

    st.divider()

    # Score history line chart
    if len(st.session_state.history) > 1:
        hist_scores = [h["score"] for h in st.session_state.history]
        fig_hist = px.line(
            y=hist_scores, markers=True, title="📈 Score History Across Runs",
            labels={"y": "Score", "index": "Run #"},
            color_discrete_sequence=["#38bdf8"],
        )
        fig_hist.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#94a3b8"), yaxis=dict(range=[0, 105], gridcolor="#1e293b"),
            xaxis=dict(gridcolor="#1e293b"),
        )
        st.plotly_chart(fig_hist, use_container_width=True)
        st.divider()

    # All-runs comparison table
    if st.session_state.history:
        st.subheader("📋 All Runs Comparison")
        rows = []
        for n, h in enumerate(st.session_state.history, 1):
            rows.append({
                "Run":        n,
                "Map Style":  h.get("map_style", "—"),
                "Stops":      h.get("num_stops", "—"),
                "Clustering": h.get("clustering", "—"),
                "Routing":    h.get("routing", "—"),
                "Score":      f"{h['score']:.1f}",
                "Stars":      "⭐" * h["stars"],
                "Distance km":f"{h['results']['total_distance']:.1f}",
                "Time min":   f"{h['results']['total_time']:.0f}",
                "Fuel L":     f"{h['fuel']:.1f}",
                "XP":         h.get("xp", 0),
                "Elapsed s":  f"{h['elapsed']:.2f}",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ── LEADERBOARD TAB ───────────────────────────────────────────────────────────
with tab_board:
    board = engine.get_leaderboard(10)

    lb_col, reset_col = st.columns([5, 1])
    lb_col.subheader("🏆 All-Time Top 10")
    if reset_col.button("🗑️ Reset", help="Clear the persistent leaderboard"):
        engine.leaderboard = []
        engine._save_leaderboard()
        st.rerun()

    if not board:
        st.info("No scores yet – run an optimisation to get on the board!")
    else:
        # Tabular leaderboard
        lb_rows = []
        for rank, entry in enumerate(board, 1):
            medal = ["🥇", "🥈", "🥉"][rank - 1] if rank <= 3 else f"#{rank}"
            lb_rows.append({
                "Rank":         medal,
                "Player":       entry["player"],
                "Score":        f"{entry['score']:.1f}",
                "Stars":        "⭐" * entry["stars"],
                "XP Earned":    entry.get("xp_earned", "—"),
                "Map Style":    entry.get("config", {}).get("map_style", "—"),
                "Routing":      entry.get("config", {}).get("routing", "—"),
                "Achievements": " ".join(entry["achievements"][:2]),
                "Date":         entry.get("timestamp", "—"),
            })
        st.dataframe(pd.DataFrame(lb_rows), use_container_width=True, hide_index=True)

        # Score distribution chart
        if len(board) > 2:
            scores = [e["score"] for e in board]
            players = [e["player"] for e in board]
            fig_lb = px.bar(
                x=players, y=scores, title="Score Distribution",
                color=scores, color_continuous_scale="blues",
                labels={"x": "Player", "y": "Score"},
            )
            fig_lb.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#94a3b8"), showlegend=False,
                coloraxis_showscale=False, margin=dict(t=40, b=0, l=0, r=0),
                yaxis=dict(range=[0, 105], gridcolor="#1e293b"),
            )
            st.plotly_chart(fig_lb, use_container_width=True)

# ── ACHIEVEMENTS TAB ─────────────────────────────────────────────────────────
with tab_ach:
    st.subheader("🏅 Achievement Gallery")
    earned_ids = st.session_state.earned_ids
    all_ach    = engine.all_achievements()

    total_possible_xp  = sum(a["xp"] for a in all_ach)
    earned_xp          = sum(a["xp"] for a in all_ach if a["id"] in earned_ids)
    pct = int(earned_xp / max(total_possible_xp, 1) * 100)

    prog_col, info_col = st.columns([3, 1])
    prog_col.progress(pct / 100, text=f"XP Progress: {earned_xp} / {total_possible_xp} ({pct}%)")
    info_col.metric("Badges Earned", f"{len(earned_ids)} / {len(all_ach)}")
    st.divider()

    # Grid of badge cards — 3 columns
    cols = st.columns(3)
    for i, ach in enumerate(all_ach):
        unlocked = ach["id"] in earned_ids
        bg    = "#064e3b" if unlocked else "#1e293b"
        fg    = "#6ee7b7" if unlocked else "#64748b"
        lock  = ""        if unlocked else "🔒 "
        with cols[i % 3]:
            st.markdown(f"""
            <div style="background:{bg};border-radius:10px;padding:14px;margin-bottom:10px;
                        border:1px solid {'#065f46' if unlocked else '#334155'}">
              <div style="font-size:1.4rem">{ach['name']}</div>
              <div style="color:{fg};font-size:.8rem;margin-top:4px">{lock}{ach['desc']}</div>
              <div style="color:#fbbf24;font-size:.8rem;margin-top:6px">✨ {ach['xp']} XP</div>
            </div>
            """, unsafe_allow_html=True)

# ── COMPARE TAB ──────────────────────────────────────────────────────────────
with tab_cmp:
    st.subheader("⚖️ Algorithm Comparison")
    st.caption("Run NN, Hybrid, and DP on the **same dataset** and compare results side by side.")

    cc1, cc2, cc3, cc4 = st.columns(4)
    cmp_stops  = cc1.slider("Stops",  5, 30, 15, 5, key="cmp_stops")
    cmp_buses  = cc2.slider("Buses",  1,  4,  2,    key="cmp_buses")
    cmp_seed   = cc3.number_input("Seed", 0, 9999, 42, key="cmp_seed")
    cmp_style  = cc4.selectbox("Map Style", MAP_STYLES, key="cmp_style")
    cmp_btn    = st.button("⚖️ Run Comparison", type="primary", key="cmp_btn")

    if cmp_btn:
        gen = DataGenerator(seed=int(cmp_seed))
        cmp_dataset = gen.generate_dataset(
            num_pickup_points=cmp_stops,
            num_buses=cmp_buses,
            map_width=100.0,
            map_height=100.0,
        )
        cmp_data = {}
        methods = ["nn", "hybrid"]
        if cmp_stops // max(cmp_buses, 1) <= 15:
            methods.append("dp")   # DP is only practical for small clusters

        with st.spinner("Running all algorithms…"):
            for method in methods:
                t0 = time.time()
                opt = BusRouteOptimization(use_rl=False)
                res = opt.optimize_routes(cmp_dataset, clustering_method="kmeans",
                                          routing_method=method, visualize=False)
                elapsed = time.time() - t0
                fuel = opt.route_optimizer.estimate_fuel_consumption(res["total_distance"])
                cmp_data[method] = {
                    "distance": res["total_distance"],
                    "time":     res["total_time"],
                    "fuel":     fuel,
                    "elapsed":  elapsed,
                }
        st.session_state.cmp_results = cmp_data
        st.success("✅ Comparison complete!")

    if st.session_state.cmp_results:
        cmp_data = st.session_state.cmp_results
        METHOD_LABELS = {"nn": "Nearest Neighbour", "hybrid": "Hybrid (NN+2-opt)", "dp": "DP (Exact)"}
        METHOD_COLORS = {"nn": "#fbbf24", "hybrid": "#38bdf8", "dp": "#34d399"}

        # Metric cards
        metric_cols = st.columns(len(cmp_data))
        for col, (method, vals) in zip(metric_cols, cmp_data.items()):
            col.markdown(f"**{METHOD_LABELS.get(method, method)}**")
            col.metric("🛣️ Distance", f"{vals['distance']:.1f} km")
            col.metric("⏰ Time",     f"{vals['time']:.0f} min")
            col.metric("⛽ Fuel",     f"{vals['fuel']:.1f} L")
            col.metric("⏱ Wall-clock",f"{vals['elapsed']:.2f}s")
        st.divider()

        # Grouped bar chart
        categories  = ["Distance (km)", "Time (min)", "Fuel (L)", "Speed (s×10)"]
        fig_cmp = go.Figure()
        for method, vals in cmp_data.items():
            fig_cmp.add_trace(go.Bar(
                name=METHOD_LABELS.get(method, method),
                x=categories,
                y=[vals["distance"], vals["time"], vals["fuel"], vals["elapsed"] * 10],
                marker_color=METHOD_COLORS.get(method, "#94a3b8"),
                text=[f"{v:.1f}" for v in [vals["distance"], vals["time"], vals["fuel"], vals["elapsed"]*10]],
                textposition="outside",
            ))
        fig_cmp.update_layout(
            barmode="group", title="Metric Comparison Across Algorithms",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#94a3b8"), legend=dict(bgcolor="rgba(0,0,0,0)"),
            yaxis=dict(gridcolor="#1e293b"), margin=dict(t=50, b=0, l=0, r=0),
        )
        st.plotly_chart(fig_cmp, use_container_width=True)

        # Radar chart
        radar_cats = ["Distance", "Time", "Fuel", "Speed"]
        # Normalise each metric so the best=1.0, worst=0.0
        def _norm(vals_list):
            lo, hi = min(vals_list), max(vals_list)
            if hi == lo:
                return [1.0] * len(vals_list)
            return [1.0 - (v - lo) / (hi - lo) for v in vals_list]

        dist_n  = _norm([v["distance"] for v in cmp_data.values()])
        time_n  = _norm([v["time"]     for v in cmp_data.values()])
        fuel_n  = _norm([v["fuel"]     for v in cmp_data.values()])
        speed_n = _norm([v["elapsed"]  for v in cmp_data.values()])

        fig_radar = go.Figure()
        for i, (method, vals) in enumerate(cmp_data.items()):
            scores = [dist_n[i], time_n[i], fuel_n[i], speed_n[i]]
            scores += [scores[0]]   # close the polygon
            fig_radar.add_trace(go.Scatterpolar(
                r=scores, theta=radar_cats + [radar_cats[0]],
                fill="toself", name=METHOD_LABELS.get(method, method),
                line=dict(color=METHOD_COLORS.get(method, "#94a3b8")),
                opacity=0.6,
            ))
        fig_radar.update_layout(
            polar=dict(
                bgcolor="#1e293b",
                radialaxis=dict(visible=True, range=[0, 1], gridcolor="#334155", color="#94a3b8"),
                angularaxis=dict(color="#94a3b8"),
            ),
            showlegend=True, title="Normalised Performance Radar (higher = better)",
            paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#94a3b8"),
            legend=dict(bgcolor="rgba(0,0,0,0)"),
            margin=dict(t=60, b=0, l=0, r=0), height=400,
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        # Show custom map for the last-run algorithm's routes
        best_method = min(cmp_data, key=lambda m: cmp_data[m]["distance"])
        st.caption(f"🗺️ Route map for best algorithm: **{METHOD_LABELS.get(best_method, best_method)}**")
        opt_best = BusRouteOptimization(use_rl=False)
        res_best = opt_best.optimize_routes(cmp_dataset, clustering_method="kmeans",
                                            routing_method=best_method, visualize=False)
        st.plotly_chart(
            render_custom_map(cmp_dataset, res_best, style=cmp_style, height=420),
            use_container_width=True,
        )

