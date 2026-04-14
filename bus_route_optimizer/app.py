"""
🚌 Bus Route Optimizer — Gamified Streamlit UI
Run with:  streamlit run app.py   (from inside bus_route_optimizer/)
"""

import os, sys, time
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# ── make sure local modules are importable regardless of cwd ──────────────────
sys.path.insert(0, os.path.dirname(__file__))

from modules.data_generator import DataGenerator
from modules.clustering import Clustering
from modules.route_optimizer import RouteOptimizer
from modules.visualization import RouteVisualizer
from modules.reinforcement_learning import RLOptimizer
from modules.gamification import GamificationEngine
from main import BusRouteOptimization

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
  .badge        { background:#334155; border-radius:8px; padding:6px 10px;
                  display:inline-block; margin:3px; font-size:.85rem; }
  .achievement  { background:#064e3b; color:#6ee7b7; border-radius:8px;
                  padding:8px 12px; margin:4px; display:inline-block; }
  .metric-card  { background:#0f172a; border-radius:10px; padding:14px;
                  border-left: 4px solid #38bdf8; }
  h1 { color:#f1f5f9 !important; }
</style>
""", unsafe_allow_html=True)

# ── State ─────────────────────────────────────────────────────────────────────
if "gamification" not in st.session_state:
    st.session_state.gamification = GamificationEngine()
if "history" not in st.session_state:
    st.session_state.history = []

engine: GamificationEngine = st.session_state.gamification

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🎮 Game Settings")
    st.divider()

    player_name = st.text_input("👤 Player Name", value="Optimizer")
    st.divider()
    st.subheader("📍 Dataset")
    num_stops  = st.slider("Pickup Stops",  min_value=5,  max_value=60, value=30, step=5)
    num_buses  = st.slider("Number of Buses", min_value=1, max_value=6, value=2)
    radius_km  = st.slider("City Radius (km)", min_value=2, max_value=25, value=10)
    seed       = st.number_input("Random Seed", min_value=0, max_value=9999, value=42, step=1)

    st.divider()
    st.subheader("⚙️ Algorithm")
    clustering_method = st.selectbox("Clustering",    ["kmeans", "dbscan"])
    routing_method    = st.selectbox("Routing (TSP)", ["hybrid", "nn", "dp"])
    use_rl            = st.checkbox("Enable RL Optimizer", value=True)

    st.divider()
    run_btn = st.button("🚀 Run Optimisation", use_container_width=True, type="primary")

    st.divider()
    st.subheader("📊 Session Stats")
    st.metric("Total Runs", len(st.session_state.history))
    if st.session_state.history:
        best = max(st.session_state.history, key=lambda x: x["score"])
        st.metric("Best Score", f"{best['score']:.1f}")

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🚌 Bus Route Optimizer")
st.caption("Shortest-route planning with K-Means clustering, TSP solving, and Q-Learning — now gamified!")

# ── Run optimisation ──────────────────────────────────────────────────────────
if run_btn:
    with st.spinner("⚙️ Generating dataset and optimising routes…"):
        t0 = time.time()
        generator = DataGenerator(seed=int(seed))
        dataset   = generator.generate_dataset(
            num_pickup_points=num_stops,
            num_buses=num_buses,
            city_center=(40.7128, -74.0060),
            radius_km=radius_km,
        )
        optimizer = BusRouteOptimization(use_rl=use_rl)
        results   = optimizer.optimize_routes(
            dataset,
            clustering_method=clustering_method,
            routing_method=routing_method,
            visualize=False,
        )
        elapsed = time.time() - t0

    total_fuel = optimizer.route_optimizer.estimate_fuel_consumption(results["total_distance"])
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
    }, achievements)

    st.session_state.history.append({**score_data, "results": results,
                                      "dataset": dataset, "achievements": achievements,
                                      "elapsed": elapsed, "fuel": total_fuel})
    st.success(f"✅ Done in {elapsed:.2f}s")

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

    # Score breakdown bar
    breakdown = {
        "Distance": latest["dist_score"],
        "Time":     latest["time_score"],
        "Fuel":     latest["fuel_score"],
        "Speed":    latest["speed_bonus"] * 10,
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
        yaxis=dict(range=[0, 110], showgrid=False, visible=False),
        xaxis=dict(color="#94a3b8"),
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
tab_map, tab_routes, tab_stats, tab_board = st.tabs(["🗺️ Map", "📍 Routes", "📊 Charts", "🏆 Leaderboard"])

# ── MAP TAB ───────────────────────────────────────────────────────────────────
with tab_map:
    try:
        from streamlit_folium import st_folium
        import folium

        pickup_locs = dataset["pickup_locations"]
        depot       = dataset["depot"]
        dest        = dataset["destination"]
        center_lat  = float(np.mean(pickup_locs[:, 0]))
        center_lon  = float(np.mean(pickup_locs[:, 1]))

        m = folium.Map(location=[center_lat, center_lon], zoom_start=13)
        COLORS = ["red","blue","green","purple","orange","darkred","cadetblue","darkgreen","darkblue","beige"]

        for bus_idx, detail in enumerate(results["route_details"]):
            color = COLORS[bus_idx % len(COLORS)]
            route_pts = detail["waypoints"]
            coords = [[float(p[0]), float(p[1])] for p in route_pts]
            folium.PolyLine(coords, color=color, weight=3, opacity=0.8,
                            tooltip=f"Bus {bus_idx+1} – {detail['distance']:.1f} km").add_to(m)
            for stop_idx, pt_idx in enumerate(detail["route"]):
                pt = pickup_locs[pt_idx]
                folium.CircleMarker(
                    location=[float(pt[0]), float(pt[1])], radius=7,
                    popup=f"Bus {bus_idx+1} Stop {stop_idx+1}",
                    color=color, fill=True, fill_color=color, fill_opacity=0.8,
                ).add_to(m)

        folium.Marker(depot.tolist(),  popup="🏠 Depot",       icon=folium.Icon(color="green",  icon="home")).add_to(m)
        folium.Marker(dest.tolist(),   popup="🏫 Destination", icon=folium.Icon(color="purple", icon="flag")).add_to(m)
        st_folium(m, use_container_width=True, height=500)
    except Exception as e:
        st.error(f"Map error: {e}")

# ── ROUTES TAB ────────────────────────────────────────────────────────────────
with tab_routes:
    for i, detail in enumerate(results["route_details"]):
        with st.expander(f"🚌 Bus {i+1}  —  {detail['num_stops']} stops  •  {detail['distance']:.1f} km  •  {detail['time_minutes']:.0f} min"):
            st.write(f"**Route order (pickup indices):** {detail['route']}")
            st.write(f"**Fuel estimate:** {detail['fuel_liters']:.2f} L")
            st.write(f"**Passenger load:** {detail['total_load']} pax")

# ── STATS TAB ─────────────────────────────────────────────────────────────────
with tab_stats:
    bus_labels = [f"Bus {i+1}" for i in range(len(results["route_details"]))]
    dists = [d["distance"] for d in results["route_details"]]
    times = [d["time_minutes"] for d in results["route_details"]]
    loads = [d["total_load"] for d in results["route_details"]]

    c1, c2, c3 = st.columns(3)
    def _bar(title, labels, vals, color):
        fig = px.bar(x=labels, y=vals, title=title, color_discrete_sequence=[color])
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          margin=dict(t=40,b=0,l=0,r=0), showlegend=False,
                          font=dict(color="#94a3b8"), yaxis=dict(gridcolor="#1e293b"))
        return fig
    c1.plotly_chart(_bar("Distance (km)", bus_labels, dists, "#38bdf8"), use_container_width=True)
    c2.plotly_chart(_bar("Time (min)",    bus_labels, times, "#818cf8"), use_container_width=True)
    c3.plotly_chart(_bar("Passengers",    bus_labels, loads, "#34d399"), use_container_width=True)

    # History line chart
    if len(st.session_state.history) > 1:
        hist_scores = [h["score"] for h in st.session_state.history]
        fig_hist = px.line(y=hist_scores, markers=True, title="Score History",
                           labels={"y":"Score","index":"Run #"},
                           color_discrete_sequence=["#38bdf8"])
        fig_hist.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                font=dict(color="#94a3b8"), yaxis=dict(range=[0,105],gridcolor="#1e293b"))
        st.plotly_chart(fig_hist, use_container_width=True)

# ── LEADERBOARD TAB ───────────────────────────────────────────────────────────
with tab_board:
    board = engine.get_leaderboard(10)
    if not board:
        st.info("No scores yet – run an optimisation!")
    else:
        for rank, entry in enumerate(board, 1):
            medal = ["🥇","🥈","🥉"][rank-1] if rank <= 3 else f"#{rank}"
            with st.container():
                cols = st.columns([1, 3, 2, 2, 4])
                cols[0].markdown(f"**{medal}**")
                cols[1].markdown(f"**{entry['player']}**")
                cols[2].markdown(f"🎯 {entry['score']:.1f}")
                cols[3].markdown("⭐" * entry["stars"])
                cols[4].markdown(" ".join(entry["achievements"][:3]))
            st.divider()

