"""
Booking-timing dynamics analysis module.
Simulates price trajectories as a function of booking lead time (days_left) across top routes.
Identifies empirical sweet spot booking windows and inflection points where fares surge.
Explicitly documented as an observational model-based estimate, not a causal claim.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import ProjectConfig
from src.features import engineer_features
from src.utils import save_table, setup_logger

logger = setup_logger("timing_analysis")


def analyze_booking_timing(
    best_pipe: Any,
    quantile_low_pipe: Any,
    quantile_high_pipe: Any,
    full_df: pd.DataFrame,
    config: ProjectConfig,
) -> pd.DataFrame:
    """
    Build predicted average price curves as days_left varies from min_days_left to max_days_left
    for the top routes, identify sweet spot windows and price surge thresholds,
    and save figures and tables.
    """
    logger.info("Executing Booking-Timing Dynamics Analysis...")

    # Identify the top N most frequent routes
    top_routes = full_df["route"].value_counts().head(config.top_routes_count).index.tolist()
    days_range = np.arange(config.min_days_left, config.max_days_left + 1)

    route_summaries = []
    fig, axes = plt.subplots(len(top_routes), 1, figsize=(12, 4 * len(top_routes)), sharex=True)
    if len(top_routes) == 1:
        axes = [axes]

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    for i, route in enumerate(top_routes):
        route_df = full_df[full_df["route"] == route]
        src_city, dest_city = route.split("_")

        # Typical flight characteristics for this route
        mode_airline = route_df["airline"].mode()[0]
        mode_stops = route_df["stops"].mode()[0]
        mode_dep = route_df["departure_time"].mode()[0]
        mode_arr = route_df["arrival_time"].mode()[0]
        mode_class = "Economy"  # Analyze standard economy fares
        median_duration = float(route_df["duration"].median())

        # Construct synthetic query grid across days_left
        grid_data = {
            "airline": [mode_airline] * len(days_range),
            "source_city": [src_city] * len(days_range),
            "destination_city": [dest_city] * len(days_range),
            "departure_time": [mode_dep] * len(days_range),
            "arrival_time": [mode_arr] * len(days_range),
            "stops": [mode_stops] * len(days_range),
            "class": [mode_class] * len(days_range),
            "duration": [median_duration] * len(days_range),
            "days_left": days_range,
        }
        query_df = pd.DataFrame(grid_data)
        query_df_feat = engineer_features(query_df)

        # Generate model predictions
        pred_log = best_pipe.predict(query_df_feat)
        pred_price = np.expm1(pred_log)

        # Quantile bounds
        if quantile_low_pipe and quantile_high_pipe:
            # Handle preprocessor if separate
            preprocessor = best_pipe.named_steps["preprocessor"]
            q_in = preprocessor.transform(query_df_feat)
            q_low = np.expm1(quantile_low_pipe.predict(q_in))
            q_high = np.expm1(quantile_high_pipe.predict(q_in))
            # Protect against quantile crossing
            q_low = np.minimum(q_low, pred_price)
            q_high = np.maximum(q_high, pred_price)
        else:
            q_low = pred_price * 0.85
            q_high = pred_price * 1.15

        # 1. Identify "Sweet Spot" Window (days where predicted price is within 5% of global minimum)
        min_price = np.min(pred_price)
        sweet_threshold = min_price * 1.05
        sweet_spot_indices = np.where(pred_price <= sweet_threshold)[0]
        sweet_spot_days = days_range[sweet_spot_indices]
        sweet_min_day = int(np.min(sweet_spot_days))
        sweet_max_day = int(np.max(sweet_spot_days))

        # 2. Identify Sharp Inflection Point (where consecutive day price change accelerates most rapidly)
        # Note: price typically rises as days_left decreases (moving from 50 to 1)
        # Price slope moving towards departure: (price[t] - price[t+1])
        daily_diff = pred_price[:-1] - pred_price[1:]
        # Find the threshold day where price starts rising sharply (> 300 INR/day or highest acceleration)
        sharp_idx = np.where(daily_diff > 350)[0]
        if len(sharp_idx) > 0:
            # Find the largest days_left where this surge starts
            sharp_threshold_day = int(days_range[np.max(sharp_idx)])
        else:
            sharp_threshold_day = int(days_range[np.argmax(daily_diff)])

        last_minute_price = float(pred_price[0])  # days_left = 1
        sweet_avg_price = float(np.mean(pred_price[sweet_spot_indices]))
        last_minute_penalty = round(last_minute_price - sweet_avg_price, 2)

        route_summaries.append(
            {
                "Route": route,
                "Airline": mode_airline,
                "Class": mode_class,
                "Stops": mode_stops,
                "Min_Predicted_Price_INR": round(float(min_price), 2),
                "Sweet_Spot_Window_Days": f"{sweet_min_day} - {sweet_max_day} days",
                "Sharp_Price_Rise_Threshold_Days": f"<= {sharp_threshold_day} days",
                "Last_Minute_Penalty_INR": last_minute_penalty,
                "Nature_of_Claim": "Model-based observational estimate (non-causal)",
            }
        )

        # Plot curve and quantile band
        ax = axes[i]
        ax.fill_between(
            days_range,
            q_low,
            q_high,
            color=colors[i % len(colors)],
            alpha=0.25,
            label="80% Prediction Band",
        )
        ax.plot(
            days_range,
            pred_price,
            color=colors[i % len(colors)],
            linewidth=2.2,
            label=f"Predicted Fare: {route}",
        )
        ax.axvspan(
            sweet_min_day,
            sweet_max_day,
            color="green",
            alpha=0.15,
            label=f"Sweet Spot ({sweet_min_day}-{sweet_max_day}d)",
        )
        ax.axvline(
            sharp_threshold_day,
            color="red",
            linestyle="--",
            label=f"Price Surge Threshold (<= {sharp_threshold_day}d)",
        )

        ax.set_title(f"Price Curve vs. Lead Time: {route} ({mode_airline}, {mode_class})", fontsize=11)
        ax.set_ylabel("Predicted Price (INR)")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, linestyle=":", alpha=0.6)

    axes[-1].set_xlabel("Days Left Before Departure (Booking Lead Time)")
    plt.tight_layout()
    fig.savefig(config.figures_dir / "booking_timing_curves.png", dpi=300)
    plt.close(fig)

    summary_df = pd.DataFrame(route_summaries)
    save_table(summary_df, config.tables_dir / "booking_timing_sweet_spots.csv", index=False)
    logger.info("Saved booking timing sweet spot analysis and figures.")

    return summary_df
