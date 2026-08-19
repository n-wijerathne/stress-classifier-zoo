"""
generate_synthetic_data.py

IMPORTANT — READ BEFORE USE
----------------------------------------------------------------------
This script generates a SYNTHETIC dataset that follows the schema and
statistical assumptions described in Section 2.1 of the Milestone 2
methodology document. It is a stand-in for real field data collected
at Dehiwala National Zoological Gardens.

It exists so that the rest of the pipeline (preprocessing, feature
engineering, baseline model, tabular models, LSTM model, evaluation)
can be built, tested, and demonstrated end-to-end BEFORE real data
collection is complete or approved.

Every assumption below is stated explicitly and, where possible, is
grounded in the cited literature (see methodology bibliography):
  - Stress-positive base rate ~15-25% for elephants, ~10-20% for
    macaques: informed by stereotypy occurrence rates reported in
    Fitskie et al. (2024) and McGuire et al. (2024).
  - Stress probability rises in the pre-feeding / anticipatory window
    and around keeper arrival: informed by McGuire et al. (2024),
    who found stereotypy onset tied to keeper arrival time.

DO NOT present output from this script as real collected data.
When real data is available, replace this script's output with the
actual merged log file at data/processed/session_bins.csv and the
rest of the pipeline (preprocess.py onwards) runs unchanged.
----------------------------------------------------------------------
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

RNG_SEED = 42
rng = np.random.default_rng(RNG_SEED)

SPECIES_CONFIG = {
    "elephant": {"n_individuals": 3, "base_stress_rate": 0.20},
    "macaque":  {"n_individuals": 6, "base_stress_rate": 0.15},
}

N_WEEKS = 8
DAYS_PER_WEEK = 7
BLOCKS_PER_DAY = 2          # e.g. morning / afternoon observation blocks
BIN_MINUTES = 5
BLOCK_DURATION_MINUTES = 90  # each observation block covered by scan sampling

KEEPER_IDS = [f"KPR_{i:02d}" for i in range(1, 6)]  # anonymised rotating codes


def _session_bins_per_block():
    return BLOCK_DURATION_MINUTES // BIN_MINUTES


def generate_for_species(species: str, cfg: dict, start_date: datetime) -> pd.DataFrame:
    records = []
    n_days = N_WEEKS * DAYS_PER_WEEK

    for individual_idx in range(cfg["n_individuals"]):
        animal_id = f"{species.upper()}_{individual_idx + 1:02d}"

        for day in range(n_days):
            date = start_date + timedelta(days=day)

            # One feeding event and one keeper-arrival event per day, jittered.
            meal_time = date.replace(hour=9, minute=0) + timedelta(
                minutes=int(rng.normal(0, 20))
            )
            keeper_arrival = meal_time - timedelta(minutes=int(rng.uniform(10, 40)))
            keeper_id = rng.choice(KEEPER_IDS)
            diet_variety_score = rng.integers(1, 6)  # 1-5 scale, more = more varied
            prev_meal_time = meal_time - timedelta(
                hours=float(rng.uniform(18, 26))
            )
            inter_meal_interval_hr = (meal_time - prev_meal_time).total_seconds() / 3600

            for block in range(BLOCKS_PER_DAY):
                block_start = date.replace(hour=8 + block * 6, minute=0)

                for b in range(_session_bins_per_block()):
                    bin_start = block_start + timedelta(minutes=b * BIN_MINUTES)
                    minutes_to_meal = (meal_time - bin_start).total_seconds() / 60
                    minutes_since_keeper_arrival = (
                        bin_start - keeper_arrival
                    ).total_seconds() / 60

                    # Anticipatory window: probability of stress rises as the
                    # animal approaches the feeding/keeper-arrival window,
                    # consistent with McGuire et al. (2024).
                    anticipatory_boost = 0.0
                    if 0 <= minutes_to_meal <= 45:
                        anticipatory_boost = 0.25 * (1 - minutes_to_meal / 45)
                    if -10 <= minutes_since_keeper_arrival <= 20:
                        anticipatory_boost += 0.15

                    stress_prob = min(
                        0.95, cfg["base_stress_rate"] + anticipatory_boost
                    )
                    stress_label = int(rng.random() < stress_prob)

                    keeper_proximity_minutes = max(
                        0.0,
                        20 - abs(minutes_since_keeper_arrival)
                        if abs(minutes_since_keeper_arrival) < 20
                        else 0.0,
                    )

                    # Baseline-only movement/trajectory features (used by the
                    # baseline model in src/baseline_model.py). Loosely
                    # correlated with the true label to emulate a real but
                    # weaker signal, per Zuerl et al. (2022) / Wang et al. (2024).
                    activity_level = np.clip(
                        rng.normal(0.4 + 0.2 * stress_label, 0.15), 0, 1
                    )
                    boundary_time_pct = np.clip(
                        rng.normal(0.3 + 0.15 * stress_label, 0.12), 0, 1
                    )
                    location_changes = max(
                        0, int(rng.poisson(3 + 4 * stress_label))
                    )

                    records.append(
                        {
                            "animal_id": animal_id,
                            "species": species,
                            "date": date.date().isoformat(),
                            "bin_start": bin_start.isoformat(),
                            "keeper_id": keeper_id,
                            "meal_time": meal_time.isoformat(),
                            "inter_meal_interval_hr": round(inter_meal_interval_hr, 2),
                            "diet_variety_score": int(diet_variety_score),
                            "keeper_arrival_time": keeper_arrival.isoformat(),
                            "minutes_to_meal": round(minutes_to_meal, 1),
                            "minutes_since_keeper_arrival": round(
                                minutes_since_keeper_arrival, 1
                            ),
                            "keeper_proximity_minutes": round(
                                keeper_proximity_minutes, 1
                            ),
                            "activity_level": round(float(activity_level), 3),
                            "boundary_time_pct": round(float(boundary_time_pct), 3),
                            "location_changes": location_changes,
                            "stress_label": stress_label,
                        }
                    )

    return pd.DataFrame.from_records(records)


def main():
    start_date = datetime(2026, 6, 1, 0, 0)
    frames = [
        generate_for_species(species, cfg, start_date)
        for species, cfg in SPECIES_CONFIG.items()
    ]
    df = pd.concat(frames, ignore_index=True)

    out_path = "data/processed/synthetic_session_bins.csv"
    df.to_csv(out_path, index=False)

    print(f"Generated {len(df):,} synthetic session-bin records -> {out_path}")
    print(df.groupby("species")["stress_label"].mean().rename("stress_positive_rate"))


if __name__ == "__main__":
    main()
