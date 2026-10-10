# Data dictionary

`processed/animal_stress_data.csv` - **real field data** collected by the authors at the National
Zoological Gardens, Dehiwala (instantaneous scan sampling + keeper/feeding log), one row per
animal session (190 rows, 28 observation days, 2026-04-01 to 2026-05-25, Asian elephants and toque
macaques). **Individual animal IDs are deliberately removed** from this public file (see "Data &
Ethics" in the main README). The full-detail file, if you have it, goes in `raw/` (gitignored).

| Column | Meaning |
|---|---|
| `Type of animal` | Species: Asian Elephant / Toque Macaque |
| `Enclosure` | Enclosure code (E1, E2 elephants; M1, M2 macaques) |
| `Observation_Date` | Observation day (m/d/YYYY) - used as the grouping key for all splits |
| `Observation Time` | Observation block: Morning / Afternoon |
| `Keeper_Arrival_Time`, `Keeper_Departure_Time` | Clock time (H:MM) the keeper arrived at / left the enclosure |
| `Meal_Start_Time`, `Meal_End_Time` | Clock time (H:MM) the meal started / ended |
| `Inter_Meal_Interval_Min` | Minutes since the previous meal |
| `Diet_Variety_Score` | Diet variety score from the feeding log (integer) |
| `Keeper_Proximity_Duration_Min` | Minutes the keeper was near the animal |
| `Time_to_Keeper_Arrival_Min` | Minutes from the observation start to keeper arrival |
| `Activity_Level(Animal's activity types)` | Activity score from the observation sheet (integer) |
| `Location_Changes` | Number of location changes within the enclosure |
| `Boundary_Time_Min` | Minutes spent near the enclosure boundary |
| `Rolling_Stereotypy_Rate` | Rolling rate of stereotypic behaviour (0-1) |
| `Stress_Label` | Target: `Stress_Present` / `Stress_Absent` |

Columns are renamed to snake_case in `src/preprocess.py`. Derived columns (keeper presence
duration, meal duration, keeper-arrival-to-meal interval, binary `target`, `group_key`) are created
there too and written to `processed/model_ready.csv` (regenerated, not committed).

> Column descriptions follow the field-sheet column names; edit this table if your protocol defines
> any of them differently.
