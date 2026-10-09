"""
Feature Engineering Module
Stress Classifier — Zoo Animals (Feeding Schedule & Keeper Routine Focus)
IT41043 — Intelligent Systems, Horizon Campus (2026)

Defines feature sets for:
- Baseline Model (Trajectory / Movement features only)
- Branch A Models (Tabular Feeding Schedule & Keeper Routine features)
- Branch B Models (Sequence / Time-Series features)
"""

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def get_feature_sets():
    """Returns baseline, full tabular, and categorical feature definitions."""
    baseline_features = [
        "Activity_Level(Animal's activity types)",
        'Location_Changes',
        'Boundary_Time_Min',
        'Rolling_Stereotypy_Rate'
    ]

    full_num_features = baseline_features + [
        'Inter_Meal_Interval_Min',
        'Diet_Variety_Score',
        'Keeper_Proximity_Duration_Min',
        'Time_to_Keeper_Arrival_Min',
        'Keeper_Presence_Duration_Calc',
        'Meal_Duration_Calc',
        'Keeper_Arrival_to_Meal_Start',
        'Month',
        'Day',
        'DayOfWeek'
    ]

    cat_features = ['Type of animal', 'Enclosure', 'Observation Time']

    return baseline_features, full_num_features, cat_features

def build_transformers():
    """Constructs scikit-learn ColumnTransformers for Baseline and Branch A."""
    baseline_features, full_num_features, cat_features = get_feature_sets()

    baseline_transformer = ColumnTransformer([
        ('num', StandardScaler(), baseline_features)
    ])

    full_transformer = ColumnTransformer([
        ('num', StandardScaler(), full_num_features),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), cat_features)
    ])

    return baseline_transformer, full_transformer
