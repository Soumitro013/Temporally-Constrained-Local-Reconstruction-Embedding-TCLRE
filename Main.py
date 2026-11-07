import pandas as pd
from TCLRE import TCLRE

# Load SCADA dataset
data = pd.read_csv("SCADA.csv")

# Ensure observations are chronologically ordered
if "Timestamp" in data.columns:
    data["Timestamp"] = pd.to_datetime(data["Timestamp"])
    data = data.sort_values("Timestamp").reset_index(drop=True)

# Select sensor features
feature_columns = [
    "Generator temperature",
    "Bearing temperature",
    "Vibration (g)",
    "Power (kw)"
]

X = data[feature_columns].values

# Optional operating-regime information
regimes = (
    data["Operating Regime"].values
    if "Operating Regime" in data.columns
    else None
)

# Initialise TCLRE
model = TCLRE(
    n_neighbors=6,
    temporal_window=10,
    temporal_strength=0.5,
    max_dim=2
)

# Generate the reduced-dimensional representation
Y = model.fit_transform(X, regimes=regimes)

print("Selected dimensions:", model.n_components_)
print("Reduced data shape:", Y.shape)

print("\nDimension evaluation:")
for d, scores in model.dimension_scores_.items():
    print(f"Dimension {d}: {scores}")

# Store the reduced representation
result = pd.DataFrame(
    Y,
    columns=[f"TCLRE_{i+1}" for i in range(model.n_components_)]
)

result.to_csv("TCLRE_Transformed_Data.csv", index=False)

import numpy as np

# Example incoming SCADA observation:
# Replace these illustrative values with live sensor readings.
new_reading = np.array([
    85.2,   # Generator temperature
    78.1,   # Bearing temperature
    0.17,   # Vibration
    420.0   # Power
])

# Supply the regime label when available
current_regime = None

# Calculate its low-dimensional representation
new_embedding = model.transform_one(
    new_reading,
    regime=current_regime
)

print("New SCADA observation:", new_reading)
print("TCLRE representation:", new_embedding)

split = int(0.8 * len(X))

X_train = X[:split]
X_test = X[split:]

r_train = regimes[:split] if regimes is not None else None
r_test = regimes[split:] if regimes is not None else None

model = TCLRE(
    n_neighbors=6,
    temporal_window=10,
    temporal_strength=0.5,
    max_dim=2
)

Y_train = model.fit_transform(X_train, regimes=r_train)
Y_test = model.transform(X_test, regimes=r_test)
