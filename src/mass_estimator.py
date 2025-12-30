# src/mass_estimator.py

from typing import Dict


# -------------------------
# SABİT YOĞUNLUKLAR (kg/m3)
# -------------------------
DENSITIES = {
    "glass object": 2500.0,
    "metal object": 5275.0,    # (Al 2700 + Steel 7850) / 2
    "plastic object": 1400.0,
    "cardboard object": 950.0
}


# -------------------------
# SOLID FRACTION (φ)
# -------------------------
SOLID_FRACTIONS = {
    "glass object": {
        "low":  0.40,
        "mid":  0.60,
        "high": 0.80
    },
    "metal object": {
        "low":  0.20,
        "mid":  0.35,
        "high": 0.50
    },
    "plastic object": {
        "low":  0.10,
        "mid":  0.20,
        "high": 0.30
    },
    "cardboard object": {
        "low":  0.08,
        "mid":  0.15,
        "high": 0.25
    }
}


def estimate_mass_scenarios(
    volume_m3: float,
    material_label: str
) -> Dict[str, float]:
    """
    3 senaryolu kütle hesabı.

    m = density * (phi * volume)

    Returns:
        {
          "density_kg_m3": float,
          "mass_low_kg": float,
          "mass_mid_kg": float,
          "mass_high_kg": float
        }
    """

    if material_label not in DENSITIES:
        raise ValueError(f"Unknown material label: {material_label}")

    rho = DENSITIES[material_label]
    phi_set = SOLID_FRACTIONS[material_label]

    results = {
        "density_kg_m3": rho
    }

    for key in ["low", "mid", "high"]:
        phi = phi_set[key]
        mass = rho * (phi * volume_m3)
        results[f"mass_{key}_kg"] = float(mass)

    return results
