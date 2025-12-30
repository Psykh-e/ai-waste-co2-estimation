# src/co2_estimator.py

from typing import Dict


# -------------------------------------------------
# CO2e TASARRUF FAKTÖRLERİ (kg CO2e / kg)
# -------------------------------------------------
CO2_FACTORS = {
    "glass object": 0.315,
    "metal object": 3.755,
    "plastic object": 0.902,
    "cardboard object": 0.140
}


def estimate_co2_scenarios(
    material_label: str,
    mass_low_kg: float,
    mass_mid_kg: float,
    mass_high_kg: float
) -> Dict[str, float]:
    """
    3 senaryolu CO2e tasarruf hesabı.

    CO2e = mass * factor

    Returns:
        {
          "co2_factor_kg_per_kg": float,
          "co2_low_kg": float,
          "co2_mid_kg": float,
          "co2_high_kg": float
        }
    """

    if material_label not in CO2_FACTORS:
        raise ValueError(f"Unknown material label for CO2: {material_label}")

    factor = CO2_FACTORS[material_label]

    return {
        "co2_factor_kg_per_kg": factor,
        "co2_low_kg": float(mass_low_kg * factor),
        "co2_mid_kg": float(mass_mid_kg * factor),
        "co2_high_kg": float(mass_high_kg * factor)
    }
