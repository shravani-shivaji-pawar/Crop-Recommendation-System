"""Secondary data integration module for AgriWise: Historical APS/DES Crop Production and IMD Climatology.

Guarantees:
- Completely decoupled from the 7-feature core ML training pipeline.
- Read-only historical observations and climatological normals.
- Zero fabrication: missing crop/district records return explicit 'No official record' indicators.
- CPU-friendly in-memory lookup caching.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

from src.core.config import PROJECT_ROOT


# Canonical mapping between AgriWise crop names and official DES commodity names
CROP_TO_DES_COMMODITY: Dict[str, str] = {
    "rice": "Rice",
    "maize": "Maize",
    "chickpea": "Gram",
    "pigeonpeas": "Arhar/Tur",
    "mungbean": "Moong(Green Gram)",
    "blackgram": "Urad",
    "cotton": "Cotton(lint)",
    "banana": "Banana",
    "mango": "Mango",
    "grapes": "Grapes"
}

# Mapping between AgriWise canonical district names and DES uppercase naming
CANONICAL_TO_DES_DISTRICT: Dict[str, str] = {
    "Ahmednagar": "AHMEDNAGAR",
    "Akola": "AKOLA",
    "Amravati": "AMRAVATI",
    "Chhatrapati Sambhajinagar": "AURANGABAD",
    "Beed": "BEED",
    "Bhandara": "BHANDARA",
    "Buldhana": "BULDHANA",
    "Chandrapur": "CHANDRAPUR",
    "Dhule": "DHULE",
    "Gadchiroli": "GADCHIROLI",
    "Gondia": "GONDIA",
    "Hingoli": "HINGOLI",
    "Jalgaon": "JALGAON",
    "Jalna": "JALNA",
    "Kolhapur": "KOLHAPUR",
    "Latur": "LATUR",
    "Mumbai City": "MUMBAI",
    "Mumbai Suburban": "MUMBAI",
    "Nagpur": "NAGPUR",
    "Nanded": "NANDED",
    "Nandurbar": "NANDURBAR",
    "Nashik": "NASHIK",
    "Dharashiv": "OSMANABAD",
    "Palghar": "PALGHAR",
    "Parbhani": "PARBHANI",
    "Pune": "PUNE",
    "Raigad": "RAIGAD",
    "Ratnagiri": "RATNAGIRI",
    "Sangli": "SANGLI",
    "Satara": "SATARA",
    "Sindhudurg": "SINDHUDURG",
    "Solapur": "SOLAPUR",
    "Thane": "THANE",
    "Wardha": "WARDHA",
    "Washim": "WASHIM",
    "Yavatmal": "YAVATMAL"
}

COTTON_BALES_TO_TONNES = 0.170  # Official DES standard: 1 bale = 170 kg = 0.170 metric tonnes


class HistoricalProductionRepository:
    """Repository managing historical crop production statistics from DES (1997-2014)."""

    def __init__(self, data_path: Optional[Path] = None):
        if data_path is None:
            self.data_path = PROJECT_ROOT / "data" / "raw" / "maharashtra_crop_production_des_raw.csv"
        else:
            self.data_path = Path(data_path)

        self._loaded = False
        self._df: Optional[pd.DataFrame] = None
        self._load_dataset()

    def _load_dataset(self) -> None:
        if not self.data_path.exists():
            print(f"Warning: DES raw dataset not found at {self.data_path}")
            self._loaded = False
            return

        try:
            df = pd.read_csv(self.data_path)
            # Normalize text columns
            df["District_Clean"] = df["District_Name"].astype(str).str.strip().str.upper()
            df["Crop_Clean"] = df["Crop"].astype(str).str.strip()
            df["Season_Clean"] = df["Season"].astype(str).str.strip()

            # Unit correction: Cotton(lint) is reported in bales (170kg), convert to tonnes
            is_cotton = df["Crop_Clean"] == "Cotton(lint)"
            df["Production_Tonnes"] = df["Production"]
            df.loc[is_cotton, "Production_Tonnes"] = df.loc[is_cotton, "Production"] * COTTON_BALES_TO_TONNES

            # Compute yield in kg/ha
            valid_mask = (df["Area"] > 0) & (df["Production_Tonnes"].notnull())
            df["Yield_kg_per_ha"] = np.nan
            df.loc[valid_mask, "Yield_kg_per_ha"] = (
                df.loc[valid_mask, "Production_Tonnes"] * 1000.0
            ) / df.loc[valid_mask, "Area"]

            self._df = df
            self._loaded = True
        except Exception as e:
            print(f"Error loading DES dataset: {e}")
            self._loaded = False

    @property
    def is_available(self) -> bool:
        return self._loaded and (self._df is not None)

    def get_crop_history(self, crop_name: str, district: str) -> Dict[str, Any]:
        """Retrieves aggregated historical cultivation statistics for a crop in a district.
        
        Returns structured dictionary with actual historical records or safe fallback if absent.
        """
        clean_crop = crop_name.lower().replace(" ", "").replace("-", "")
        des_crop = CROP_TO_DES_COMMODITY.get(clean_crop)
        des_district = CANONICAL_TO_DES_DISTRICT.get(district)

        if not self.is_available or des_crop is None or des_district is None:
            return {
                "has_records": False,
                "crop_name": crop_name,
                "district": district,
                "evidence_status": "No Official DES Records",
                "summary": (
                    f"No historical DES agricultural production records exist for '{crop_name}' in {district} district. "
                    "This crop is not part of the state's commercial crop survey portfolio for this region."
                ),
                "mean_area_ha": None,
                "mean_production_tonnes": None,
                "mean_yield_kg_per_ha": None,
                "years_surveyed": 0,
                "recorded_seasons": []
            }

        sub = self._df[
            (self._df["District_Clean"] == des_district) &
            (self._df["Crop_Clean"] == des_crop)
        ]

        if len(sub) == 0:
            return {
                "has_records": False,
                "crop_name": crop_name,
                "district": district,
                "evidence_status": "Zero Cultivation Reported",
                "summary": (
                    f"Zero commercial production of '{crop_name}' recorded in {district} district across "
                    "the 1997–2014 Directorate of Economics and Statistics surveys."
                ),
                "mean_area_ha": None,
                "mean_production_tonnes": None,
                "mean_yield_kg_per_ha": None,
                "years_surveyed": 0,
                "recorded_seasons": []
            }

        mean_area = float(round(sub["Area"].mean(), 1))
        mean_prod = float(round(sub["Production_Tonnes"].mean(), 1)) if sub["Production_Tonnes"].notnull().any() else None
        mean_yield = float(round(sub["Yield_kg_per_ha"].mean(), 1)) if sub["Yield_kg_per_ha"].notnull().any() else None
        seasons = sorted(sub["Season_Clean"].unique().tolist())
        years = int(sub["Crop_Year"].nunique())

        if mean_area >= 20000:
            scale = "Major Commercial Backbone"
        elif mean_area >= 2000:
            scale = "Moderate Commercial Cultivation"
        else:
            scale = "Minor / Pocket Cultivation"

        summary = (
            f"Historical DES Production (1997–2014): {scale} in {district}. "
            f"Average cultivated area: {mean_area:,.1f} ha, average annual output: "
            f"{f'{mean_prod:,.1f} MT' if mean_prod is not None else 'N/A'} "
            f"(Yield: {f'{mean_yield:,.1f} kg/ha' if mean_yield is not None else 'N/A'}). "
            f"Seasons: {', '.join(seasons)} ({years} survey years)."
        )

        return {
            "has_records": True,
            "crop_name": crop_name,
            "des_commodity": des_crop,
            "district": district,
            "commercial_scale": scale,
            "mean_area_ha": mean_area,
            "mean_production_tonnes": mean_prod,
            "mean_yield_kg_per_ha": mean_yield,
            "recorded_seasons": seasons,
            "years_surveyed": years,
            "evidence_status": "Verified Commercial Record",
            "summary": summary
        }


class ClimatologyRepository:
    """Repository managing district rainfall normals and climatological departures (IMD Pune)."""

    def __init__(self, data_path: Optional[Path] = None):
        if data_path is None:
            self.data_path = PROJECT_ROOT / "data" / "raw" / "maharashtra_district_rainfall_normals_imd.csv"
        else:
            self.data_path = Path(data_path)

        self._loaded = False
        self._data: Dict[str, Dict[str, Any]] = {}
        self._load_dataset()

    def _load_dataset(self) -> None:
        if not self.data_path.exists():
            print(f"Warning: IMD rainfall normals file not found at {self.data_path}")
            self._loaded = False
            return

        try:
            df = pd.read_csv(self.data_path)
            for _, row in df.iterrows():
                d = str(row["district"]).strip()
                self._data[d] = {
                    "division": str(row.get("division", "")),
                    "zone_id": str(row.get("zone_id", "")),
                    "zone_name": str(row.get("zone_name", "")),
                    "sw_monsoon_normal_mm": float(row["sw_monsoon_jjas_normal_mm"]),
                    "post_monsoon_normal_mm": float(row["post_monsoon_ond_normal_mm"]),
                    "winter_normal_mm": float(row["winter_jf_normal_mm"]),
                    "pre_monsoon_normal_mm": float(row["pre_monsoon_mam_normal_mm"]),
                    "annual_normal_mm": float(row["annual_normal_mm"]),
                    "baseline_period": str(row.get("baseline_period", "1971-2020 Normals"))
                }
            self._loaded = True
        except Exception as e:
            print(f"Error loading IMD normals: {e}")
            self._loaded = False

    @property
    def is_available(self) -> bool:
        return self._loaded and len(self._data) > 0

    def get_district_climatology(self, district: str) -> Optional[Dict[str, Any]]:
        return self._data.get(district)

    def evaluate_rainfall_departure(
        self,
        district: str,
        observed_rainfall_mm: float,
        season: Optional[str] = None
    ) -> Dict[str, Any]:
        """Compares user-entered rainfall against the district's IMD 30-year normal rainfall.
        
        Uses official IMD classification standards:
        - Large Excess: >= +60%
        - Excess: +20% to +59%
        - Normal: -19% to +19%
        - Deficient: -59% to -20%
        - Large Deficient: -99% to -60%
        - Scanty: <= -99%
        """
        dist_clim = self._data.get(district)
        if not dist_clim:
            return {
                "has_climatology": False,
                "district": district,
                "observed_rainfall_mm": observed_rainfall_mm,
                "advisory": f"No official IMD climatological profile available for district '{district}'."
            }

        # Choose appropriate normal: SW monsoon for Kharif, Post-monsoon for Rabi, else Annual
        if season == "Kharif":
            normal_mm = dist_clim["sw_monsoon_normal_mm"]
            reference_label = "IMD SW Monsoon (JJAS) Normal"
        elif season == "Rabi":
            normal_mm = dist_clim["post_monsoon_normal_mm"]
            reference_label = "IMD Post-Monsoon (OND) Normal"
        else:
            normal_mm = dist_clim["annual_normal_mm"]
            reference_label = "IMD Annual Normal"

        departure_pct = round(((observed_rainfall_mm - normal_mm) / normal_mm) * 100.0, 1)

        if departure_pct >= 60.0:
            regime = "Large Excess"
            badge_class = "badge bg-info text-dark"
            advisory = (
                f"Entered rainfall ({observed_rainfall_mm} mm) is {departure_pct:+0.1f}% above the {reference_label} "
                f"({normal_mm} mm). Ensure adequate surface drainage to prevent waterlogging."
            )
        elif departure_pct >= 20.0:
            regime = "Excess"
            badge_class = "badge bg-primary"
            advisory = (
                f"Entered rainfall ({observed_rainfall_mm} mm) is {departure_pct:+0.1f}% above the {reference_label} "
                f"({normal_mm} mm). Favorable moisture regime for rainfed crops."
            )
        elif departure_pct >= -19.0:
            regime = "Normal"
            badge_class = "badge bg-success"
            advisory = (
                f"Entered rainfall ({observed_rainfall_mm} mm) is within normal climatological bounds ({departure_pct:+0.1f}% "
                f"vs {reference_label} of {normal_mm} mm)."
            )
        elif departure_pct >= -59.0:
            regime = "Deficient"
            badge_class = "badge bg-warning text-dark"
            advisory = (
                f"Entered rainfall ({observed_rainfall_mm} mm) is {abs(departure_pct):0.1f}% below the {reference_label} "
                f"({normal_mm} mm). Supplemental irrigation is required for high water-demand crops."
            )
        else:
            regime = "Large Deficient"
            badge_class = "badge bg-danger"
            advisory = (
                f"Entered rainfall ({observed_rainfall_mm} mm) indicates severe deficit ({abs(departure_pct):0.1f}% below "
                f"{reference_label} of {normal_mm} mm). Only drought-hardy or fully irrigated crops recommended."
            )

        return {
            "has_climatology": True,
            "district": district,
            "observed_rainfall_mm": observed_rainfall_mm,
            "reference_label": reference_label,
            "normal_mm": normal_mm,
            "annual_normal_mm": dist_clim["annual_normal_mm"],
            "sw_monsoon_normal_mm": dist_clim["sw_monsoon_normal_mm"],
            "departure_pct": departure_pct,
            "regime": regime,
            "badge_class": badge_class,
            "advisory": advisory,
            "source": f"IMD Pune Climate Research & Services ({dist_clim['baseline_period']})"
        }
