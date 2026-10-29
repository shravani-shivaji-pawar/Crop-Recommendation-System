"""RegionalDecisionLayer: Non-destructive regional and seasonal decision synthesis engine."""

from typing import Dict, Any, List, Optional, Union
from src.core.exceptions import ValidationError
from src.modules.regional.maharashtra_data import (
    MAHARASHTRA_DISTRICTS,
    AGRO_CLIMATIC_ZONES,
    CROP_REGIONAL_PROFILES,
    SEASONS,
    ViabilityStatus
)
from src.modules.regional.secondary_data import (
    HistoricalProductionRepository,
    ClimatologyRepository
)


class RegionalDecisionLayer:
    """Evaluates crop recommendation candidates against regional and seasonal constraints.
    
    Guarantees:
    - Original ML Top-5 rankings and confidence scores are NEVER modified.
    - Evaluates each candidate against certified Maharashtra Agro-Climatic Zones and university calendars.
    - Categorizes candidates into 4 mutually exclusive statuses.
    - Transparently documents the agronomic rationale and literature source for every assessment.
    - Enriches assessments with read-only historical DES production statistics and IMD rainfall departures.
    """

    def __init__(self):
        self.districts = MAHARASHTRA_DISTRICTS
        self.zones = AGRO_CLIMATIC_ZONES
        self.profiles = CROP_REGIONAL_PROFILES
        self.des_repo = HistoricalProductionRepository()
        self.climatology_repo = ClimatologyRepository()

    def validate_district(self, district: str) -> str:
        """Validates district against the 36 official Maharashtra districts (case-insensitive)."""
        if not district or not isinstance(district, str):
            raise ValidationError("District name is required for regional decision layer.", field="district")

        clean_d = district.strip().lower()
        matched = None
        for canonical in self.districts.keys():
            if canonical.lower() == clean_d:
                matched = canonical
                break

        if matched is None:
            # Check common spelling variations
            variations = {
                "aurangabad": "Chhatrapati Sambhajinagar",
                "osmanabad": "Dharashiv",
                "mumbai": "Mumbai City",
                "bombay": "Mumbai City"
            }
            matched = variations.get(clean_d)

        if matched is None:
            raise ValidationError(
                f"Unknown Maharashtra district: '{district}'. Must be one of the 36 districts.",
                field="district",
                value=district
            )

        return matched

    def resolve_season(self, season: Optional[str] = None, month: Optional[int] = None) -> str:
        """Resolves season name from explicit parameter or calendar month."""
        if season:
            clean_s = season.strip().capitalize()
            # Match standard seasons
            for valid_s in ["Kharif", "Rabi", "Summer", "Perennial"]:
                if valid_s.lower() == clean_s.lower():
                    return valid_s
            raise ValidationError(
                f"Invalid season '{season}'. Must be 'Kharif', 'Rabi', 'Summer', or 'Perennial'.",
                field="season",
                value=season
            )

        if month is not None:
            try:
                m = int(month)
                if 1 <= m <= 12:
                    if m in [6, 7, 8, 9]:
                        return "Kharif"
                    elif m in [10, 11, 12, 1, 2]:
                        return "Rabi"
                    else:
                        return "Summer"
            except (ValueError, TypeError):
                pass

        # Default fallback
        return "Kharif"

    def assess_recommendation(
        self,
        prediction_result: Dict[str, Any],
        district: str,
        season: Optional[str] = None,
        month: Optional[int] = None
    ) -> Dict[str, Any]:
        """Performs regional and seasonal feasibility assessment on Top-K prediction candidates.
        
        Args:
            prediction_result: Dictionary returned by CropPredictor.predict()
            district: Target Maharashtra district
            season: 'Kharif', 'Rabi', 'Summer', or 'Perennial' (optional)
            month: Calendar month integer 1..12 (optional fallback)
            
        Returns:
            Dictionary containing unmodified ML ranking, detailed per-candidate regional assessments,
            in-season recommended candidates, flagged candidates, and scientific disclaimers.
        """
        canonical_district = self.validate_district(district)
        canonical_season = self.resolve_season(season=season, month=month)

        dist_info = self.districts[canonical_district]
        zone_id = dist_info["zone_id"]
        zone_info = self.zones[zone_id]

        raw_candidates = prediction_result.get("top_candidates", [])

        regional_assessments: List[Dict[str, Any]] = []
        in_season_recommended: List[Dict[str, Any]] = []
        flagged_candidates: List[Dict[str, Any]] = []

        for cand in raw_candidates:
            crop_name = cand["crop_name"]
            crop_key = crop_name.lower().replace(" ", "").replace("-", "")
            # Canonical key resolution
            key_map = {
                "chickpea": "chickpea", "rice": "rice", "maize": "maize",
                "cotton": "cotton", "pigeonpeas": "pigeonpeas", "blackgram": "blackgram",
                "mungbean": "mungbean", "mothbeans": "mothbeans", "pomegranate": "pomegranate",
                "grapes": "grapes", "banana": "banana", "mango": "mango",
                "orange": "orange", "papaya": "papaya", "watermelon": "watermelon",
                "muskmelon": "muskmelon", "coconut": "coconut", "lentil": "lentil",
                "coffee": "coffee", "kidneybeans": "kidneybeans", "jute": "jute", "apple": "apple"
            }
            c_key = key_map.get(crop_key, crop_key)
            profile = self.profiles.get(c_key)

            if profile is None:
                status = ViabilityStatus.INSUFFICIENT_EVIDENCE
                rationale = f"Insufficient agronomic documentation for crop '{crop_name}' in Maharashtra."
                sowing_window = "Not Documented"
                source = "N/A"
                irrigation_note = "N/A"
            else:
                sowing_window = profile.get("sowing_window_display", "N/A")
                source = profile.get("source", "N/A")
                irrigation_note = profile.get("irrigation_dependency", "N/A")

                # Rule 1: Statewide Incompatible
                if profile.get("is_statewide_incompatible", False):
                    status = ViabilityStatus.INCOMPATIBLE
                    rationale = (
                        f"Climatically Incompatible: {profile['irrigation_dependency']}. "
                        f"Zero commercial production history in Maharashtra. Source: {source}."
                    )

                # Rule 2: Micro-pocket crops outside target zone (Coffee)
                elif c_key == "coffee" and canonical_district != "Amravati":
                    status = ViabilityStatus.INCOMPATIBLE
                    rationale = (
                        f"Geographically Restricted: Coffee requires high-altitude montane shade "
                        f"(restricted strictly to Chikhaldara hill station, Amravati). "
                        f"Unsuited to the plains and plateau of {canonical_district}. Source: {source}."
                    )

                # Rule 3: Coastal plantation crops in inland plateau (Coconut)
                elif c_key == "coconut" and zone_id not in ["MH-Z1", "MH-Z2"]:
                    status = ViabilityStatus.INCOMPATIBLE
                    rationale = (
                        f"Geographically Restricted: Coconut requires tropical humid coastal climate "
                        f"(relative humidity >60%) and high water table. Unsuited to the continental "
                        f"dry air of {canonical_district} ({zone_info['name']}). Source: {source}."
                    )

                # Rule 4: Non-traditional / Experimental pulse (Kidneybeans)
                elif c_key == "kidneybeans" and canonical_district not in ["Pune", "Nashik"]:
                    status = ViabilityStatus.INSUFFICIENT_EVIDENCE
                    rationale = (
                        f"Insufficient Evidence: Kidneybeans (Rajma) are a temperate northern hill crop. "
                        f"Cultivation in {canonical_district} lacks state university extension recommendation. Source: {source}."
                    )

                # Rule 5: Seasonal Check
                else:
                    crop_seasons = profile.get("seasons", [])
                    is_perennial = "Perennial" in crop_seasons

                    if is_perennial or canonical_season in crop_seasons:
                        # District / Zone suitability check
                        suitable_zones = profile.get("suitable_zones", [])
                        prominent_districts = profile.get("prominent_districts", [])

                        if zone_id in suitable_zones or canonical_district in prominent_districts or len(suitable_zones) >= 4:
                            status = ViabilityStatus.CULTIVABLE_IN_SEASON
                            rationale = (
                                f"Certified Cultivable: Well-adapted to {zone_info['name']} in {canonical_season} season. "
                                f"Sowing window: {sowing_window}. Source: {source}."
                            )
                        else:
                            # Cultivable in season with irrigation or marginal zone
                            status = ViabilityStatus.CULTIVABLE_IN_SEASON
                            rationale = (
                                f"Cultivable with Management: Secondary zone for {crop_name} in {canonical_district}. "
                                f"{irrigation_note}. Source: {source}."
                            )
                    else:
                        # Off-Season
                        status = ViabilityStatus.CULTIVABLE_OFF_SEASON
                        valid_seasons_str = ", ".join(crop_seasons)
                        rationale = (
                            f"Seasonal Mismatch: {crop_name} is cultivable in {canonical_district}, but its certified "
                            f"season is {valid_seasons_str}. Sowing during {canonical_season} risks crop failure or disease. "
                            f"Recommended window: {sowing_window}. Source: {source}."
                        )

            # Retrieve historical DES cultivation statistics (read-only secondary data)
            hist_stats = self.des_repo.get_crop_history(crop_name, canonical_district)

            assessment_item = {
                "crop_name": crop_name,
                "ml_rank": cand["rank"],
                "model_confidence_score": cand["model_confidence_score"],
                "regional_status": status.value,
                "sowing_window": sowing_window,
                "agro_climatic_zone": f"{zone_id}: {zone_info['name']}",
                "district": canonical_district,
                "season": canonical_season,
                "rationale": rationale,
                "irrigation_note": irrigation_note,
                "historical_production_summary": hist_stats
            }

            regional_assessments.append(assessment_item)

            if status == ViabilityStatus.CULTIVABLE_IN_SEASON:
                in_season_recommended.append(assessment_item)
            else:
                flagged_candidates.append(assessment_item)

        # Generate diagnostic summary
        top_ml_crop = raw_candidates[0]["crop_name"] if raw_candidates else "Unknown"
        top_regional = in_season_recommended[0]["crop_name"] if in_season_recommended else None

        if top_regional == top_ml_crop:
            summary = (
                f"Consensus: The ML Top-1 recommendation (**{top_ml_crop}**) is certified as "
                f"**Cultivable & In-Season** for {canonical_district} district during {canonical_season} season."
            )
        elif top_regional is not None:
            summary = (
                f"Regional Adjustment: The ML Top-1 match (**{top_ml_crop}**) was flagged or restricted. "
                f"The highest-ranked viable candidate for {canonical_district} in {canonical_season} is **{top_regional}**."
            )
        else:
            summary = (
                f"All Top-5 ML candidate crops were flagged as off-season or incompatible for "
                f"{canonical_district} in {canonical_season}. Explore alternative cropping plans."
            )

        # Evaluate IMD rainfall departure if input rainfall is present
        climatology_eval = None
        obs_rain = None
        if "inputs_validated" in prediction_result and "rainfall" in prediction_result["inputs_validated"]:
            obs_rain = prediction_result["inputs_validated"]["rainfall"]["value"]
        elif "rainfall" in prediction_result:
            obs_rain = prediction_result["rainfall"]

        if obs_rain is not None:
            try:
                climatology_eval = self.climatology_repo.evaluate_rainfall_departure(
                    canonical_district,
                    observed_rainfall_mm=float(obs_rain),
                    season=canonical_season
                )
            except Exception as cr_err:
                print(f"Warning: IMD climatology departure evaluation failed: {cr_err}")

        return {
            "status": "success",
            "district": canonical_district,
            "season": canonical_season,
            "agro_climatic_zone": {
                "zone_id": zone_id,
                "zone_name": zone_info["name"],
                "rainfall_range_mm": zone_info["rainfall_range_mm"],
                "dominant_soils": zone_info["dominant_soils"],
                "university": zone_info["university"],
                "annual_normal_rainfall_mm": dist_info["annual_rainfall_normal_mm"]
            },
            "climatology_comparison": climatology_eval,
            "unfiltered_ml_ranking": raw_candidates,
            "regional_assessments": regional_assessments,
            "in_season_recommended": in_season_recommended,
            "flagged_candidates": flagged_candidates,
            "diagnostic_summary": summary,
            "disclaimer": (
                "Regional Notice: Regional and seasonal assessments are compiled from the Department of Agriculture, "
                "Government of Maharashtra and State Agricultural Universities (MPKV, VNMKV, PDKV, BSKKV). "
                "They reflect regional zoning and historical production patterns, not a real-time guarantee of individual farm yields."
            )
        }
