"""Maharashtra regional agricultural data, agro-climatic zones, and 22-crop profiles.

References:
- Department of Agriculture, Government of Maharashtra (krishi.maharashtra.gov.in)
- State Agricultural Universities: MPKV Rahuri, VNMKV Parbhani, Dr. PDKV Akola, Dr. BSKKV Dapoli
- Directorate of Economics & Statistics (DES / APS, Ministry of Agriculture)
- India Meteorological Department (IMD Pune) District Climatology (1991-2020)
- ICAR-NIASM Baramati & ICAR institutes (CCRI Nagpur, NRCP Solapur, NRCG Pune, CICR Nagpur, CITH Srinagar)
"""

from enum import Enum
from typing import Dict, Any, List


class ViabilityStatus(str, Enum):
    """The four mutually exclusive regional/seasonal viability statuses."""
    CULTIVABLE_IN_SEASON = "Cultivable & In-Season"
    CULTIVABLE_OFF_SEASON = "Cultivable but Off-Season"
    INCOMPATIBLE = "Agronomically Incompatible / Climatically Unsuited"
    INSUFFICIENT_EVIDENCE = "Insufficient Evidence / Not Assessed"


# Standard cropping seasons in Maharashtra
SEASONS = {
    "Kharif": {
        "name": "Kharif (Monsoon)",
        "months": [6, 7, 8, 9, 10],
        "description": "Monsoon cropping season (June to October)"
    },
    "Rabi": {
        "name": "Rabi (Winter/Post-Monsoon)",
        "months": [10, 11, 12, 1, 2, 3],
        "description": "Post-monsoon winter cropping season (October to March)"
    },
    "Summer": {
        "name": "Summer (Zaid/Hot Weather)",
        "months": [1, 2, 3, 4, 5],
        "description": "Irrigated summer season (January to May)"
    },
    "Perennial": {
        "name": "Perennial / Multi-Season",
        "months": list(range(1, 13)),
        "description": "Perennial orchard or year-round plantation crop"
    }
}


# The 9 official Agro-Climatic Zones of Maharashtra
AGRO_CLIMATIC_ZONES: Dict[str, Dict[str, Any]] = {
    "MH-Z1": {
        "name": "South Konkan Coastal Zone",
        "rainfall_range_mm": [3000, 3500],
        "dominant_soils": ["Laterite", "Coastal Alluvium"],
        "primary_districts": ["Ratnagiri", "Sindhudurg"],
        "university": "Dr. BSKKV, Dapoli"
    },
    "MH-Z2": {
        "name": "North Konkan Coastal Zone",
        "rainfall_range_mm": [2500, 3000],
        "dominant_soils": ["Non-lateritic Coastal Alluvium", "Coarse Shallow"],
        "primary_districts": ["Thane", "Palghar", "Raigad", "Mumbai Suburban", "Mumbai City"],
        "university": "Dr. BSKKV, Dapoli"
    },
    "MH-Z3": {
        "name": "Western Ghat Zone",
        "rainfall_range_mm": [3000, 5000],
        "dominant_soils": ["Lateritic Red Hills", "Forest Loam"],
        "primary_districts": ["Ghat portions of Kolhapur, Sangli, Satara, Pune, Nashik"],
        "university": "MPKV, Rahuri"
    },
    "MH-Z4": {
        "name": "Transition Zone 1 (Sub-Montane)",
        "rainfall_range_mm": [1250, 2500],
        "dominant_soils": ["Reddish Brown Clayey", "Medium Black"],
        "primary_districts": ["Kolhapur", "Satara", "Sangli", "Western Pune"],
        "university": "MPKV, Rahuri"
    },
    "MH-Z5": {
        "name": "Transition Zone 2 (Plain Zone)",
        "rainfall_range_mm": [700, 1250],
        "dominant_soils": ["Medium Black (Vertic)", "Clay Loam"],
        "primary_districts": ["Pune", "Ahmednagar", "Satara"],
        "university": "MPKV, Rahuri"
    },
    "MH-Z6": {
        "name": "Scarcity Zone",
        "rainfall_range_mm": [500, 700],
        "dominant_soils": ["Shallow Calcareous", "Medium Black", "Light Black"],
        "primary_districts": ["Solapur", "Eastern Ahmednagar", "Eastern Pune", "Dhule", "Eastern Sangli"],
        "university": "MPKV, Rahuri"
    },
    "MH-Z7": {
        "name": "Assured Rainfall Zone (Marathwada & Khandesh)",
        "rainfall_range_mm": [700, 900],
        "dominant_soils": ["Deep Black (Vertisols)", "Clayey"],
        "primary_districts": [
            "Chhatrapati Sambhajinagar", "Jalna", "Beed", "Parbhani",
            "Latur", "Dharashiv", "Nanded", "Hingoli", "Jalgaon"
        ],
        "university": "VNMKV, Parbhani & MPKV, Rahuri"
    },
    "MH-Z8": {
        "name": "Moderate Rainfall Zone (Western Vidarbha)",
        "rainfall_range_mm": [900, 1250],
        "dominant_soils": ["Deep Black Clayey (Vertisols)"],
        "primary_districts": ["Buldhana", "Akola", "Washim", "Amravati", "Yavatmal", "Wardha"],
        "university": "Dr. PDKV, Akola"
    },
    "MH-Z9": {
        "name": "High Rainfall Zone (Eastern Vidarbha)",
        "rainfall_range_mm": [1250, 1700],
        "dominant_soils": ["Yellowish Brown Clay Loam", "Sandy Clay Loam (Wainganga Basin)"],
        "primary_districts": ["Nagpur", "Bhandara", "Gondia", "Chandrapur", "Gadchiroli"],
        "university": "Dr. PDKV, Akola"
    }
}


# The 36 Districts of Maharashtra with official Zone assignment and IMD annual normals
MAHARASHTRA_DISTRICTS: Dict[str, Dict[str, Any]] = {
    "Ahmednagar": {"zone_id": "MH-Z6", "division": "Nashik", "annual_rainfall_normal_mm": 579.5},
    "Akola": {"zone_id": "MH-Z8", "division": "Amravati", "annual_rainfall_normal_mm": 776.4},
    "Amravati": {"zone_id": "MH-Z8", "division": "Amravati", "annual_rainfall_normal_mm": 863.0},
    "Chhatrapati Sambhajinagar": {"zone_id": "MH-Z7", "division": "Aurangabad", "annual_rainfall_normal_mm": 718.3},
    "Beed": {"zone_id": "MH-Z7", "division": "Aurangabad", "annual_rainfall_normal_mm": 666.1},
    "Bhandara": {"zone_id": "MH-Z9", "division": "Nagpur", "annual_rainfall_normal_mm": 1330.2},
    "Buldhana": {"zone_id": "MH-Z8", "division": "Amravati", "annual_rainfall_normal_mm": 743.0},
    "Chandrapur": {"zone_id": "MH-Z9", "division": "Nagpur", "annual_rainfall_normal_mm": 1249.4},
    "Dhule": {"zone_id": "MH-Z6", "division": "Nashik", "annual_rainfall_normal_mm": 612.0},
    "Gadchiroli": {"zone_id": "MH-Z9", "division": "Nagpur", "annual_rainfall_normal_mm": 1424.0},
    "Gondia": {"zone_id": "MH-Z9", "division": "Nagpur", "annual_rainfall_normal_mm": 1410.0},
    "Hingoli": {"zone_id": "MH-Z7", "division": "Aurangabad", "annual_rainfall_normal_mm": 890.5},
    "Jalgaon": {"zone_id": "MH-Z7", "division": "Nashik", "annual_rainfall_normal_mm": 720.0},
    "Jalna": {"zone_id": "MH-Z7", "division": "Aurangabad", "annual_rainfall_normal_mm": 710.0},
    "Kolhapur": {"zone_id": "MH-Z4", "division": "Pune", "annual_rainfall_normal_mm": 1780.0},
    "Latur": {"zone_id": "MH-Z7", "division": "Aurangabad", "annual_rainfall_normal_mm": 790.0},
    "Mumbai City": {"zone_id": "MH-Z2", "division": "Konkan", "annual_rainfall_normal_mm": 2420.0},
    "Mumbai Suburban": {"zone_id": "MH-Z2", "division": "Konkan", "annual_rainfall_normal_mm": 2420.0},
    "Nagpur": {"zone_id": "MH-Z9", "division": "Nagpur", "annual_rainfall_normal_mm": 1160.0},
    "Nanded": {"zone_id": "MH-Z7", "division": "Aurangabad", "annual_rainfall_normal_mm": 895.0},
    "Nandurbar": {"zone_id": "MH-Z7", "division": "Nashik", "annual_rainfall_normal_mm": 860.0},
    "Nashik": {"zone_id": "MH-Z5", "division": "Nashik", "annual_rainfall_normal_mm": 810.0},
    "Dharashiv": {"zone_id": "MH-Z7", "division": "Aurangabad", "annual_rainfall_normal_mm": 730.0},
    "Palghar": {"zone_id": "MH-Z2", "division": "Konkan", "annual_rainfall_normal_mm": 2600.0},
    "Parbhani": {"zone_id": "MH-Z7", "division": "Aurangabad", "annual_rainfall_normal_mm": 825.0},
    "Pune": {"zone_id": "MH-Z5", "division": "Pune", "annual_rainfall_normal_mm": 750.0},
    "Raigad": {"zone_id": "MH-Z2", "division": "Konkan", "annual_rainfall_normal_mm": 3100.0},
    "Ratnagiri": {"zone_id": "MH-Z1", "division": "Konkan", "annual_rainfall_normal_mm": 3360.0},
    "Sangli": {"zone_id": "MH-Z5", "division": "Pune", "annual_rainfall_normal_mm": 640.0},
    "Satara": {"zone_id": "MH-Z4", "division": "Pune", "annual_rainfall_normal_mm": 920.0},
    "Sindhudurg": {"zone_id": "MH-Z1", "division": "Konkan", "annual_rainfall_normal_mm": 3480.0},
    "Solapur": {"zone_id": "MH-Z6", "division": "Pune", "annual_rainfall_normal_mm": 560.0},
    "Thane": {"zone_id": "MH-Z2", "division": "Konkan", "annual_rainfall_normal_mm": 2550.0},
    "Wardha": {"zone_id": "MH-Z8", "division": "Nagpur", "annual_rainfall_normal_mm": 1050.0},
    "Washim": {"zone_id": "MH-Z8", "division": "Amravati", "annual_rainfall_normal_mm": 915.0},
    "Yavatmal": {"zone_id": "MH-Z8", "division": "Amravati", "annual_rainfall_normal_mm": 990.0}
}


# Comprehensive Reconciled 22-Crop Regional Agronomic Profiles for Maharashtra
CROP_REGIONAL_PROFILES: Dict[str, Dict[str, Any]] = {
    "rice": {
        "crop_name": "Rice",
        "scientific_name": "Oryza sativa",
        "is_statewide_incompatible": False,
        "seasons": ["Kharif"],
        "sowing_start_month": 6,
        "sowing_end_month": 7,
        "sowing_window_display": "June 15 – July 15 (Transplanting: July 10 – August 5)",
        "harvest_window_display": "October 15 – November 25",
        "suitable_zones": ["MH-Z1", "MH-Z2", "MH-Z3", "MH-Z4", "MH-Z9"],
        "prominent_districts": ["Ratnagiri", "Sindhudurg", "Raigad", "Thane", "Palghar", "Bhandara", "Gondia", "Chandrapur", "Gadchiroli", "Kolhapur"],
        "water_requirement_mm": [1100, 1400],
        "irrigation_dependency": "Rainfed in high rainfall zones; requires ponded water/assured irrigation elsewhere",
        "source": "Dr. BSKKV Dapoli / MPKV Rahuri Rice Package of Practices; DES Kharif Statistics"
    },
    "maize": {
        "crop_name": "Maize",
        "scientific_name": "Zea mays",
        "is_statewide_incompatible": False,
        "seasons": ["Kharif", "Rabi", "Summer"],
        "sowing_start_month": 6,
        "sowing_end_month": 2,  # Multi-season
        "sowing_window_display": "Kharif: June 20 – July 15; Rabi: Oct 15 – Nov 15; Summer: Jan 15 – Feb 15",
        "harvest_window_display": "Kharif: Sept–Oct; Rabi: Feb–March; Summer: April–May",
        "suitable_zones": ["MH-Z4", "MH-Z5", "MH-Z6", "MH-Z7", "MH-Z8"],
        "prominent_districts": ["Nashik", "Kolhapur", "Sangli", "Chhatrapati Sambhajinagar", "Jalgaon", "Ahmednagar", "Pune"],
        "water_requirement_mm": [500, 700],
        "irrigation_dependency": "Rainfed in Kharif; requires 4-6 irrigations in Rabi/Summer",
        "source": "MPKV Rahuri Maize Agronomy Guidelines"
    },
    "chickpea": {
        "crop_name": "Chickpea",
        "scientific_name": "Cicer arietinum",
        "is_statewide_incompatible": False,
        "seasons": ["Rabi"],
        "sowing_start_month": 10,
        "sowing_end_month": 11,
        "sowing_window_display": "October 10 – November 10 (Post-monsoon)",
        "harvest_window_display": "February 1 – February 28",
        "suitable_zones": ["MH-Z5", "MH-Z6", "MH-Z7", "MH-Z8"],
        "prominent_districts": ["Latur", "Parbhani", "Beed", "Dharashiv", "Solapur", "Ahmednagar", "Pune", "Akola", "Amravati"],
        "water_requirement_mm": [250, 350],
        "irrigation_dependency": "Grown predominantly on residual soil moisture in Vertisols; 1-2 protective irrigations",
        "source": "ICAR-IIPR & MPKV Rahuri Gram Research Program"
    },
    "pigeonpeas": {
        "crop_name": "Pigeonpeas",
        "scientific_name": "Cajanus cajan",
        "is_statewide_incompatible": False,
        "seasons": ["Kharif"],
        "sowing_start_month": 6,
        "sowing_end_month": 7,
        "sowing_window_display": "June 20 – July 15",
        "harvest_window_display": "December 15 – January 31 (Long duration: 150-180 days)",
        "suitable_zones": ["MH-Z6", "MH-Z7", "MH-Z8"],
        "prominent_districts": ["Latur", "Parbhani", "Nanded", "Hingoli", "Amravati", "Akola", "Yavatmal", "Wardha", "Jalgaon"],
        "water_requirement_mm": [600, 750],
        "irrigation_dependency": "Deep root system enables rainfed drought resilience; protective irrigation at flowering/pod set",
        "source": "VNMKV Parbhani Tur Package of Practices"
    },
    "cotton": {
        "crop_name": "Cotton",
        "scientific_name": "Gossypium hirsutum / arboreum",
        "is_statewide_incompatible": False,
        "seasons": ["Kharif"],
        "sowing_start_month": 6,
        "sowing_end_month": 7,
        "sowing_window_display": "June 15 – July 10 (Monsoon onset); Irrigated: May 15 – June 1",
        "harvest_window_display": "November – January (Multiple pickings)",
        "suitable_zones": ["MH-Z6", "MH-Z7", "MH-Z8"],
        "prominent_districts": ["Yavatmal", "Akola", "Amravati", "Wardha", "Buldhana", "Nanded", "Jalna", "Parbhani", "Jalgaon", "Dhule"],
        "water_requirement_mm": [700, 900],
        "irrigation_dependency": "Predominantly rainfed in deep black cotton soils (Vertisols); drip irrigation in Khandesh",
        "source": "ICAR-CICR Nagpur / Dr. PDKV Akola Cotton Research"
    },
    "blackgram": {
        "crop_name": "Blackgram",
        "scientific_name": "Vigna mungo",
        "is_statewide_incompatible": False,
        "seasons": ["Kharif"],
        "sowing_start_month": 6,
        "sowing_end_month": 7,
        "sowing_window_display": "June 20 – July 10",
        "harvest_window_display": "August 25 – September 15 (Short duration: 70-80 days)",
        "suitable_zones": ["MH-Z4", "MH-Z5", "MH-Z7", "MH-Z8"],
        "prominent_districts": ["Latur", "Nanded", "Dharashiv", "Yavatmal", "Wardha", "Akola", "Kolhapur"],
        "water_requirement_mm": [350, 450],
        "irrigation_dependency": "Rainfed pulse",
        "source": "VNMKV Parbhani & Dr. PDKV Akola Pulse Guidelines"
    },
    "mungbean": {
        "crop_name": "Mungbean",
        "scientific_name": "Vigna radiata",
        "is_statewide_incompatible": False,
        "seasons": ["Kharif"],
        "sowing_start_month": 6,
        "sowing_end_month": 7,
        "sowing_window_display": "June 15 – July 5",
        "harvest_window_display": "August 15 – September 5 (Short duration: 60-70 days)",
        "suitable_zones": ["MH-Z5", "MH-Z6", "MH-Z7", "MH-Z8"],
        "prominent_districts": ["Jalna", "Parbhani", "Nanded", "Amravati", "Akola", "Buldhana", "Jalgaon", "Dhule"],
        "water_requirement_mm": [300, 400],
        "irrigation_dependency": "Rainfed early Kharif pulse",
        "source": "VNMKV Parbhani Pulse Research"
    },
    "mothbeans": {
        "crop_name": "Mothbeans",
        "scientific_name": "Vigna aconitifolia",
        "is_statewide_incompatible": False,
        "seasons": ["Kharif"],
        "sowing_start_month": 6,
        "sowing_end_month": 7,
        "sowing_window_display": "June 25 – July 20",
        "harvest_window_display": "October 1 – October 20",
        "suitable_zones": ["MH-Z6"],
        "prominent_districts": ["Ahmednagar", "Solapur", "Dhule", "Jalgaon", "Sangli", "Pune (Eastern)"],
        "water_requirement_mm": [200, 300],
        "irrigation_dependency": "High drought tolerance; thrives on light gravelly marginal soils in rain deficit tracts",
        "source": "MPKV Rahuri Scarcity Zone Pulse Division"
    },
    "pomegranate": {
        "crop_name": "Pomegranate",
        "scientific_name": "Punica granatum",
        "is_statewide_incompatible": False,
        "seasons": ["Perennial"],
        "sowing_start_month": 1,
        "sowing_end_month": 12,
        "sowing_window_display": "Orchard Planting: July–August. Regulated via Bahar treatment: Ambe (Jan), Mrig (June), Hasta (Sept)",
        "harvest_window_display": "5-6 months following flowering of respective Bahar",
        "suitable_zones": ["MH-Z5", "MH-Z6", "MH-Z7"],
        "prominent_districts": ["Solapur", "Nashik", "Ahmednagar", "Sangli", "Dharashiv", "Pune"],
        "water_requirement_mm": [600, 800],
        "irrigation_dependency": "Requires drip irrigation and controlled water stress for Bahar induction; well-drained soil",
        "source": "ICAR-National Research Centre on Pomegranate (NRCP), Solapur"
    },
    "grapes": {
        "crop_name": "Grapes",
        "scientific_name": "Vitis vinifera",
        "is_statewide_incompatible": False,
        "seasons": ["Perennial"],
        "sowing_start_month": 1,
        "sowing_end_month": 12,
        "sowing_window_display": "Vineyard Planting: Feb–March. Pruning cycles: Back pruning (April), Forward/Fruit pruning (October)",
        "harvest_window_display": "January – March (Following forward pruning)",
        "suitable_zones": ["MH-Z5", "MH-Z6"],
        "prominent_districts": ["Nashik", "Sangli", "Solapur", "Pune", "Ahmednagar"],
        "water_requirement_mm": [600, 750],
        "irrigation_dependency": "Drip irrigation mandatory; requires dry sunny weather during veraison/ripening",
        "source": "ICAR-National Research Centre for Grapes (NRCG), Pune"
    },
    "banana": {
        "crop_name": "Banana",
        "scientific_name": "Musa paradisiaca",
        "is_statewide_incompatible": False,
        "seasons": ["Perennial"],
        "sowing_start_month": 6,
        "sowing_end_month": 11,
        "sowing_window_display": "Mrig planting: June 15 – July 15; Kandebar planting: October – November",
        "harvest_window_display": "10-12 months following sucker/tissue culture plantation",
        "suitable_zones": ["MH-Z6", "MH-Z7"],
        "prominent_districts": ["Jalgaon", "Nanded", "Parbhani", "Solapur", "Hingoli"],
        "water_requirement_mm": [1800, 2200],
        "irrigation_dependency": "High water-intensity crop; assured perennial irrigation (drip) essential",
        "source": "MPKV Rahuri & Jain R&D Jalgaon Extension Guidelines"
    },
    "mango": {
        "crop_name": "Mango",
        "scientific_name": "Mangifera indica",
        "is_statewide_incompatible": False,
        "seasons": ["Perennial"],
        "sowing_start_month": 6,
        "sowing_end_month": 8,
        "sowing_window_display": "Orchard Plantation: July – August (Monsoon); Flowering: Dec – Jan",
        "harvest_window_display": "March – May",
        "suitable_zones": ["MH-Z1", "MH-Z2", "MH-Z7"],
        "prominent_districts": ["Ratnagiri", "Sindhudurg", "Raigad", "Chhatrapati Sambhajinagar", "Jalna", "Beed"],
        "water_requirement_mm": [700, 1000],
        "irrigation_dependency": "Requires protective irrigation in dryland Kesar tract; dry rain-free spell essential during flowering",
        "source": "Dr. BSKKV Dapoli (Alphonso) / VNMKV Parbhani (Kesar)"
    },
    "orange": {
        "crop_name": "Orange",
        "scientific_name": "Citrus reticulata",
        "is_statewide_incompatible": False,
        "seasons": ["Perennial"],
        "sowing_start_month": 6,
        "sowing_end_month": 8,
        "sowing_window_display": "Planting: July – August; Ambia Bahar flowering (Jan–Feb), Mrig Bahar flowering (June–July)",
        "harvest_window_display": "Ambia crop: Nov–Dec; Mrig crop: Feb–March",
        "suitable_zones": ["MH-Z8", "MH-Z9"],
        "prominent_districts": ["Nagpur", "Amravati", "Wardha"],
        "water_requirement_mm": [900, 1100],
        "irrigation_dependency": "Drip irrigation; deep black soils with good drainage to prevent collar rot",
        "source": "ICAR-Central Citrus Research Institute (CCRI), Nagpur"
    },
    "papaya": {
        "crop_name": "Papaya",
        "scientific_name": "Carica papaya",
        "is_statewide_incompatible": False,
        "seasons": ["Perennial"],
        "sowing_start_month": 6,
        "sowing_end_month": 10,
        "sowing_window_display": "Monsoon: June – July; Autumn: September – October; Spring: Feb – March",
        "harvest_window_display": "Starts 8-10 months following transplanting",
        "suitable_zones": ["MH-Z5", "MH-Z6", "MH-Z7"],
        "prominent_districts": ["Nandurbar", "Jalgaon", "Pune", "Solapur", "Ahmednagar"],
        "water_requirement_mm": [1000, 1300],
        "irrigation_dependency": "Regular irrigation; highly sensitive to waterlogging and collar rot",
        "source": "MPKV Rahuri Horticulture Department"
    },
    "watermelon": {
        "crop_name": "Watermelon",
        "scientific_name": "Citrullus lanatus",
        "is_statewide_incompatible": False,
        "seasons": ["Summer"],
        "sowing_start_month": 1,
        "sowing_end_month": 2,
        "sowing_window_display": "January 15 – February 28",
        "harvest_window_display": "March 20 – May 15",
        "suitable_zones": ["MH-Z1", "MH-Z2", "MH-Z5", "MH-Z6", "MH-Z7"],
        "prominent_districts": ["Solapur", "Pune", "Ahmednagar", "Jalgaon", "Raigad", "Ratnagiri"],
        "water_requirement_mm": [400, 600],
        "irrigation_dependency": "Assured irrigation / drip / riverbed alluvial cultivation required",
        "source": "MPKV Rahuri Vegetable Research"
    },
    "muskmelon": {
        "crop_name": "Muskmelon",
        "scientific_name": "Cucumis melo",
        "is_statewide_incompatible": False,
        "seasons": ["Summer"],
        "sowing_start_month": 1,
        "sowing_end_month": 2,
        "sowing_window_display": "January 15 – February 28",
        "harvest_window_display": "April 1 – May 10",
        "suitable_zones": ["MH-Z5", "MH-Z6", "MH-Z7"],
        "prominent_districts": ["Pune", "Solapur", "Ahmednagar", "Jalgaon", "Dhule"],
        "water_requirement_mm": [400, 550],
        "irrigation_dependency": "Irrigated cucurbit; dry atmosphere required during fruit ripening",
        "source": "MPKV Rahuri Vegetable Research"
    },
    "coconut": {
        "crop_name": "Coconut",
        "scientific_name": "Cocos nucifera",
        "is_statewide_incompatible": False,
        "seasons": ["Perennial"],
        "sowing_start_month": 6,
        "sowing_end_month": 8,
        "sowing_window_display": "Planting: June – August (Monsoon)",
        "harvest_window_display": "Perennial (Harvesting every 45-60 days once mature)",
        "suitable_zones": ["MH-Z1", "MH-Z2"],
        "prominent_districts": ["Sindhudurg", "Ratnagiri", "Raigad", "Palghar", "Thane"],
        "water_requirement_mm": [1300, 1600],
        "irrigation_dependency": "Requires humid coastal tropical climate (>60% humidity) and high water table; unsuited to dry inland plateau",
        "source": "Dr. BSKKV Dapoli Regional Fruit Research Station, Vengurla"
    },
    "lentil": {
        "crop_name": "Lentil",
        "scientific_name": "Lens culinaris",
        "is_statewide_incompatible": False,
        "seasons": ["Rabi"],
        "sowing_start_month": 10,
        "sowing_end_month": 11,
        "sowing_window_display": "October 15 – November 15",
        "harvest_window_display": "February 1 – February 25",
        "suitable_zones": ["MH-Z9"],
        "prominent_districts": ["Nagpur", "Wardha", "Bhandara", "Chandrapur"],
        "water_requirement_mm": [250, 350],
        "irrigation_dependency": "Minor winter pulse; grown on residual soil moisture after Kharif paddy in Eastern Vidarbha",
        "source": "Dr. PDKV Akola & DES Maharashtra Production Records"
    },
    "coffee": {
        "crop_name": "Coffee",
        "scientific_name": "Coffea arabica / canephora",
        "is_statewide_incompatible": False,
        "seasons": ["Perennial"],
        "sowing_start_month": 6,
        "sowing_end_month": 8,
        "sowing_window_display": "Monsoon planting; shade-grown",
        "harvest_window_display": "November – January",
        "suitable_zones": ["MH-Z8 (Restricted Micro-Pocket)"],
        "prominent_districts": ["Amravati (Chikhaldara Hill Station only, ~1100m MSL, <100 ha)"],
        "water_requirement_mm": [1500, 2200],
        "irrigation_dependency": "High altitude montane mist, shade trees, and deep organic forest soil required; unsuited to commercial plateau cultivation",
        "source": "Coffee Board of India / Dr. PDKV Akola"
    },
    "kidneybeans": {
        "crop_name": "Kidneybeans",
        "scientific_name": "Phaseolus vulgaris",
        "is_statewide_incompatible": False,
        "seasons": ["Rabi"],
        "sowing_start_month": 10,
        "sowing_end_month": 11,
        "sowing_window_display": "October 15 – November 10",
        "harvest_window_display": "January 20 – February 15",
        "suitable_zones": ["MH-Z5 (Limited / Experimental)"],
        "prominent_districts": ["Pune (Khed/Ambegaon pockets)", "Nashik"],
        "water_requirement_mm": [350, 450],
        "irrigation_dependency": "Non-traditional pulse in Maharashtra; requires cold weather and assured irrigation; not part of standard statewide extension",
        "source": "MPKV Rahuri Extension Literature / Experimental Reports"
    },
    "jute": {
        "crop_name": "Jute",
        "scientific_name": "Corchorus capsularis / olitorius",
        "is_statewide_incompatible": True,
        "seasons": [],
        "sowing_start_month": None,
        "sowing_end_month": None,
        "sowing_window_display": "N/A (Not cultivated commercially in Maharashtra)",
        "harvest_window_display": "N/A",
        "suitable_zones": [],
        "prominent_districts": [],
        "water_requirement_mm": [1500, 2000],
        "irrigation_dependency": "Requires deltaic slow-moving floodwaters (>34°C) for 15-20 days of stem retting; Maharashtra lacks deltaic hydrology and retting infrastructure",
        "source": "Directorate of Jute Development, Ministry of Agriculture, Kolkata / DES Area Statistics (Zero acreage)"
    },
    "apple": {
        "crop_name": "Apple",
        "scientific_name": "Malus domestica",
        "is_statewide_incompatible": True,
        "seasons": [],
        "sowing_start_month": None,
        "sowing_end_month": None,
        "sowing_window_display": "N/A (Climatically unsuited for commercial production in Maharashtra)",
        "harvest_window_display": "N/A",
        "suitable_zones": [],
        "prominent_districts": [],
        "water_requirement_mm": [800, 1100],
        "irrigation_dependency": "Commercial cultivars require 800-1200 winter chilling hours (<7°C, per ICAR-CITH) for floral bud break; absent in tropical peninsular Maharashtra",
        "source": "ICAR-Central Institute of Temperate Horticulture (CITH), Srinagar"
    }
}
