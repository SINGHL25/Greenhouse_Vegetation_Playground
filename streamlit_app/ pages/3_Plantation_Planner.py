
# pages/3_Plantation_Planner.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import calendar

# Page configuration
st.set_page_config(
    page_title="Plantation Planner",
    page_icon="🗺️",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .parcel-card {
        background: linear-gradient(135deg, #56ab2f 0%, #a8e6cf 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .planning-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #4CAF50;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    
    .crop-rotation-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 8px;
        color: white;
        margin: 0.5rem 0;
    }
    
    .season-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: bold;
        margin: 0.25rem;
    }
    
    .season-spring { background-color: #98FB98; color: #2F4F2F; }
    .season-summer { background-color: #FFD700; color: #8B4513; }
    .season-autumn { background-color: #DEB887; color: #8B4513; }
    .season-winter { background-color: #ADD8E6; color: #2F4F4F; }
    
    .status-badge {
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: bold;
        margin: 0.25rem;
    }
    
    .status-planned { background-color: #E3F2FD; color: #1976D2; }
    .status-planted { background-color: #E8F5E8; color: #2E7D32; }
    .status-growing { background-color: #FFF3E0; color: #F57C00; }
    .status-harvesting { background-color: #F3E5F5; color: #7B1FA2; }
    .status-completed { background-color: #E0E0E0; color: #424242; }
    
    .calendar-cell {
        background: white;
        border: 1px solid #e0e0e0;
        padding: 0.5rem;
        min-height: 60px;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

class PlantationPlanner:
    def __init__(self):
        self.load_data()
        self.init_crop_database()
        self.init_rotation_rules()
    
    def load_data(self):
        """Load plantation planning data"""
        if 'plantation_data' not in st.session_state:
            self.generate_sample_data()
    
    def init_crop_database(self):
        """Initialize crop database with characteristics"""
        self.crop_database = {
            'Tomato': {
                'planting_season': ['Spring', 'Summer'],
                'growth_days': 75,
                'yield_per_hectare': 50000,  # kg
                'plant_spacing': 0.6,  # meters
                'soil_ph_range': (6.0, 6.8),
                'water_needs': 'High',
                'fertilizer_needs': 'High',
                'companion_plants': ['Basil', 'Lettuce', 'Peppers'],
                'rotation_family': 'Nightshade',
                'harvest_duration': 60,  # days
                'market_price_kg': 3.5,  # USD per kg
                'icon': '🍅'
            },
            'Lettuce': {
                'planting_season': ['Spring', 'Autumn', 'Winter'],
                'growth_days': 45,
                'yield_per_hectare': 25000,
                'plant_spacing': 0.3,
                'soil_ph_range': (6.0, 7.0),
                'water_needs': 'Medium',
                'fertilizer_needs': 'Medium',
                'companion_plants': ['Tomato', 'Carrots', 'Onions'],
                'rotation_family': 'Leafy',
                'harvest_duration': 21,
                'market_price_kg': 4.2,
                'icon': '🥬'
            },
            'Pepper': {
                'planting_season': ['Spring', 'Summer'],
                'growth_days': 80,
                'yield_per_hectare': 30000,
                'plant_spacing': 0.5,
                'soil_ph_range': (6.2, 7.0),
                'water_needs': 'High',
                'fertilizer_needs': 'High',
                'companion_plants': ['Tomato', 'Basil', 'Onions'],
                'rotation_family': 'Nightshade',
                'harvest_duration': 90,
                'market_price_kg': 4.8,
                'icon': '🌶️'
            },
            'Cucumber': {
                'planting_season': ['Spring', 'Summer'],
                'growth_days': 60,
                'yield_per_hectare': 40000,
                'plant_spacing': 0.4,
                'soil_ph_range': (6.0, 7.0),
                'water_needs': 'High',
                'fertilizer_needs': 'Medium',
                'companion_plants': ['Lettuce', 'Radishes', 'Beans'],
                'rotation_family': 'Cucurbit',
                'harvest_duration': 45,
                'market_price_kg': 2.8,
                'icon': '🥒'
            },
            'Basil': {
                'planting_season': ['Spring', 'Summer'],
                'growth_days': 70,
                'yield_per_hectare': 8000,
                'plant_spacing': 0.25,
                'soil_ph_range': (6.0, 7.5),
                'water_needs': 'Medium',
                'fertilizer_needs': 'Low',
                'companion_plants': ['Tomato', 'Peppers'],
                'rotation_family': 'Herb',
                'harvest_duration': 120,
                'market_price_kg': 12.0,
                'icon': '🌿'
            },
            'Carrots': {
                'planting_season': ['Spring',
