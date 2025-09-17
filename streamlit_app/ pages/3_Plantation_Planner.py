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
                'planting_season': ['Spring', 'Autumn'],
                'growth_days': 90,
                'yield_per_hectare': 35000,
                'plant_spacing': 0.1,
                'soil_ph_range': (6.0, 6.8),
                'water_needs': 'Medium',
                'fertilizer_needs': 'Low',
                'companion_plants': ['Lettuce', 'Onions', 'Tomato'],
                'rotation_family': 'Root',
                'harvest_duration': 30,
                'market_price_kg': 2.1,
                'icon': '🥕'
            },
            'Spinach': {
                'planting_season': ['Spring', 'Autumn', 'Winter'],
                'growth_days': 40,
                'yield_per_hectare': 20000,
                'plant_spacing': 0.2,
                'soil_ph_range': (6.5, 7.5),
                'water_needs': 'Medium',
                'fertilizer_needs': 'Medium',
                'companion_plants': ['Lettuce', 'Carrots', 'Onions'],
                'rotation_family': 'Leafy',
                'harvest_duration': 21,
                'market_price_kg': 5.5,
                'icon': '🥬'
            }
        }
    
    def init_rotation_rules(self):
        """Initialize crop rotation rules and compatibility"""
        self.rotation_rules = {
            'Nightshade': {
                'avoid_after': ['Nightshade'],
                'good_after': ['Leafy', 'Root', 'Legume'],
                'rest_period': 2  # years
            },
            'Leafy': {
                'avoid_after': [],
                'good_after': ['Nightshade', 'Cucurbit', 'Root'],
                'rest_period': 1
            },
            'Cucurbit': {
                'avoid_after': ['Cucurbit'],
                'good_after': ['Leafy', 'Root', 'Legume'],
                'rest_period': 2
            },
            'Root': {
                'avoid_after': [],
                'good_after': ['Nightshade', 'Leafy'],
                'rest_period': 1
            },
            'Herb': {
                'avoid_after': [],
                'good_after': ['Nightshade', 'Leafy', 'Root', 'Cucurbit'],
                'rest_period': 0
            },
            'Legume': {
                'avoid_after': [],
                'good_after': ['Nightshade', 'Cucurbit'],
                'rest_period': 1
            }
        }
    
    def generate_sample_data(self):
        """Generate sample plantation data"""
        parcels = []
        planning_data = []
        
        # Generate parcels
        for i in range(1, 13):  # 12 parcels
            parcel_id = f'P{i:03d}'
            area = np.random.uniform(1.5, 5.0)
            soil_type = np.random.choice(['Loam', 'Clay Loam', 'Sandy Loam', 'Silt Loam'])
            
            parcel = {
                'parcel_id': parcel_id,
                'area_hectares': round(area, 2),
                'soil_type': soil_type,
                'soil_ph': round(np.random.uniform(6.0, 7.5), 1),
                'irrigation_type': np.random.choice(['Drip', 'Sprinkler', 'Flood']),
                'slope': np.random.choice(['Flat', 'Gentle', 'Moderate']),
                'sun_exposure': np.random.choice(['Full Sun', 'Partial Sun', 'Mixed']),
                'last_crop': np.random.choice(list(self.crop_database.keys())),
                'last_harvest_date': datetime.now().date() - timedelta(days=np.random.randint(30, 180)),
                'soil_health_score': np.random.uniform(70, 95),
                'drainage': np.random.choice(['Excellent', 'Good', 'Fair']),
                'accessibility': np.random.choice(['Easy', 'Moderate', 'Difficult'])
            }
            parcels.append(parcel)
        
        # Generate planning data for next 12 months
        current_date = datetime.now().date()
        crops = list(self.crop_database.keys())
        
        for i, parcel in enumerate(parcels):
            # Plan 1-3 crops per parcel for the year
            num_plantings = np.random.randint(1, 4)
            
            for j in range(num_plantings):
                crop = np.random.choice(crops)
                crop_info = self.crop_database[crop]
                
                # Calculate planting date based on season
                planting_date = current_date + timedelta(days=np.random.randint(0, 180))
                harvest_date = planting_date + timedelta(days=crop_info['growth_days'])
                
                # Determine status based on dates
                if planting_date > current_date + timedelta(days=30):
                    status = 'Planned'
                elif planting_date > current_date:
                    status = 'Preparing'
                elif harvest_date > current_date:
                    status = 'Growing'
                else:
                    status = 'Completed'
                
                # Calculate expected yield and revenue
                expected_yield = parcel['area_hectares'] * crop_info['yield_per_hectare'] * np.random.uniform(0.8, 1.2)
                expected_revenue = expected_yield * crop_info['market_price_kg']
                
                planning_item = {
                    'plan_id': f'PL{i*10+j+1:03d}',
                    'parcel_id': parcel['parcel_id'],
                    'crop_type': crop,
                    'variety': f'{crop} Variety {np.random.randint(1, 4)}',
                    'planting_date': planting_date,
                    'expected_harvest_date': harvest_date,
                    'area_hectares': parcel['area_hectares'],
                    'expected_yield_kg': int(expected_yield),
                    'expected_revenue': round(expected_revenue, 2),
                    'status': 'Planned',
                    'priority': priority,
                    'notes': notes,
                    'cost_estimate': round(expected_revenue * 0.4, 2),
                    'profit_estimate': round(expected_revenue * 0.6, 2)
                }
                
                # Add to dataframe
                new_plan_df = pd.DataFrame([new_plan])
                st.session_state.planning_data = pd.concat([
                    st.session_state.planning_data,
                    new_plan_df
                ], ignore_index=True)
                
                st.success(f"New plantation plan {new_plan_id} added successfully!")
                st.rerun()

if __name__ == "__main__":
    main()status,
                    'priority': np.random.choice(['High', 'Medium', 'Low']),
                    'notes': f'Planned {crop} cultivation in {parcel["parcel_id"]}',
                    'cost_estimate': round(expected_revenue * np.random.uniform(0.3, 0.6), 2),
                    'profit_estimate': round(expected_revenue * np.random.uniform(0.4, 0.7), 2)
                }
                planning_data.append(planning_item)
        
        st.session_state.parcel_data = pd.DataFrame(parcels)
        st.session_state.planning_data = pd.DataFrame(planning_data)
        
        # Generate historical data for rotation analysis
        self.generate_rotation_history()
    
    def generate_rotation_history(self):
        """Generate historical rotation data for analysis"""
        history_data = []
        parcels = st.session_state.parcel_data['parcel_id'].tolist()
        
        for parcel_id in parcels:
            # Generate 3 years of history
            for year in range(2022, 2025):
                # 1-2 crops per year
                num_crops = np.random.randint(1, 3)
                for season in range(num_crops):
                    crop = np.random.choice(list(self.crop_database.keys()))
                    
                    history_item = {
                        'parcel_id': parcel_id,
                        'year': year,
                        'season': season + 1,
                        'crop_type': crop,
                        'rotation_family': self.crop_database[crop]['rotation_family'],
                        'yield_achieved': np.random.uniform(0.7, 1.3),  # Yield multiplier
                        'success_score': np.random.uniform(60, 95)
                    }
                    history_data.append(history_item)
        
        st.session_state.rotation_history = pd.DataFrame(history_data)
    
    def create_parcel_overview(self):
        """Create parcel overview dashboard"""
        st.subheader("🗺️ Farm Parcel Overview")
        
        parcels = st.session_state.parcel_data
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_area = parcels['area_hectares'].sum()
            st.metric("Total Farm Area", f"{total_area:.1f} ha")
        
        with col2:
            avg_soil_health = parcels['soil_health_score'].mean()
            st.metric("Avg Soil Health", f"{avg_soil_health:.1f}%")
        
        with col3:
            active_parcels = len(parcels)
            st.metric("Active Parcels", active_parcels)
        
        with col4:
            optimal_ph_parcels = len(parcels[(parcels['soil_ph'] >= 6.0) & (parcels['soil_ph'] <= 7.0)])
            st.metric("Optimal pH Parcels", optimal_ph_parcels)
        
        # Parcel visualization
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Parcel area distribution
            fig_area = px.treemap(
                parcels,
                values='area_hectares',
                names='parcel_id',
                title='Parcel Area Distribution',
                color='soil_health_score',
                color_continuous_scale='Greens',
                hover_data=['soil_type', 'soil_ph', 'irrigation_type']
            )
            fig_area.update_layout(height=400)
            st.plotly_chart(fig_area, use_container_width=True)
        
        with col2:
            # Soil type distribution
            soil_counts = parcels['soil_type'].value_counts()
            fig_soil = px.pie(
                values=soil_counts.values,
                names=soil_counts.index,
                title='Soil Type Distribution'
            )
            fig_soil.update_layout(height=300)
            st.plotly_chart(fig_soil, use_container_width=True)
    
    def create_planning_calendar(self):
        """Create interactive planning calendar"""
        st.subheader("📅 Planting Calendar")
        
        planning_data = st.session_state.planning_data
        
        # Month selection
        col1, col2, col3 = st.columns(3)
        
        with col1:
            current_year = datetime.now().year
            selected_year = st.selectbox("Year", [current_year, current_year + 1])
        
        with col2:
            selected_month = st.selectbox(
                "Month", 
                list(range(1, 13)),
                format_func=lambda x: calendar.month_name[x]
            )
        
        with col3:
            view_type = st.selectbox("View Type", ["Planting", "Harvesting", "Both"])
        
        # Filter data for selected month
        month_start = datetime(selected_year, selected_month, 1).date()
        if selected_month == 12:
            month_end = datetime(selected_year + 1, 1, 1).date() - timedelta(days=1)
        else:
            month_end = datetime(selected_year, selected_month + 1, 1).date() - timedelta(days=1)
        
        # Filter planning data
        if view_type == "Planting":
            month_data = planning_data[
                (planning_data['planting_date'] >= month_start) &
                (planning_data['planting_date'] <= month_end)
            ]
            date_col = 'planting_date'
        elif view_type == "Harvesting":
            month_data = planning_data[
                (planning_data['expected_harvest_date'] >= month_start) &
                (planning_data['expected_harvest_date'] <= month_end)
            ]
            date_col = 'expected_harvest_date'
        else:  # Both
            planting_data = planning_data[
                (planning_data['planting_date'] >= month_start) &
                (planning_data['planting_date'] <= month_end)
            ].copy()
            planting_data['activity_type'] = 'Planting'
            planting_data['activity_date'] = planting_data['planting_date']
            
            harvest_data = planning_data[
                (planning_data['expected_harvest_date'] >= month_start) &
                (planning_data['expected_harvest_date'] <= month_end)
            ].copy()
            harvest_data['activity_type'] = 'Harvesting'
            harvest_data['activity_date'] = harvest_data['expected_harvest_date']
            
            month_data = pd.concat([planting_data, harvest_data])
            date_col = 'activity_date'
        
        # Create calendar view
        if not month_data.empty:
            if view_type == "Both":
                fig = px.scatter(
                    month_data,
                    x=date_col,
                    y='parcel_id',
                    color='activity_type',
                    size='area_hectares',
                    hover_data=['crop_type', 'expected_yield_kg'],
                    title=f'{calendar.month_name[selected_month]} {selected_year} - Farm Calendar'
                )
            else:
                fig = px.scatter(
                    month_data,
                    x=date_col,
                    y='parcel_id',
                    color='crop_type',
                    size='area_hectares',
                    hover_data=['expected_yield_kg', 'status'],
                    title=f'{calendar.month_name[selected_month]} {selected_year} - {view_type} Schedule'
                )
            
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Activity list
            st.subheader(f"📋 {view_type} Activities")
            
            display_cols = ['plan_id', 'parcel_id', 'crop_type', date_col, 'area_hectares', 'expected_yield_kg', 'status']
            if view_type == "Both":
                display_cols.append('activity_type')
            
            st.dataframe(
                month_data[display_cols].sort_values(date_col),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info(f"No {view_type.lower()} activities scheduled for {calendar.month_name[selected_month]} {selected_year}")
    
    def create_crop_rotation_planner(self):
        """Create crop rotation planning tool"""
        st.subheader("🔄 Crop Rotation Planner")
        
        parcels = st.session_state.parcel_data
        history = st.session_state.rotation_history
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### Select Parcel")
            selected_parcel = st.selectbox(
                "Parcel",
                parcels['parcel_id'].tolist()
            )
            
            # Show parcel details
            parcel_info = parcels[parcels['parcel_id'] == selected_parcel].iloc[0]
            
            st.markdown("#### Parcel Details")
            st.write(f"**Area:** {parcel_info['area_hectares']} ha")
            st.write(f"**Soil Type:** {parcel_info['soil_type']}")
            st.write(f"**Soil pH:** {parcel_info['soil_ph']}")
            st.write(f"**Last Crop:** {parcel_info['last_crop']}")
            st.write(f"**Soil Health:** {parcel_info['soil_health_score']:.1f}%")
        
        with col2:
            st.markdown("### Rotation History & Recommendations")
            
            # Get rotation history for selected parcel
            parcel_history = history[history['parcel_id'] == selected_parcel].sort_values(['year', 'season'])
            
            if not parcel_history.empty:
                # Display rotation history
                st.markdown("#### 3-Year Rotation History")
                
                for year in sorted(parcel_history['year'].unique(), reverse=True):
                    year_data = parcel_history[parcel_history['year'] == year]
                    crops_text = " → ".join(year_data['crop_type'].tolist())
                    st.markdown(f"**{year}:** {crops_text}")
                
                # Rotation analysis
                st.markdown("#### Rotation Analysis")
                
                # Check for rotation violations
                recent_families = parcel_history.tail(4)['rotation_family'].tolist()
                violations = self.check_rotation_violations(recent_families)
                
                if violations:
                    st.warning("⚠️ Rotation Concerns:")
                    for violation in violations:
                        st.write(f"• {violation}")
                else:
                    st.success("✅ Good rotation practices observed")
                
                # Generate recommendations
                st.markdown("#### Recommended Next Crops")
                recommendations = self.generate_crop_recommendations(
                    parcel_info, recent_families[-1] if recent_families else None
                )
                
                for i, rec in enumerate(recommendations[:3], 1):
                    crop_info = self.crop_database[rec['crop']]
                    st.markdown(f"""
                    <div class="crop-rotation-card">
                        <h4>{i}. {crop_info['icon']} {rec['crop']}</h4>
                        <p><strong>Reason:</strong> {rec['reason']}</p>
                        <p><strong>Expected Yield:</strong> {rec['expected_yield']:,} kg</p>
                        <p><strong>Estimated Revenue:</strong> ${rec['estimated_revenue']:,.2f}</p>
                    </div>
                    """, unsafe_allow_html=True)
    
    def check_rotation_violations(self, recent_families):
        """Check for crop rotation violations"""
        violations = []
        
        if len(recent_families) < 2:
            return violations
        
        # Check for repeated families
        family_counts = {}
        for family in recent_families[-4:]:  # Last 4 seasons
            family_counts[family] = family_counts.get(family, 0) + 1
        
        for family, count in family_counts.items():
            if family in self.rotation_rules:
                rest_period = self.rotation_rules[family]['rest_period']
                if count > 1 and rest_period > 0:
                    violations.append(f"Too frequent {family} crops (appeared {count} times in last 4 seasons)")
        
        return violations
    
    def generate_crop_recommendations(self, parcel_info, last_family):
        """Generate crop recommendations based on rotation rules and parcel characteristics"""
        recommendations = []
        
        for crop, crop_info in self.crop_database.items():
            score = 0
            reason_parts = []
            
            # Rotation compatibility
            if last_family:
                crop_family = crop_info['rotation_family']
                rotation_rule = self.rotation_rules.get(crop_family, {})
                good_after = rotation_rule.get('good_after', [])
                avoid_after = rotation_rule.get('avoid_after', [])
                
                if last_family in good_after:
                    score += 3
                    reason_parts.append("excellent rotation fit")
                elif last_family not in avoid_after:
                    score += 1
                    reason_parts.append("compatible rotation")
                else:
                    score -= 2
                    reason_parts.append("rotation conflict")
            
            # Soil pH compatibility
            ph_min, ph_max = crop_info['soil_ph_range']
            if ph_min <= parcel_info['soil_ph'] <= ph_max:
                score += 2
                reason_parts.append("optimal soil pH")
            elif abs(parcel_info['soil_ph'] - (ph_min + ph_max)/2) < 0.5:
                score += 1
                reason_parts.append("acceptable soil pH")
            
            # Calculate potential yield and revenue
            base_yield = parcel_info['area_hectares'] * crop_info['yield_per_hectare']
            
            # Adjust for soil health
            health_multiplier = parcel_info['soil_health_score'] / 100
            expected_yield = base_yield * health_multiplier
            estimated_revenue = expected_yield * crop_info['market_price_kg']
            
            # Add economic factor to score
            if crop_info['market_price_kg'] > 4.0:
                score += 1
                reason_parts.append("high market value")
            
            recommendations.append({
                'crop': crop,
                'score': score,
                'reason': ", ".join(reason_parts) if reason_parts else "basic compatibility",
                'expected_yield': int(expected_yield),
                'estimated_revenue': estimated_revenue
            })
        
        # Sort by score
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        return recommendations
    
    def create_economic_projections(self):
        """Create economic projections for plantation plans"""
        st.subheader("💰 Economic Projections")
        
        planning_data = st.session_state.planning_data
        
        # Time period selection
        col1, col2 = st.columns(2)
        
        with col1:
            projection_period = st.selectbox(
                "Projection Period",
                ["Next 3 Months", "Next 6 Months", "Next Year", "Custom Range"]
            )
        
        with col2:
            if projection_period == "Custom Range":
                start_date = st.date_input("Start Date", datetime.now().date())
                end_date = st.date_input("End Date", datetime.now().date() + timedelta(days=365))
            else:
                start_date = datetime.now().date()
                if projection_period == "Next 3 Months":
                    end_date = start_date + timedelta(days=90)
                elif projection_period == "Next 6 Months":
                    end_date = start_date + timedelta(days=180)
                else:  # Next Year
                    end_date = start_date + timedelta(days=365)
        
        # Filter data
        period_data = planning_data[
            (planning_data['expected_harvest_date'] >= start_date) &
            (planning_data['expected_harvest_date'] <= end_date)
        ]
        
        if not period_data.empty:
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_revenue = period_data['expected_revenue'].sum()
                st.metric("Expected Revenue", f"${total_revenue:,.2f}")
            
            with col2:
                total_costs = period_data['cost_estimate'].sum()
                st.metric("Estimated Costs", f"${total_costs:,.2f}")
            
            with col3:
                total_profit = period_data['profit_estimate'].sum()
                st.metric("Expected Profit", f"${total_profit:,.2f}")
            
            with col4:
                profit_margin = (total_profit / total_revenue * 100) if total_revenue > 0 else 0
                st.metric("Profit Margin", f"{profit_margin:.1f}%")
            
            # Revenue by crop
            col1, col2 = st.columns(2)
            
            with col1:
                crop_revenue = period_data.groupby('crop_type')['expected_revenue'].sum().sort_values(ascending=False)
                fig_revenue = px.bar(
                    x=crop_revenue.values,
                    y=crop_revenue.index,
                    orientation='h',
                    title='Expected Revenue by Crop',
                    color=crop_revenue.values,
                    color_continuous_scale='Greens'
                )
                fig_revenue.update_layout(height=400)
                st.plotly_chart(fig_revenue, use_container_width=True)
            
            with col2:
                # Monthly revenue projection
                period_data['harvest_month'] = pd.to_datetime(period_data['expected_harvest_date']).dt.to_period('M')
                monthly_revenue = period_data.groupby('harvest_month')['expected_revenue'].sum()
                
                fig_monthly = px.line(
                    x=monthly_revenue.index.astype(str),
                    y=monthly_revenue.values,
                    title='Monthly Revenue Projection',
                    markers=True
                )
                fig_monthly.update_layout(height=400)
                st.plotly_chart(fig_monthly, use_container_width=True)
            
            # Detailed projection table
            st.subheader("📊 Detailed Projections")
            
            display_data = period_data[[
                'plan_id', 'parcel_id', 'crop_type', 'expected_harvest_date',
                'expected_yield_kg', 'expected_revenue', 'cost_estimate', 'profit_estimate'
            ]].sort_values('expected_harvest_date')
            
            st.dataframe(display_data, use_container_width=True, hide_index=True)
        else:
            st.info("No plantations planned for the selected period.")

def main():
    st.title("🗺️ Plantation Planner")
    st.markdown("Plan and optimize crop rotations across your farm parcels")
    
    planner = PlantationPlanner()
    
    # Sidebar controls
    with st.sidebar:
        st.header("🎛️ Planning Tools")
        
        # View selection
        view_mode = st.radio(
            "Select View",
            ["Parcel Overview", "Planning Calendar", "Crop Rotation", "Economic Projections", "Add New Plan"]
        )
        
        st.markdown("---")
        
        # Quick actions
        if st.button("🔄 Refresh Data"):
            keys_to_clear = ['parcel_data', 'planning_data', 'rotation_history']
            for key in keys_to_clear:
                if key in st.session_state:
                    del st.session_state[key]
            st.success("Data refreshed!")
            st.rerun()
        
        if st.button("📊 Generate Reports"):
            st.success("Reports generated!")
        
        st.markdown("---")
        
        # Quick stats
        if 'planning_data' in st.session_state:
            planning_data = st.session_state.planning_data
            st.markdown("### 📈 Quick Stats")
            st.metric("Total Plans", len(planning_data))
            st.metric("Expected Revenue", f"${planning_data['expected_revenue'].sum():,.0f}")
            st.metric("Active Parcels", len(st.session_state.parcel_data))
    
    # Main content based on view mode
    if view_mode == "Parcel Overview":
        planner.create_parcel_overview()
        
    elif view_mode == "Planning Calendar":
        planner.create_planning_calendar()
        
    elif view_mode == "Crop Rotation":
        planner.create_crop_rotation_planner()
        
    elif view_mode == "Economic Projections":
        planner.create_economic_projections()
        
    elif view_mode == "Add New Plan":
        st.subheader("➕ Add New Plantation Plan")
        
        with st.form("new_plan_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                # Get available parcels
                available_parcels = st.session_state.parcel_data['parcel_id'].tolist()
                selected_parcel = st.selectbox("Select Parcel", available_parcels)
                
                crop_type = st.selectbox("Crop Type", list(planner.crop_database.keys()))
                variety = st.text_input("Variety", f"{crop_type} Variety 1")
                planting_date = st.date_input("Planting Date", datetime.now().date())
            
            with col2:
                # Get parcel info for calculations
                parcel_info = st.session_state.parcel_data[
                    st.session_state.parcel_data['parcel_id'] == selected_parcel
                ].iloc[0]
                
                area_to_use = st.number_input(
                    "Area to Use (ha)", 
                    min_value=0.1, 
                    max_value=float(parcel_info['area_hectares']),
                    value=float(parcel_info['area_hectares'])
                )
                
                priority = st.selectbox("Priority", ["High", "Medium", "Low"])
                notes = st.text_area("Notes", "New plantation plan")
            
            # Calculate projections
            if crop_type in planner.crop_database:
                crop_info = planner.crop_database[crop_type]
                harvest_date = planting_date + timedelta(days=crop_info['growth_days'])
                expected_yield = area_to_use * crop_info['yield_per_hectare']
                expected_revenue = expected_yield * crop_info['market_price_kg']
                
                st.markdown("### 📊 Projections")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Expected Harvest", harvest_date.strftime("%Y-%m-%d"))
                
                with col2:
                    st.metric("Expected Yield", f"{expected_yield:,.0f} kg")
                
                with col3:
                    st.metric("Expected Revenue", f"${expected_revenue:,.2f}")
            
            if st.form_submit_button("Add Plan"):
                # Generate new plan ID
                existing_plans = st.session_state.planning_data['plan_id'].tolist()
                plan_numbers = [int(p[2:]) for p in existing_plans if p[2:].isdigit()]
                new_number = max(plan_numbers) + 1 if plan_numbers else 1
                new_plan_id = f"PL{new_number:03d}"
                
                # Create new plan
                new_plan = {
                    'plan_id': new_plan_id,
                    'parcel_id': selected_parcel,
                    'crop_type': crop_type,
                    'variety': variety,
                    'planting_date': planting_date,
                    'expected_harvest_date': harvest_date,
                    'area_hectares': area_to_use,
                    'expected_yield_kg': int(expected_yield),
                    'expected_revenue': round(expected_revenue, 2),
                    'status':
