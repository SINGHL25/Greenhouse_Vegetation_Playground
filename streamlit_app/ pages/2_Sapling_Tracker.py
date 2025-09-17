
# pages/2_Sapling_Tracker.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import json

# Page configuration
st.set_page_config(
    page_title="Sapling Tracker",
    page_icon="🌱",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .batch-card {
        background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: bold;
        margin: 0.25rem;
    }
    
    .status-seedling { background-color: #ffeaa7; color: #2d3436; }
    .status-growing { background-color: #55a3ff; color: white; }
    .status-ready { background-color: #00b894; color: white; }
    .status-transplanted { background-color: #6c5ce7; color: white; }
    .status-failed { background-color: #e17055; color: white; }
    
    .growth-metric {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .progress-bar {
        background-color: #ecf0f1;
        border-radius: 10px;
        overflow: hidden;
        height: 20px;
        margin: 0.5rem 0;
    }
    
    .progress-fill {
        height: 100%;
        transition: width 0.3s ease;
    }
</style>
""", unsafe_allow_html=True)

class SaplingTracker:
    def __init__(self):
        self.load_data()
        self.init_growth_stages()
    
    def load_data(self):
        """Load or generate sapling tracking data"""
        if 'sapling_batches' not in st.session_state:
            self.generate_sample_data()
    
    def init_growth_stages(self):
        """Define growth stages and their characteristics"""
        self.growth_stages = {
            'seed': {
                'name': 'Seed',
                'days': 0,
                'description': 'Seeds planted, awaiting germination',
                'color': '#8b4513',
                'icon': '🌰'
            },
            'germination': {
                'name': 'Germination',
                'days': 3,
                'description': 'Seeds germinating, first signs of life',
                'color': '#ffeaa7',
                'icon': '🌱'
            },
            'seedling': {
                'name': 'Seedling',
                'days': 14,
                'description': 'First true leaves developing',
                'color': '#55a3ff',
                'icon': '🌿'
            },
            'young_plant': {
                'name': 'Young Plant',
                'days': 28,
                'description': 'Established root system, multiple leaves',
                'color': '#00b894',
                'icon': '🌳'
            },
            'ready_transplant': {
                'name': 'Ready for Transplant',
                'days': 42,
                'description': 'Strong enough for field transplanting',
                'color': '#6c5ce7',
                'icon': '🚀'
            }
        }
    
    def generate_sample_data(self):
        """Generate comprehensive sample sapling data"""
        # Generate batch data
        batches = []
        for i in range(1, 16):  # 15 batches
            seed_date = datetime.now().date() - timedelta(days=np.random.randint(1, 90))
            days_since_seeding = (datetime.now().date() - seed_date).days
            
            # Determine growth stage based on days
            stage = self.determine_growth_stage(days_since_seeding)
            
            initial_seeds = np.random.randint(800, 1200)
            germination_rate = np.random.uniform(0.75, 0.95)
            survival_rate = np.random.uniform(0.85, 0.98)
            
            germinated_count = int(initial_seeds * germination_rate)
            current_count = int(germinated_count * survival_rate)
            
            # Growth metrics
            expected_height = self.calculate_expected_height(days_since_seeding, stage)
            avg_height = expected_height * np.random.uniform(0.8, 1.2)
            
            batch = {
                'batch_id': f'B{i:03d}',
                'crop_type': np.random.choice(['Tomato', 'Pepper', 'Lettuce', 'Basil', 'Cucumber']),
                'variety': np.random.choice(['Roma', 'Cherry', 'Beefsteak', 'Bell', 'Hot', 'Butter', 'Iceberg']),
                'seed_date': seed_date,
                'days_since_seeding': days_since_seeding,
                'initial_seed_count': initial_seeds,
                'germinated_count': germinated_count,
                'current_count': current_count,
                'failed_count': initial_seeds - current_count,
                'germination_rate': germination_rate * 100,
                'survival_rate': survival_rate * 100,
                'current_stage': stage,
                'avg_height_cm': avg_height,
                'avg_leaves': max(2, int(days_since_seeding / 7) + np.random.randint(-1, 3)),
                'ready_for_transplant': current_count if stage == 'ready_transplant' else 0,
                'greenhouse_section': f'Section {np.random.choice(["A", "B", "C", "D"])}',
                'tray_count': np.random.randint(15, 40),
                'health_score': np.random.uniform(75, 98),
                'notes': self.generate_batch_notes(stage, days_since_seeding),
                'last_watered': datetime.now().date() - timedelta(days=np.random.randint(0, 3)),
                'last_fertilized': datetime.now().date() - timedelta(days=np.random.randint(0, 14)),
                'expected_transplant_date': seed_date + timedelta(days=42)
            }
            batches.append(batch)
        
        st.session_state.sapling_batches = pd.DataFrame(batches)
        
        # Generate daily growth tracking data
        self.generate_growth_tracking_data()
    
    def generate_batch_notes(self, stage, days):
        """Generate realistic notes for each batch"""
        notes_options = {
            'seed': ['Seeds planted in sterile medium', 'Optimal temperature maintained', 'Awaiting germination'],
            'germination': ['First sprouts visible', 'Good germination rate observed', '70-85% germination achieved'],
            'seedling': ['First true leaves emerging', 'Healthy green color', 'Regular watering maintained'],
            'young_plant': ['Strong root development', 'Multiple leaf sets', 'Plants showing vigorous growth'],
            'ready_transplant': ['Ready for field planting', 'Robust root system', 'Plants hardened successfully']
        }
        
        stage_notes = notes_options.get(stage, ['Normal development'])
        return np.random.choice(stage_notes)
    
    def determine_growth_stage(self, days_since_seeding):
        """Determine current growth stage based on days since seeding"""
        if days_since_seeding < 3:
            return 'seed'
        elif days_since_seeding < 14:
            return 'germination'
        elif days_since_seeding < 28:
            return 'seedling'
        elif days_since_seeding < 42:
            return 'young_plant'
        else:
            return 'ready_transplant'
    
    def calculate_expected_height(self, days, stage):
        """Calculate expected height based on growth stage"""
        height_map = {
            'seed': 0,
            'germination': np.random.uniform(0.5, 1.5),
            'seedling': np.random.uniform(2, 5),
            'young_plant': np.random.uniform(5, 12),
            'ready_transplant': np.random.uniform(12, 20)
        }
        return height_map.get(stage, 0)
    
    def generate_growth_tracking_data(self):
        """Generate daily growth measurements for selected batches"""
        if 'growth_tracking' not in st.session_state:
            tracking_data = []
            
            # Select a few batches for detailed tracking
            sample_batches = st.session_state.sapling_batches.sample(5)['batch_id'].tolist()
            
            for batch_id in sample_batches:
                batch_info = st.session_state.sapling_batches[
                    st.session_state.sapling_batches['batch_id'] == batch_id
                ].iloc[0]
                
                # Generate daily measurements from seed date to now
                current_date = batch_info['seed_date']
                end_date = datetime.now().date()
                
                base_height = 0
                while current_date <= end_date:
                    days_since_seed = (current_date - batch_info['seed_date']).days
                    stage = self.determine_growth_stage(days_since_seed)
                    
                    # Simulate growth with some randomness
                    if days_since_seed > 3:  # After germination
                        daily_growth = np.random.uniform(0.1, 0.5) if days_since_seed < 30 else np.random.uniform(0.05, 0.2)
                        base_height += daily_growth
                    
                    # Add some measurement noise
                    measured_height = base_height * np.random.uniform(0.95, 1.05)
                    
                    tracking_data.append({
                        'batch_id': batch_id,
                        'date': current_date,
                        'days_since_seeding': days_since_seed,
                        'avg_height_cm': max(0, measured_height),
                        'leaf_count': max(0, int(days_since_seed / 7) + np.random.randint(-1, 2)),
                        'survival_count': int(batch_info['current_count'] * np.random.uniform(0.98, 1.0)),
                        'growth_stage': stage,
                        'notes': f'Daily measurement - {stage} stage'
                    })
                    
                    current_date += timedelta(days=1)
            
            st.session_state.growth_tracking = pd.DataFrame(tracking_data)
    
    def create_batch_overview(self):
        """Create batch overview cards"""
        st.subheader("📊 Batch Overview")
        
        batches = st.session_state.sapling_batches.sort_values('seed_date', ascending=False)
        
        # Summary metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            total_batches = len(batches)
            st.metric("Total Batches", total_batches)
        
        with col2:
            active_batches = len(batches[batches['current_stage'] != 'ready_transplant'])
            st.metric("Active Batches", active_batches)
        
        with col3:
            total_saplings = batches['current_count'].sum()
            st.metric("Total Saplings", f"{total_saplings:,}")
        
        with col4:
            ready_count = batches['ready_for_transplant'].sum()
            st.metric("Ready for Transplant", f"{ready_count:,}")
        
        with col5:
            avg_health = batches['health_score'].mean()
            st.metric("Avg Health Score", f"{avg_health:.1f}%")
        
        # Stage distribution
        st.subheader("🌱 Growth Stage Distribution")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            stage_counts = batches['current_stage'].value_counts()
            
            # Create stage distribution chart
            fig = px.bar(
                x=stage_counts.index,
                y=stage_counts.values,
                title="Batches by Growth Stage",
                color=stage_counts.values,
                color_continuous_scale="Greens"
            )
            fig.update_layout(
                xaxis_title="Growth Stage",
                yaxis_title="Number of Batches",
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Stage pie chart
            fig_pie = px.pie(
                values=stage_counts.values,
                names=stage_counts.index,
                title="Stage Distribution"
            )
            fig_pie.update_layout(height=300)
            st.plotly_chart(fig_pie, use_container_width=True)
    
    def create_batch_details(self):
        """Create detailed batch information cards"""
        st.subheader("📋 Detailed Batch Information")
        
        batches = st.session_state.sapling_batches.sort_values('days_since_seeding', ascending=False)
        
        # Batch filtering
        col1, col2, col3 = st.columns(3)
        
        with col1:
            stage_filter = st.selectbox(
                "Filter by Stage",
                ['All'] + list(batches['current_stage'].unique())
            )
        
        with col2:
            crop_filter = st.selectbox(
                "Filter by Crop",
                ['All'] + list(batches['crop_type'].unique())
            )
        
        with col3:
            sort_by = st.selectbox(
                "Sort by",
                ['Seed Date', 'Days Since Seeding', 'Health Score', 'Count']
            )
        
        # Apply filters
        filtered_batches = batches.copy()
        if stage_filter != 'All':
            filtered_batches = filtered_batches[filtered_batches['current_stage'] == stage_filter]
        if crop_filter != 'All':
            filtered_batches = filtered_batches[filtered_batches['crop_type'] == crop_filter]
        
        # Display batch cards
        for idx, (_, batch) in enumerate(filtered_batches.head(10).iterrows()):
            self.create_batch_card(batch)
    
    def create_batch_card(self, batch):
        """Create individual batch card"""
        stage_info = self.growth_stages.get(batch['current_stage'], {})
        stage_color = stage_info.get('color', '#74b9ff')
        stage_icon = stage_info.get('icon', '🌱')
        
        # Progress calculation
        days_in_stage = batch['days_since_seeding']
        max_days = 42  # Days to transplant ready
        progress = min(100, (days_in_stage / max_days) * 100)
        
        col1, col2, col3 = st.columns([2, 2, 1])
        
        with col1:
            st.markdown(f"""
            <div class="batch-card" style="background: linear-gradient(135deg, {stage_color} 0%, {stage_color}dd 100%);">
                <h3>{stage_icon} {batch['batch_id']} - {batch['crop_type']} ({batch['variety']})</h3>
                <p><strong>Seeded:</strong> {batch['seed_date']} ({batch['days_since_seeding']} days ago)</p>
                <p><strong>Current Stage:</strong> {stage_info.get('name', batch['current_stage'])}</p>
                <p><strong>Section:</strong> {batch['greenhouse_section']} | <strong>Trays:</strong> {batch['tray_count']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Growth metrics
            st.markdown("### 📊 Growth Metrics")
            
            col2a, col2b, col2c = st.columns(3)
            with col2a:
                st.metric("Current Count", f"{batch['current_count']:,}")
                st.metric("Survival Rate", f"{batch['survival_rate']:.1f}%")
            
            with col2b:
                st.metric("Avg Height", f"{batch['avg_height_cm']:.1f} cm")
                st.metric("Avg Leaves", f"{batch['avg_leaves']}")
            
            with col2c:
                st.metric("Health Score", f"{batch['health_score']:.1f}%")
                st.metric("Ready Count", f"{batch['ready_for_transplant']:,}")
            
            # Progress bar
            st.markdown("### Progress to Transplant")
            st.progress(progress / 100)
            st.caption(f"{progress:.1f}% complete")
        
        with col3:
            st.markdown("### 🎯 Actions")
            
            if st.button(f"📝 Update {batch['batch_id']}", key=f"update_{batch['batch_id']}"):
                self.show_update_form(batch['batch_id'])
            
            if st.button(f"📊 Growth Chart {batch['batch_id']}", key=f"chart_{batch['batch_id']}"):
                self.show_growth_chart(batch['batch_id'])
            
            if batch['ready_for_transplant'] > 0:
                if st.button(f"🚀 Transplant {batch['batch_id']}", key=f"transplant_{batch['batch_id']}"):
                    st.success(f"Transplant scheduled for {batch['batch_id']}")
        
        st.markdown("---")
    
    def create_growth_analytics(self):
        """Create growth analytics dashboard"""
        st.subheader("📈 Growth Analytics")
        
        if 'growth_tracking' not in st.session_state:
            st.warning("No detailed growth tracking data available. Generate sample data first.")
            return
        
        tracking_data = st.session_state.growth_tracking
        
        # Batch selection for detailed view
        available_batches = tracking_data['batch_id'].unique()
        selected_batch = st.selectbox("Select Batch for Detailed Analysis", available_batches)
        
        batch_data = tracking_data[tracking_data['batch_id'] == selected_batch]
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Height growth over time
            fig_height = px.line(
                batch_data,
                x='date',
                y='avg_height_cm',
                title=f'Height Growth - {selected_batch}',
                markers=True
            )
            fig_height.update_layout(
                xaxis_title="Date",
                yaxis_title="Average Height (cm)"
            )
            st.plotly_chart(fig_height, use_container_width=True)
        
        with col2:
            # Leaf count growth
            fig_leaves = px.line(
                batch_data,
                x='date',
                y='leaf_count',
                title=f'Leaf Development - {selected_batch}',
                markers=True,
                color_discrete_sequence=['green']
            )
            fig_leaves.update_layout(
                xaxis_title="Date",
                yaxis_title="Average Leaf Count"
            )
            st.plotly_chart(fig_leaves, use_container_width=True)
        
        # Growth rate analysis
        st.subheader("📊 Growth Rate Analysis")
        
        # Calculate growth rates
        batch_data_sorted = batch_data.sort_values('date')
        batch_data_sorted['height_growth_rate'] = batch_data_sorted['avg_height_cm'].diff()
        batch_data_sorted['days_diff'] = batch_data_sorted['days_since_seeding'].diff()
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Daily growth rate
            fig_rate = px.bar(
                batch_data_sorted.tail(14),  # Last 14 days
                x='date',
                y='height_growth_rate',
                title='Daily Growth Rate (Last 14 Days)',
                color='height_growth_rate',
                color_continuous_scale='Greens'
            )
            fig_rate.update_layout(
                xaxis_title="Date",
                yaxis_title="Height Growth Rate (cm/day)"
            )
            st.plotly_chart(fig_rate, use_container_width=True)
        
        with col2:
            # Growth stage timeline
            stage_timeline = batch_data.groupby('growth_stage').agg({
                'days_since_seeding': ['min', 'max'],
                'avg_height_cm': 'mean'
            }).round(2)
            
            st.markdown("### Growth Stage Timeline")
            st.dataframe(stage_timeline, use_container_width=True)
    
    def show_update_form(self, batch_id):
        """Show batch update form in modal"""
        st.subheader(f"Update Batch {batch_id}")
        
        # Get current batch data
        current_data = st.session_state.sapling_batches[
            st.session_state.sapling_batches['batch_id'] == batch_id
        ].iloc[0]
        
        with st.form(f"update_form_{batch_id}"):
            col1, col2 = st.columns(2)
            
            with col1:
                new_height = st.number_input(
                    "Average Height (cm)",
                    value=float(current_data['avg_height_cm']),
                    min_value=0.0,
                    step=0.1
                )
                
                new_leaves = st.number_input(
                    "Average Leaves",
                    value=int(current_data['avg_leaves']),
                    min_value=0,
                    step=1
                )
                
                new_count = st.number_input(
                    "Current Count",
                    value=int(current_data['current_count']),
                    min_value=0,
                    step=1
                )
            
            with col2:
                new_health = st.slider(
                    "Health Score",
                    min_value=0.0,
                    max_value=100.0,
                    value=float(current_data['health_score']),
                    step=0.1
                )
                
                new_ready = st.number_input(
                    "Ready for Transplant",
                    value=int(current_data['ready_for_transplant']),
                    min_value=0,
                    max_value=int(current_data['current_count']),
                    step=1
                )
                
                notes = st.text_area(
                    "Notes",
                    value=current_data['notes']
                )
            
            if st.form_submit_button("Update Batch"):
                # Update the batch data
                mask = st.session_state.sapling_batches['batch_id'] == batch_id
                st.session_state.sapling_batches.loc[mask, 'avg_height_cm'] = new_height
                st.session_state.sapling_batches.loc[mask, 'avg_leaves'] = new_leaves
                st.session_state.sapling_batches.loc[mask, 'current_count'] = new_count
                st.session_state.sapling_batches.loc[mask, 'health_score'] = new_health
                st.session_state.sapling_batches.loc[mask, 'ready_for_transplant'] = new_ready
                st.session_state.sapling_batches.loc[mask, 'notes'] = notes
                
                st.success(f"Batch {batch_id} updated successfully!")
                st.rerun()

def main():
    st.title("🌱 Sapling Growth Tracker")
    st.markdown("Monitor and manage sapling development from seed to transplant")
    
    tracker = SaplingTracker()
    
    # Sidebar controls
    with st.sidebar:
        st.header("🎛️ Tracker Controls")
        
        # View selection
        view_mode = st.radio(
            "Select View",
            ["Overview", "Batch Details", "Growth Analytics", "Add New Batch"]
        )
        
        st.markdown("---")
        
        # Quick actions
        if st.button("🔄 Refresh Data"):
            # Clear and regenerate data
            keys_to_clear = ['sapling_batches', 'growth_tracking']
            for key in keys_to_clear:
                if key in st.session_state:
                    del st.session_state[key]
            st.success("Data refreshed!")
            st.rerun()
        
        if st.button("📊 Generate Sample Tracking"):
            tracker.generate_growth_tracking_data()
            st.success("Sample tracking data generated!")
        
        st.markdown("---")
        
        # Summary stats
        if 'sapling_batches' in st.session_state:
            batches = st.session_state.sapling_batches
            st.markdown("### 📈 Quick Stats")
            st.metric("Total Batches", len(batches))
            st.metric("Total Saplings", f"{batches['current_count'].sum():,}")
            st.metric("Avg Health", f"{batches['health_score'].mean():.1f}%")
    
    # Main content based on view mode
    if view_mode == "Overview":
        tracker.create_batch_overview()
        
    elif view_mode == "Batch Details":
        tracker.create_batch_details()
        
    elif view_mode == "Growth Analytics":
        tracker.create_growth_analytics()
        
    elif view_mode == "Add New Batch":
        st.subheader("➕ Add New Batch")
        
        with st.form("new_batch_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                crop_type = st.selectbox("Crop Type", ["Tomato", "Pepper", "Lettuce", "Basil", "Cucumber"])
                variety = st.text_input("Variety", "")
                seed_count = st.number_input("Initial Seed Count", min_value=1, value=1000)
                greenhouse_section = st.selectbox("Greenhouse Section", ["Section A", "Section B", "Section C", "Section D"])
            
            with col2:
                seed_date = st.date_input("Seed Date", datetime.now().date())
                tray_count = st.number_input("Number of Trays", min_value=1, value=25)
                notes = st.text_area("Initial Notes", "New batch planted")
            
            if st.form_submit_button("Add Batch"):
                # Generate new batch ID
                existing_batches = st.session_state.sapling_batches['batch_id'].tolist()
                batch_numbers = [int(b[1:]) for b in existing_batches if b[1:].isdigit()]
                new_number = max(batch_numbers) + 1 if batch_numbers else 1
                new_batch_id = f"B{new_number:03d}"
                
                # Create new batch record
                new_batch = {
                    'batch_id': new_batch_id,
                    'crop_type': crop_type,
                    'variety': variety,
                    'seed_date': seed_date,
                    'days_since_seeding': (datetime.now().date() - seed_date).days,
                    'initial_seed_count': seed_count,
                    'germinated_count': 0,
                    'current_count': seed_count,
                    'failed_count': 0,
                    'germination_rate': 0.0,
                    'survival_rate': 100.0,
                    'current_stage': 'seed',
                    'avg_height_cm': 0.0,
                    'avg_leaves': 0,
                    'ready_for_transplant': 0,
                    'greenhouse_section': greenhouse_section,
                    'tray_count': tray_count,
                    'health_score': 95.0,
                    'notes': notes,
                    'last_watered': datetime.now().date(),
                    'last_fertilized': datetime.now().date(),
                    'expected_transplant_date': seed_date + timedelta(days=42)
                }
                
                # Add to dataframe
                new_batch_df = pd.DataFrame([new_batch])
                st.session_state.sapling_batches = pd.concat([
                    st.session_state.sapling_batches,
                    new_batch_df
                ], ignore_index=True)
                
                st.success(f"New batch {new_batch_id} added successfully!")
                st.rerun()

if __name__ == "__main__":
    main()
