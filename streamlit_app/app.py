
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# Configure page settings
st.set_page_config(
    page_title="AgriTech Management System",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #2E7D32 0%, #4CAF50 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #4CAF50;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    
    .status-good { color: #4CAF50; font-weight: bold; }
    .status-warning { color: #FF9800; font-weight: bold; }
    .status-alert { color: #f44336; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

def init_sample_data():
    """Initialize sample data for demonstration"""
    if 'greenhouse_data' not in st.session_state:
        # Sample greenhouse sensor data
        dates = pd.date_range(start='2024-01-01', end='2024-09-17', freq='H')
        st.session_state.greenhouse_data = pd.DataFrame({
            'timestamp': dates,
            'temperature': np.random.normal(25, 3, len(dates)),
            'humidity': np.random.normal(65, 10, len(dates)),
            'soil_ph': np.random.normal(6.5, 0.3, len(dates)),
            'soil_moisture': np.random.normal(45, 8, len(dates)),
            'light_intensity': np.random.normal(800, 200, len(dates)),
            'co2_level': np.random.normal(400, 50, len(dates))
        })
    
    if 'sapling_data' not in st.session_state:
        # Sample sapling tracking data
        st.session_state.sapling_data = pd.DataFrame({
            'batch_id': ['B001', 'B002', 'B003', 'B004', 'B005'],
            'seed_date': ['2024-06-01', '2024-06-15', '2024-07-01', '2024-07-15', '2024-08-01'],
            'germination_rate': [85, 92, 78, 88, 90],
            'current_height': [12.5, 10.8, 8.2, 6.5, 4.1],
            'survival_rate': [92, 95, 85, 90, 88],
            'ready_for_transplant': [45, 38, 25, 18, 8]
        })
    
    if 'plantation_data' not in st.session_state:
        # Sample plantation planning data
        st.session_state.plantation_data = pd.DataFrame({
            'parcel_id': ['P001', 'P002', 'P003', 'P004', 'P005'],
            'area_hectares': [2.5, 3.2, 1.8, 4.1, 2.9],
            'crop_type': ['Tomatoes', 'Peppers', 'Lettuce', 'Tomatoes', 'Herbs'],
            'planting_date': ['2024-08-15', '2024-08-20', '2024-09-01', '2024-09-10', '2024-09-15'],
            'expected_yield': [125, 96, 54, 205, 87],
            'status': ['Growing', 'Growing', 'Planted', 'Planted', 'Preparing']
        })

def display_overview_dashboard():
    """Display the main overview dashboard"""
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🌱 AgriTech Management System</h1>
        <p>Comprehensive Agricultural Management for Modern Farmers</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Quick Stats Row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="🌡️ Avg Temperature",
            value="24.8°C",
            delta="0.5°C from yesterday"
        )
    
    with col2:
        st.metric(
            label="💧 Soil Moisture",
            value="47.2%",
            delta="-2.1% from yesterday"
        )
    
    with col3:
        st.metric(
            label="🌱 Active Saplings",
            value="2,450",
            delta="85 new germinations"
        )
    
    with col4:
        st.metric(
            label="💰 Monthly Revenue",
            value="$15,240",
            delta="12.5% increase"
        )
    
    st.markdown("---")
    
    # Main Content Area
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        st.subheader("🏠 Greenhouse Environmental Status")
        
        # Environmental data chart
        greenhouse_data = st.session_state.greenhouse_data
        recent_data = greenhouse_data.tail(168)  # Last 7 days
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Temperature (°C)', 'Humidity (%)', 'Soil pH', 'Soil Moisture (%)'),
            vertical_spacing=0.08
        )
        
        fig.add_trace(
            go.Scatter(x=recent_data['timestamp'], y=recent_data['temperature'], 
                      name='Temperature', line=dict(color='#FF6B6B')),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=recent_data['timestamp'], y=recent_data['humidity'], 
                      name='Humidity', line=dict(color='#4ECDC4')),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Scatter(x=recent_data['timestamp'], y=recent_data['soil_ph'], 
                      name='Soil pH', line=dict(color='#45B7D1')),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=recent_data['timestamp'], y=recent_data['soil_moisture'], 
                      name='Soil Moisture', line=dict(color='#96CEB4')),
            row=2, col=2
        )
        
        fig.update_layout(height=400, showlegend=False, title_text="Environmental Monitoring - Last 7 Days")
        st.plotly_chart(fig, use_container_width=True)
    
    with col_right:
        st.subheader("🚨 System Alerts")
        
        # Alert system
        alerts = [
            {"type": "warning", "message": "Greenhouse 2: Humidity above optimal range", "time": "2 hours ago"},
            {"type": "good", "message": "Batch B003: Ready for transplanting", "time": "4 hours ago"},
            {"type": "alert", "message": "Parcel P002: Irrigation system maintenance needed", "time": "1 day ago"},
            {"type": "good", "message": "Sapling survival rate: 91% (Above target)", "time": "2 days ago"}
        ]
        
        for alert in alerts:
            status_class = f"status-{alert['type']}"
            st.markdown(f"""
            <div class="metric-card">
                <span class="{status_class}">●</span> {alert['message']}
                <br><small style="color: #666;">{alert['time']}</small>
            </div>
            """, unsafe_allow_html=True)
        
        st.subheader("📊 Quick Actions")
        if st.button("🔄 Refresh Sensor Data", use_container_width=True):
            st.success("Sensor data refreshed!")
            st.rerun()
        
        if st.button("📝 Log Manual Entry", use_container_width=True):
            st.info("Manual entry form opened!")
        
        if st.button("📱 Send Alerts", use_container_width=True):
            st.success("Alerts sent to mobile devices!")
    
    # Sapling Status Overview
    st.markdown("---")
    st.subheader("🌱 Sapling Growth Overview")
    
    sapling_data = st.session_state.sapling_data
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Germination rates chart
        fig_germ = px.bar(
            sapling_data, 
            x='batch_id', 
            y='germination_rate',
            title='Germination Rates by Batch',
            color='germination_rate',
            color_continuous_scale='Greens'
        )
        fig_germ.update_layout(height=300)
        st.plotly_chart(fig_germ, use_container_width=True)
    
    with col2:
        # Growth progress
        fig_height = px.scatter(
            sapling_data,
            x='batch_id',
            y='current_height',
            size='ready_for_transplant',
            title='Sapling Height vs Ready for Transplant',
            color='survival_rate',
            color_continuous_scale='Viridis'
        )
        fig_height.update_layout(height=300)
        st.plotly_chart(fig_height, use_container_width=True)

def setup_sidebar():
    """Setup the sidebar navigation and controls"""
    with st.sidebar:
        st.markdown("### 🎛️ System Controls")
        
        # Farm selection
        farm_name = st.selectbox(
            "Select Farm",
            ["Green Valley Farm", "Sunrise Agriculture", "Highland Crops"]
        )
        
        # Date range selector
        date_range = st.date_input(
            "Data Date Range",
            value=[datetime.now().date() - timedelta(days=7), datetime.now().date()],
            max_value=datetime.now().date()
        )
        
        st.markdown("---")
        st.markdown("### 📊 Data Sources")
        
        # Data source status
        data_sources = {
            "🌡️ Temperature Sensors": "online",
            "💧 Moisture Sensors": "online", 
            "📱 Manual Entries": "active",
            "🌐 Weather API": "online",
            "💾 Database": "healthy"
        }
        
        for source, status in data_sources.items():
            if status == "online" or status == "active" or status == "healthy":
                st.markdown(f"{source} <span class='status-good'>●</span>", unsafe_allow_html=True)
            else:
                st.markdown(f"{source} <span class='status-alert'>●</span>", unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### ⚙️ Quick Settings")
        
        # Auto-refresh toggle
        auto_refresh = st.checkbox("Auto-refresh data", value=True)
        
        # Alert preferences
        alert_threshold = st.slider("Alert Sensitivity", 1, 5, 3)
        
        # Export options
        if st.button("📥 Export Today's Data"):
            st.success("Data exported successfully!")

def main():
    """Main application function"""
    
    # Initialize sample data
    init_sample_data()
    
    # Setup sidebar
    setup_sidebar()
    
    # Display main dashboard
    display_overview_dashboard()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p>AgriTech Management System v1.0 | Last Updated: {}</p>
    </div>
    """.format(datetime.now().strftime("%Y-%m-%d %H:%M")), unsafe_allow_html=True)

if __name__ == "__main__":
    main()
