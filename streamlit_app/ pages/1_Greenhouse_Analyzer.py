
# pages/1_Greenhouse_Analyzer.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import sqlite3

# Page configuration
st.set_page_config(
    page_title="Greenhouse Analyzer",
    page_icon="🏠",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 1rem;
    }
    
    .alert-card {
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        border-left: 4px solid;
    }
    
    .alert-critical { border-left-color: #f44336; background-color: #ffebee; }
    .alert-warning { border-left-color: #ff9800; background-color: #fff3e0; }
    .alert-good { border-left-color: #4caf50; background-color: #e8f5e8; }
    
    .status-indicator {
        width: 12px;
        height: 12px;
        border-radius: 50%;
        display: inline-block;
        margin-right: 8px;
    }
    
    .status-online { background-color: #4caf50; }
    .status-warning { background-color: #ff9800; }
    .status-offline { background-color: #f44336; }
</style>
""", unsafe_allow_html=True)

class GreenhouseAnalyzer:
    def __init__(self):
        self.load_data()
    
    def load_data(self):
        """Load greenhouse data from database or generate sample data"""
        if 'greenhouse_data' not in st.session_state:
            # Generate comprehensive sample data for multiple greenhouses
            self.generate_sample_data()
    
    def generate_sample_data(self):
        """Generate realistic sample data for demonstration"""
        # Generate data for the last 30 days, hourly readings
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        date_range = pd.date_range(start=start_date, end=end_date, freq='H')
        
        greenhouses = ['GH001', 'GH002', 'GH003']
        all_data = []
        
        for gh_id in greenhouses:
            # Base parameters for each greenhouse (slight variations)
            base_temp = 24 + (hash(gh_id) % 3)
            base_humidity = 65 + (hash(gh_id) % 10)
            base_ph = 6.5 + (hash(gh_id) % 3) * 0.1
            
            for timestamp in date_range:
                # Add daily and seasonal variations
                hour_factor = np.sin(2 * np.pi * timestamp.hour / 24) * 3
                day_factor = np.sin(2 * np.pi * timestamp.timetuple().tm_yday / 365) * 2
                
                # Add some realistic noise and trends
                temp_noise = np.random.normal(0, 1.5)
                humidity_noise = np.random.normal(0, 5)
                ph_noise = np.random.normal(0, 0.2)
                
                data_point = {
                    'timestamp': timestamp,
                    'greenhouse_id': gh_id,
                    'temperature': base_temp + hour_factor + day_factor + temp_noise,
                    'humidity': base_humidity + humidity_noise - hour_factor * 2,
                    'soil_ph': base_ph + ph_noise,
                    'soil_moisture': 45 + np.random.normal(0, 8),
                    'light_intensity': max(0, 800 + np.sin(2 * np.pi * timestamp.hour / 24) * 600 + np.random.normal(0, 100)),
                    'co2_level': 400 + np.random.normal(0, 30),
                    'soil_ec': 1.2 + np.random.normal(0, 0.3),  # Electrical conductivity
                    'leaf_wetness': np.random.uniform(0, 100),
                    'par_light': max(0, 400 + np.sin(2 * np.pi * timestamp.hour / 24) * 300 + np.random.normal(0, 50))
                }
                all_data.append(data_point)
        
        st.session_state.greenhouse_data = pd.DataFrame(all_data)
        
        # Generate alert data
        self.generate_alert_data()
    
    def generate_alert_data(self):
        """Generate sample alert data"""
        alerts = [
            {
                'timestamp': datetime.now() - timedelta(hours=2),
                'greenhouse_id': 'GH002',
                'alert_type': 'temperature',
                'severity': 'warning',
                'message': 'Temperature above optimal range (32.5°C)',
                'current_value': 32.5,
                'threshold': 30.0,
                'resolved': False
            },
            {
                'timestamp': datetime.now() - timedelta(hours=5),
                'greenhouse_id': 'GH001',
                'alert_type': 'humidity',
                'severity': 'critical',
                'message': 'Humidity critically low (25%)',
                'current_value': 25.0,
                'threshold': 40.0,
                'resolved': True
            },
            {
                'timestamp': datetime.now() - timedelta(days=1),
                'greenhouse_id': 'GH003',
                'alert_type': 'soil_ph',
                'severity': 'warning',
                'message': 'Soil pH outside optimal range (5.8)',
                'current_value': 5.8,
                'threshold': 6.0,
                'resolved': False
            }
        ]
        st.session_state.alerts_data = pd.DataFrame(alerts)
    
    def calculate_optimal_ranges(self, crop_type="tomato"):
        """Define optimal ranges for different parameters based on crop type"""
        ranges = {
            "tomato": {
                "temperature": (18, 26),
                "humidity": (60, 80),
                "soil_ph": (6.0, 6.8),
                "soil_moisture": (40, 60),
                "co2_level": (300, 500),
                "light_intensity": (600, 1000)
            },
            "lettuce": {
                "temperature": (15, 23),
                "humidity": (50, 70),
                "soil_ph": (6.0, 7.0),
                "soil_moisture": (35, 55),
                "co2_level": (350, 450),
                "light_intensity": (300, 600)
            },
            "pepper": {
                "temperature": (20, 28),
                "humidity": (60, 75),
                "soil_ph": (6.2, 7.0),
                "soil_moisture": (45, 65),
                "co2_level": (400, 600),
                "light_intensity": (700, 1200)
            }
        }
        return ranges.get(crop_type, ranges["tomato"])
    
    def create_environmental_dashboard(self, greenhouse_id, date_range):
        """Create comprehensive environmental monitoring dashboard"""
        data = st.session_state.greenhouse_data
        filtered_data = data[
            (data['greenhouse_id'] == greenhouse_id) &
            (data['timestamp'] >= pd.Timestamp(date_range[0])) &
            (data['timestamp'] <= pd.Timestamp(date_range[1]))
        ].copy()
        
        if filtered_data.empty:
            st.warning("No data available for selected greenhouse and date range.")
            return
        
        # Current status metrics
        latest_data = filtered_data.iloc[-1]
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            temp_status = self.get_status_color(latest_data['temperature'], 18, 26)
            st.markdown(f"""
            <div class="metric-card">
                <div class="status-indicator status-{temp_status}"></div>
                <h3>🌡️ Temperature</h3>
                <h2>{latest_data['temperature']:.1f}°C</h2>
                <small>Range: 18-26°C</small>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            humidity_status = self.get_status_color(latest_data['humidity'], 60, 80)
            st.markdown(f"""
            <div class="metric-card">
                <div class="status-indicator status-{humidity_status}"></div>
                <h3>💧 Humidity</h3>
                <h2>{latest_data['humidity']:.1f}%</h2>
                <small>Range: 60-80%</small>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            ph_status = self.get_status_color(latest_data['soil_ph'], 6.0, 6.8)
            st.markdown(f"""
            <div class="metric-card">
                <div class="status-indicator status-{ph_status}"></div>
                <h3>🧪 Soil pH</h3>
                <h2>{latest_data['soil_ph']:.1f}</h2>
                <small>Range: 6.0-6.8</small>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            moisture_status = self.get_status_color(latest_data['soil_moisture'], 40, 60)
            st.markdown(f"""
            <div class="metric-card">
                <div class="status-indicator status-{moisture_status}"></div>
                <h3>🌱 Soil Moisture</h3>
                <h2>{latest_data['soil_moisture']:.1f}%</h2>
                <small>Range: 40-60%</small>
            </div>
            """, unsafe_allow_html=True)
        
        # Time series charts
        st.subheader("📊 Environmental Trends")
        
        # Create comprehensive multi-parameter chart
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Temperature (°C)', 'Humidity (%)', 
                'Soil pH', 'Soil Moisture (%)',
                'Light Intensity (µmol/m²/s)', 'CO₂ Level (ppm)'
            ),
            vertical_spacing=0.08,
            horizontal_spacing=0.08
        )
        
        # Temperature
        fig.add_trace(
            go.Scatter(x=filtered_data['timestamp'], y=filtered_data['temperature'],
                      name='Temperature', line=dict(color='#FF6B6B', width=2)),
            row=1, col=1
        )
        fig.add_hline(y=18, line_dash="dash", line_color="blue", opacity=0.5, row=1, col=1)
        fig.add_hline(y=26, line_dash="dash", line_color="blue", opacity=0.5, row=1, col=1)
        
        # Humidity
        fig.add_trace(
            go.Scatter(x=filtered_data['timestamp'], y=filtered_data['humidity'],
                      name='Humidity', line=dict(color='#4ECDC4', width=2)),
            row=1, col=2
        )
        fig.add_hline(y=60, line_dash="dash", line_color="blue", opacity=0.5, row=1, col=2)
        fig.add_hline(y=80, line_dash="dash", line_color="blue", opacity=0.5, row=1, col=2)
        
        # Soil pH
        fig.add_trace(
            go.Scatter(x=filtered_data['timestamp'], y=filtered_data['soil_ph'],
                      name='Soil pH', line=dict(color='#45B7D1', width=2)),
            row=2, col=1
        )
        fig.add_hline(y=6.0, line_dash="dash", line_color="blue", opacity=0.5, row=2, col=1)
        fig.add_hline(y=6.8, line_dash="dash", line_color="blue", opacity=0.5, row=2, col=1)
        
        # Soil Moisture
        fig.add_trace(
            go.Scatter(x=filtered_data['timestamp'], y=filtered_data['soil_moisture'],
                      name='Soil Moisture', line=dict(color='#96CEB4', width=2)),
            row=2, col=2
        )
        fig.add_hline(y=40, line_dash="dash", line_color="blue", opacity=0.5, row=2, col=2)
        fig.add_hline(y=60, line_dash="dash", line_color="blue", opacity=0.5, row=2, col=2)
        
        # Light Intensity
        fig.add_trace(
            go.Scatter(x=filtered_data['timestamp'], y=filtered_data['light_intensity'],
                      name='Light Intensity', line=dict(color='#FFA726', width=2)),
            row=3, col=1
        )
        
        # CO2 Level
        fig.add_trace(
            go.Scatter(x=filtered_data['timestamp'], y=filtered_data['co2_level'],
                      name='CO₂ Level', line=dict(color='#AB47BC', width=2)),
            row=3, col=2
        )
        
        fig.update_layout(
            height=800,
            showlegend=False,
            title_text=f"Environmental Parameters - {greenhouse_id}",
            title_x=0.5
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        return filtered_data
    
    def get_status_color(self, value, min_optimal, max_optimal):
        """Determine status color based on optimal range"""
        if min_optimal <= value <= max_optimal:
            return "online"
        elif min_optimal * 0.9 <= value <= max_optimal * 1.1:
            return "warning"
        else:
            return "offline"
    
    def create_correlation_analysis(self, data):
        """Create correlation analysis between environmental parameters"""
        st.subheader("🔗 Parameter Correlation Analysis")
        
        # Calculate correlation matrix
        numeric_cols = ['temperature', 'humidity', 'soil_ph', 'soil_moisture', 'light_intensity', 'co2_level']
        correlation_matrix = data[numeric_cols].corr()
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Create correlation heatmap
            fig = px.imshow(
                correlation_matrix,
                text_auto=True,
                aspect="auto",
                title="Environmental Parameter Correlations",
                color_continuous_scale="RdBu_r"
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 📈 Key Insights")
            
            # Find strongest correlations
            correlations = []
            for i, col1 in enumerate(numeric_cols):
                for j, col2 in enumerate(numeric_cols):
                    if i < j:
                        corr_value = correlation_matrix.loc[col1, col2]
                        correlations.append({
                            'pair': f"{col1} - {col2}",
                            'correlation': abs(corr_value),
                            'direction': 'positive' if corr_value > 0 else 'negative'
                        })
            
            # Sort by strength
            correlations.sort(key=lambda x: x['correlation'], reverse=True)
            
            for i, corr in enumerate(correlations[:3]):
                direction_icon = "📈" if corr['direction'] == 'positive' else "📉"
                st.markdown(f"{direction_icon} **{corr['pair']}**")
                st.markdown(f"Correlation: {corr['correlation']:.2f}")
                st.markdown("---")
    
    def create_alert_system(self, greenhouse_id):
        """Create alert monitoring system"""
        st.subheader("🚨 Alert System")
        
        if 'alerts_data' in st.session_state:
            alerts = st.session_state.alerts_data
            greenhouse_alerts = alerts[alerts['greenhouse_id'] == greenhouse_id]
            
            # Active alerts
            active_alerts = greenhouse_alerts[~greenhouse_alerts['resolved']]
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                if not active_alerts.empty:
                    st.markdown("### Active Alerts")
                    for _, alert in active_alerts.iterrows():
                        severity_class = f"alert-{alert['severity']}" if alert['severity'] != 'critical' else "alert-critical"
                        
                        st.markdown(f"""
                        <div class="{severity_class}">
                            <strong>{alert['alert_type'].upper()}</strong> - {alert['severity'].upper()}
                            <br>{alert['message']}
                            <br><small>Current: {alert['current_value']} | Threshold: {alert['threshold']}</small>
                            <br><small>{alert['timestamp'].strftime('%Y-%m-%d %H:%M')}</small>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.success("✅ No active alerts for this greenhouse")
            
            with col2:
                st.markdown("### Alert Statistics")
                total_alerts = len(greenhouse_alerts)
                resolved_alerts = len(greenhouse_alerts[greenhouse_alerts['resolved']])
                
                st.metric("Total Alerts", total_alerts)
                st.metric("Resolved", resolved_alerts)
                st.metric("Active", total_alerts - resolved_alerts)
                
                # Alert by severity
                if not greenhouse_alerts.empty:
                    severity_counts = greenhouse_alerts['severity'].value_counts()
                    fig = px.pie(
                        values=severity_counts.values,
                        names=severity_counts.index,
                        title="Alerts by Severity"
                    )
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)

def main():
    st.title("🏠 Greenhouse Environmental Analyzer")
    st.markdown("Monitor and analyze environmental conditions across your greenhouse facilities")
    
    analyzer = GreenhouseAnalyzer()
    
    # Sidebar controls
    with st.sidebar:
        st.header("🎛️ Controls")
        
        # Greenhouse selection
        greenhouse_options = ['GH001', 'GH002', 'GH003']
        selected_greenhouse = st.selectbox(
            "Select Greenhouse",
            greenhouse_options,
            index=0
        )
        
        # Date range selection
        default_start = datetime.now().date() - timedelta(days=7)
        default_end = datetime.now().date()
        
        date_range = st.date_input(
            "Date Range",
            value=[default_start, default_end],
            max_value=datetime.now().date()
        )
        
        # Crop type for optimal ranges
        crop_type = st.selectbox(
            "Crop Type",
            ["tomato", "lettuce", "pepper"],
            index=0
        )
        
        st.markdown("---")
        
        # Refresh data button
        if st.button("🔄 Refresh Data"):
            # Clear cached data to regenerate
            if 'greenhouse_data' in st.session_state:
                del st.session_state.greenhouse_data
            st.success("Data refreshed!")
            st.rerun()
        
        # Export data button
        if st.button("📥 Export Data"):
            st.success("Data exported successfully!")
    
    # Main content
    if len(date_range) == 2:
        # Environmental dashboard
        filtered_data = analyzer.create_environmental_dashboard(selected_greenhouse, date_range)
        
        if filtered_data is not None and not filtered_data.empty:
            
            # Tabs for different analyses
            tab1, tab2, tab3, tab4 = st.tabs(["📊 Statistical Analysis", "🔗 Correlations", "🚨 Alerts", "📋 Data Table"])
            
            with tab1:
                st.subheader("📊 Statistical Summary")
                
                # Statistical analysis
                numeric_cols = ['temperature', 'humidity', 'soil_ph', 'soil_moisture', 'light_intensity', 'co2_level']
                stats = filtered_data[numeric_cols].describe()
                
                # Display statistics in a formatted way
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### Temperature & Humidity")
                    temp_stats = stats['temperature']
                    humidity_stats = stats['humidity']
                    
                    st.write(f"**Temperature:**")
                    st.write(f"• Average: {temp_stats['mean']:.1f}°C")
                    st.write(f"• Range: {temp_stats['min']:.1f}°C - {temp_stats['max']:.1f}°C")
                    st.write(f"• Standard Deviation: {temp_stats['std']:.1f}°C")
                    
                    st.write(f"**Humidity:**")
                    st.write(f"• Average: {humidity_stats['mean']:.1f}%")
                    st.write(f"• Range: {humidity_stats['min']:.1f}% - {humidity_stats['max']:.1f}%")
                    st.write(f"• Standard Deviation: {humidity_stats['std']:.1f}%")
                
                with col2:
                    st.markdown("### Soil Conditions")
                    ph_stats = stats['soil_ph']
                    moisture_stats = stats['soil_moisture']
                    
                    st.write(f"**Soil pH:**")
                    st.write(f"• Average: {ph_stats['mean']:.2f}")
                    st.write(f"• Range: {ph_stats['min']:.2f} - {ph_stats['max']:.2f}")
                    st.write(f"• Standard Deviation: {ph_stats['std']:.2f}")
                    
                    st.write(f"**Soil Moisture:**")
                    st.write(f"• Average: {moisture_stats['mean']:.1f}%")
                    st.write(f"• Range: {moisture_stats['min']:.1f}% - {moisture_stats['max']:.1f}%")
                    st.write(f"• Standard Deviation: {moisture_stats['std']:.1f}%")
            
            with tab2:
                analyzer.create_correlation_analysis(filtered_data)
            
            with tab3:
                analyzer.create_alert_system(selected_greenhouse)
            
            with tab4:
                st.subheader("📋 Raw Data")
                st.dataframe(
                    filtered_data.sort_values('timestamp', ascending=False),
                    use_container_width=True,
                    hide_index=True
                )
                
                # Download button for CSV
                csv = filtered_data.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name=f"{selected_greenhouse}_data_{date_range[0]}_{date_range[1]}.csv",
                    mime="text/csv"
                )

if __name__ == "__main__":
    main()
