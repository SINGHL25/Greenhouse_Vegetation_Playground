# utils/visual_helpers.py
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import base64
import io

class VisualHelpers:
    """
    Utility class for creating consistent visualizations across the AgriTech system
    """
    
    def __init__(self):
        self.color_schemes = {
            'green_gradient': ['#E8F5E8', '#C8E6C9', '#A5D6A7', '#81C784', '#66BB6A', '#4CAF50', '#43A047', '#388E3C', '#2E7D32', '#1B5E20'],
            'blue_gradient': ['#E3F2FD', '#BBDEFB', '#90CAF9', '#64B5F6', '#42A5F5', '#2196F3', '#1E88E5', '#1976D2', '#1565C0', '#0D47A1'],
            'earth_tones': ['#8D6E63', '#A1887F', '#BCAAA4', '#D7CCC8', '#EFEBE9', '#6D4C41', '#5D4037', '#4E342E', '#3E2723'],
            'agriculture': ['#4CAF50', '#8BC34A', '#CDDC39', '#FF9800', '#795548', '#607D8B'],
            'financial': ['#2E7D32', '#388E3C', '#43A047', '#4CAF50', '#66BB6A', '#81C784'],
            'alerts': ['#4CAF50', '#FF9800', '#F44336']
        }
        
        self.icons = {
            'temperature': '🌡️',
            'humidity': '💧',
            'ph': '🧪',
            'moisture': '🌱',
            'light': '☀️',
            'co2': '💨',
            'money': '💰',
            'profit': '📈',
            'loss': '📉',
            'warning': '⚠️',
            'success': '✅',
            'error': '❌',
            'info': 'ℹ️'
        }
    
    def create_metric_card(self, title, value, delta=None, icon=None, color_scheme='green'):
        """Create a styled metric card"""
        colors = {
            'green': '#4CAF50',
            'blue': '#2196F3', 
            'orange': '#FF9800',
            'red': '#F44336',
            'purple': '#9C27B0'
        }
        
        card_color = colors.get(color_scheme, colors['green'])
        icon_display = icon if icon else ''
        delta_display = f"<small>{delta}</small>" if delta else ""
        
        card_html = f"""
        <div style="
            background: linear-gradient(135deg, {card_color} 0%, {card_color}dd 100%);
            padding: 1.5rem;
            border-radius: 12px;
            color: white;
            margin-bottom: 1rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        ">
            <h3>{icon_display} {title}</h3>
            <h2>{value}</h2>
            {delta_display}
        </div>
        """
        
        return card_html
    
    def create_gauge_chart(self, value, min_val=0, max_val=100, title="", 
                          optimal_range=None, color_scheme='green'):
        """Create a gauge chart for displaying metrics with optimal ranges"""
        
        # Determine color based on optimal range
        if optimal_range:
            min_opt, max_opt = optimal_range
            if min_opt <= value <= max_opt:
                gauge_color = '#4CAF50'  # Green for optimal
            elif value < min_opt * 0.8 or value > max_opt * 1.2:
                gauge_color = '#F44336'  # Red for critical
            else:
                gauge_color = '#FF9800'  # Orange for warning
        else:
            gauge_color = self.color_schemes[f'{color_scheme}_gradient'][5]
        
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = value,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': title},
            gauge = {
                'axis': {'range': [None, max_val]},
                'bar': {'color': gauge_color},
                'steps': [
                    {'range': [min_val, max_val * 0.6], 'color': "lightgray"},
                    {'range': [max_val * 0.6, max_val * 0.9], 'color': "gray"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': max_val * 0.9
                }
            }
        ))
        
        if optimal_range:
            # Add optimal range indicators
            min_opt, max_opt = optimal_range
            fig.add_shape(
                type="rect",
                x0=0, y0=min_opt/max_val, x1=1, y1=max_opt/max_val,
                fillcolor="green", opacity=0.2,
                line=dict(color="green", width=2)
            )
        
        fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
        
        return fig
    
    def create_time_series_chart(self, data, x_col, y_cols, title="", 
                                colors=None, show_alerts=False):
        """Create a multi-line time series chart with optional alert zones"""
        
        if not colors:
            colors = self.color_schemes['agriculture'][:len(y_cols)]
        
        fig = go.Figure()
        
        for i, y_col in enumerate(y_cols):
            fig.add_trace(go.Scatter(
                x=data[x_col],
                y=data[y_col],
                mode='lines+markers',
                name=y_col.replace('_', ' ').title(),
                line=dict(color=colors[i % len(colors)], width=2),
                marker=dict(size=4)
            ))
        
        if show_alerts:
            # Add alert zones (example thresholds)
            fig.add_hline(y=30, line_dash="dash", line_color="red", 
                         annotation_text="Critical High", opacity=0.7)
            fig.add_hline(y=10, line_dash="dash", line_color="red", 
                         annotation_text="Critical Low", opacity=0.7)
        
        fig.update_layout(
            title=title,
            xaxis_title=x_col.replace('_', ' ').title(),
            yaxis_title="Value",
            hovermode='x unified',
            height=400,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        return fig
    
    def create_correlation_heatmap(self, data, title="Correlation Matrix"):
        """Create a correlation heatmap for numerical columns"""
        
        # Select only numerical columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        correlation_matrix = data[numeric_cols].corr()
        
        fig = px.imshow(
            correlation_matrix,
            text_auto=True,
            aspect="auto",
            title=title,
            color_continuous_scale="RdBu_r",
            range_color=[-1, 1]
        )
        
        fig.update_layout(height=500)
        
        return fig
    
    def create_distribution_chart(self, data, column, title="", chart_type='histogram'):
        """Create distribution charts (histogram, box plot, or violin plot)"""
        
        if chart_type == 'histogram':
            fig = px.histogram(
                data, 
                x=column, 
                title=title or f'Distribution of {column.replace("_", " ").title()}',
                nbins=30,
                color_discrete_sequence=['#4CAF50']
            )
            
        elif chart_type == 'box':
            fig = px.box(
                data, 
                y=column, 
                title=title or f'Box Plot of {column.replace("_", " ").title()}',
                color_discrete_sequence=['#4CAF50']
            )
            
        elif chart_type == 'violin':
            fig = px.violin(
                data, 
                y=column, 
                box=True,
                title=title or f'Violin Plot of {column.replace("_", " ").title()}',
                color_discrete_sequence=['#4CAF50']
            )
        
        fig.update_layout(height=400)
        
        return fig
    
    def create_comparison_chart(self, data, categories, values, title="", 
                              chart_type='bar', orientation='v'):
        """Create comparison charts (bar, horizontal bar, or pie)"""
        
        if chart_type == 'bar':
            if orientation == 'h':
                fig = px.bar(
                    x=values, 
                    y=categories, 
                    orientation='h',
                    title=title,
                    color=values,
                    color_continuous_scale='Greens'
                )
            else:
                fig = px.bar(
                    x=categories, 
                    y=values,
                    title=title,
                    color=values,
                    color_continuous_scale='Greens'
                )
                
        elif chart_type == 'pie':
            fig = px.pie(
                values=values,
                names=categories,
                title=title,
                color_discrete_sequence=self.color_schemes['agriculture']
            )
            
        fig.update_layout(height=400)
        
        return fig
    
    def create_geographic_chart(self, data, lat_col, lon_col, size_col=None, 
                              color_col=None, title="Farm Locations"):
        """Create a map visualization for farm locations"""
        
        fig = px.scatter_mapbox(
            data,
            lat=lat_col,
            lon=lon_col,
            size=size_col,
            color=color_col,
            hover_name=data.index if 'name' not in data.columns else 'name',
            title=title,
            mapbox_style="open-street-map",
            height=500
        )
        
        fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
        
        return fig
    
    def create_performance_dashboard(self, kpis_data, title="Performance Dashboard"):
        """Create a comprehensive KPI dashboard"""
        
        # Create subplot structure
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=list(kpis_data.keys())[:6],
            specs=[[{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}],
                   [{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}]]
        )
        
        positions = [(1,1), (1,2), (1,3), (2,1), (2,2), (2,3)]
        
        for i, (kpi_name, kpi_data) in enumerate(list(kpis_data.items())[:6]):
            row, col = positions[i]
            
            # Determine color based on performance
            if 'target' in kpi_data and 'value' in kpi_data:
                performance = kpi_data['value'] / kpi_data['target']
                if performance >= 1.0:
                    color = "#4CAF50"
                elif performance >= 0.8:
                    color = "#FF9800" 
                else:
                    color = "#F44336"
            else:
                color = "#2196F3"
            
            fig.add_trace(
                go.Indicator(
                    mode = "number+gauge",
                    value = kpi_data.get('value', 0),
                    title = {"text": kpi_name},
                    gauge = {'bar': {'color': color}},
                    domain = {'row': row-1, 'column': col-1}
                ),
                row=row, col=col
            )
        
        fig.update_layout(
            height=500,
            title_text=title,
            title_x=0.5
        )
        
        return fig
    
    def create_alert_timeline(self, alerts_data, title="Alert Timeline"):
        """Create a timeline visualization for alerts"""
        
        color_map = {
            'critical': '#F44336',
            'warning': '#FF9800', 
            'info': '#2196F3',
            'resolved': '#4CAF50'
        }
        
        fig = px.scatter(
            alerts_data,
            x='timestamp',
            y='alert_type',
            color='severity',
            size='impact_score' if 'impact_score' in alerts_data.columns else None,
            hover_data=['message'],
            title=title,
            color_discrete_map=color_map
        )
        
        fig.update_layout(height=400)
        
        return fig
    
    def export_chart_as_image(self, fig, filename="chart", format="png"):
        """Export plotly chart as image"""
        
        # Convert plotly figure to image
        img_bytes = fig.to_image(format=format, width=1200, height=600, scale=2)
        
        # Create download link
        b64 = base64.b64encode(img_bytes).decode()
        href = f'<a href="data:image/{format};base64,{b64}" download="{filename}.{format}">Download Chart</a>'
        
        return href
    
    def create_data_quality_report(self, data, title="Data Quality Report"):
        """Generate a data quality visualization"""
        
        quality_metrics = {}
        
        for column in data.columns:
            missing_pct = (data[column].isnull().sum() / len(data)) * 100
            
            if data[column].dtype in ['int64', 'float64']:
                # For numerical columns
                outliers = len(data[(np.abs(data[column] - data[column].mean()) > 3 * data[column].std())])
                outlier_pct = (outliers / len(data)) * 100
                
                quality_metrics[column] = {
                    'missing_pct': missing_pct,
                    'outlier_pct': outlier_pct,
                    'quality_score': 100 - missing_pct - outlier_pct
                }
            else:
                # For categorical columns
                unique_ratio = data[column].nunique() / len(data)
                
                quality_metrics[column] = {
                    'missing_pct': missing_pct,
                    'unique_ratio': unique_ratio * 100,
                    'quality_score': 100 - missing_pct
                }
        
        # Create visualization
        df_quality = pd.DataFrame(quality_metrics).T
        
        fig = px.bar(
            df_quality,
            y=df_quality.index,
            x='quality_score',
            title=title,
            color='quality_score',
            color_continuous_scale='RdYlGn',
            orientation='h'
        )
        
        fig.update_layout(height=400)
        
        return fig, df_quality
    
    def create_seasonal_analysis(self, data, date_col, value_col, title="Seasonal Analysis"):
        """Create seasonal decomposition and analysis"""
        
        # Ensure datetime
        data[date_col] = pd.to_datetime(data[date_col])
        data_sorted = data.sort_values(date_col)
        
        # Add seasonal components
        data_sorted['month'] = data_sorted[date_col].dt.month
        data_sorted['quarter'] = data_sorted[date_col].dt.quarter
        data_sorted['year'] = data_sorted[date_col].dt.year
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Monthly Average', 'Quarterly Trends', 
                           'Year-over-Year', 'Seasonal Pattern'],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Monthly averages
        monthly_avg = data_sorted.groupby('month')[value_col].mean()
        fig.add_trace(
            go.Bar(x=monthly_avg.index, y=monthly_avg.values, name='Monthly Avg'),
            row=1, col=1
        )
        
        # Quarterly trends
        quarterly_avg = data_sorted.groupby('quarter')[value_col].mean()
        fig.add_trace(
            go.Scatter(x=quarterly_avg.index, y=quarterly_avg.values, 
                      mode='lines+markers', name='Quarterly'),
            row=1, col=2
        )
        
        # Year-over-year
        yearly_avg = data_sorted.groupby('year')[value_col].mean()
        fig.add_trace(
            go.Scatter(x=yearly_avg.index, y=yearly_avg.values, 
                      mode='lines+markers', name='Yearly'),
            row=2, col=1
        )
        
        # Seasonal pattern (by month across all years)
        seasonal_pattern = data_sorted.groupby('month')[value_col].agg(['mean', 'std'])
        fig.add_trace(
            go.Scatter(x=seasonal_pattern.index, y=seasonal_pattern['mean'], 
                      mode='lines', name='Seasonal Mean'),
            row=2, col=2
        )
        
        # Add confidence interval
        fig.add_trace(
            go.Scatter(
                x=seasonal_pattern.index.tolist() + seasonal_pattern.index.tolist()[::-1],
                y=(seasonal_pattern['mean'] + seasonal_pattern['std']).tolist() + 
                  (seasonal_pattern['mean'] - seasonal_pattern['std']).tolist()[::-1],
                fill='tonexty',
                fillcolor='rgba(0,100,80,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='Confidence Interval',
                showlegend=False
            ),
            row=2, col=2
        )
        
        fig.update_layout(height=600, title_text=title, showlegend=False)
        
        return fig
    
    def create_growth_projection(self, historical_data, projection_periods=12, 
                               growth_rate=0.05, title="Growth Projection"):
        """Create growth projection chart with confidence intervals"""
        
        # Calculate trend from historical data
        if len(historical_data) > 1:
            x_vals = np.arange(len(historical_data))
            coeffs = np.polyfit(x_vals, historical_data, 1)
            trend_growth = coeffs[0]
        else:
            trend_growth = growth_rate
        
        # Generate projections
        last_value = historical_data.iloc[-1] if isinstance(historical_data, pd.Series) else historical_data[-1]
        projections = []
        
        for i in range(1, projection_periods + 1):
            # Add some randomness for realistic projections
            noise = np.random.normal(0, 0.02)  # 2% standard deviation
            projected_value = last_value * (1 + growth_rate + noise) ** i
            projections.append(projected_value)
        
        # Create date range
        if isinstance(historical_data, pd.Series) and hasattr(historical_data.index, 'dtype'):
            last_date = historical_data.index[-1]
            future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), 
                                       periods=projection_periods, freq='M')
            historical_dates = historical_data.index
        else:
            historical_dates = range(len(historical_data))
            future_dates = range(len(historical_data), len(historical_data) + projection_periods)
        
        # Create chart
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(go.Scatter(
            x=historical_dates,
            y=historical_data,
            mode='lines+markers',
            name='Historical',
            line=dict(color='#2196F3', width=3)
        ))
        
        # Projected data
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=projections,
            mode='lines+markers',
            name='Projected',
            line=dict(color='#4CAF50', width=3, dash='dash')
        ))
        
        # Confidence intervals
        upper_bound = [p * 1.1 for p in projections]
        lower_bound = [p * 0.9 for p in projections]
        
        fig.add_trace(go.Scatter(
            x=list(future_dates) + list(future_dates)[::-1],
            y=upper_bound + lower_bound[::-1],
            fill='tonexty',
            fillcolor='rgba(76, 175, 80, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='Confidence Interval',
            showlegend=False
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Period',
            yaxis_title='Value',
            height=400
        )
        
        return fig
    
    def create_comparative_analysis(self, data_dict, title="Comparative Analysis"):
        """Create comparative analysis across multiple datasets"""
        
        fig = go.Figure()
        colors = self.color_schemes['agriculture']
        
        for i, (name, data) in enumerate(data_dict.items()):
            fig.add_trace(go.Scatter(
                x=data.index if hasattr(data, 'index') else range(len(data)),
                y=data.values if hasattr(data, 'values') else data,
                mode='lines+markers',
                name=name,
                line=dict(color=colors[i % len(colors)], width=2)
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Period',
            yaxis_title='Value',
            height=400,
            hovermode='x unified'
        )
        
        return fig
    
    def create_efficiency_analysis(self, input_data, output_data, title="Efficiency Analysis"):
        """Create efficiency analysis (input vs output)"""
        
        # Calculate efficiency ratios
        efficiency_ratios = output_data / input_data
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=['Input vs Output', 'Efficiency Ratio'],
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Input vs Output scatter plot
        fig.add_trace(
            go.Scatter(x=input_data, y=output_data, mode='markers',
                      name='Actual', marker=dict(color='#4CAF50')),
            row=1, col=1
        )
        
        # Add trend line
        z = np.polyfit(input_data, output_data, 1)
        p = np.poly1d(z)
        fig.add_trace(
            go.Scatter(x=input_data, y=p(input_data), mode='lines',
                      name='Trend', line=dict(color='#F44336', dash='dash')),
            row=1, col=1
        )
        
        # Efficiency ratio over time
        fig.add_trace(
            go.Scatter(x=range(len(efficiency_ratios)), y=efficiency_ratios,
                      mode='lines+markers', name='Efficiency',
                      line=dict(color='#2196F3')),
            row=1, col=2
        )
        
        fig.update_layout(height=400, title_text=title)
        
        return fig
    
    def format_currency(self, amount, currency='):
        """Format currency with proper separators"""
        return f"{currency}{amount:,.2f}"
    
    def format_percentage(self, value, decimal_places=1):
        """Format percentage with specified decimal places"""
        return f"{value:.{decimal_places}f}%"
    
    def format_large_numbers(self, value):
        """Format large numbers with K, M, B suffixes"""
        if abs(value) >= 1e9:
            return f"{value/1e9:.1f}B"
        elif abs(value) >= 1e6:
            return f"{value/1e6:.1f}M"
        elif abs(value) >= 1e3:
            return f"{value/1e3:.1f}K"
        else:
            return f"{value:.0f}"
    
    def get_status_color(self, value, thresholds):
        """Get color based on value and thresholds"""
        if 'critical_low' in thresholds and value < thresholds['critical_low']:
            return '#F44336'  # Red
        elif 'warning_low' in thresholds and value < thresholds['warning_low']:
            return '#FF9800'  # Orange
        elif 'critical_high' in thresholds and value > thresholds['critical_high']:
            return '#F44336'  # Red
        elif 'warning_high' in thresholds and value > thresholds['warning_high']:
            return '#FF9800'  # Orange
        else:
            return '#4CAF50'  # Green
    
    def create_summary_table(self, data, title="Summary Statistics"):
        """Create a formatted summary statistics table"""
        
        summary_stats = data.describe()
        
        # Format the data for better display
        formatted_stats = summary_stats.round(2)
        
        # Create a Plotly table
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=['Statistic'] + list(formatted_stats.columns),
                fill_color='#4CAF50',
                font=dict(color='white', size=12),
                align='left'
            ),
            cells=dict(
                values=[formatted_stats.index] + [formatted_stats[col] for col in formatted_stats.columns],
                fill_color='#E8F5E8',
                align='left',
                font_size=11
            )
        )])
        
        fig.update_layout(
            title=title,
            height=400,
            margin=dict(l=0, r=0, t=40, b=0)
        )
        
        return fig

# Convenience functions for quick access
def create_quick_metric(title, value, delta=None, color='green'):
    """Quick metric card creation"""
    helper = VisualHelpers()
    return helper.create_metric_card(title, value, delta, color_scheme=color)

def create_quick_chart(data, x_col, y_col, chart_type='line', title=""):
    """Quick chart creation"""
    helper = VisualHelpers()
    
    if chart_type == 'line':
        return helper.create_time_series_chart(data, x_col, [y_col], title)
    elif chart_type == 'bar':
        return helper.create_comparison_chart(data[x_col], data[y_col], title, 'bar')
    elif chart_type == 'scatter':
        fig = px.scatter(data, x=x_col, y=y_col, title=title)
        return fig

def export_data_to_csv(data, filename="export_data"):
    """Export data to CSV with download link"""
    csv = data.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}.csv">📥 Download CSV</a>'
    return href

# Example usage and testing
if __name__ == "__main__":
    # Test the visual helpers
    helper = VisualHelpers()
    
    # Create sample data for testing
    sample_data = pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=100, freq='D'),
        'temperature': np.random.normal(25, 3, 100),
        'humidity': np.random.normal(65, 10, 100),
        'yield': np.random.normal(1000, 200, 100)
    })
    
    print("Visual Helpers initialized successfully!")
    print("Available color schemes:", list(helper.color_schemes.keys()))
    print("Available icons:", list(helper.icons.keys()))
