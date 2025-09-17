
# pages/4_Economics_Dashboard.py
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
    page_title="Economics Dashboard",
    page_icon="💰",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .profit-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1rem;
    }
    
    .expense-card {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1rem;
    }
    
    .kpi-container {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
        text-align: center;
    }
    
    .cost-category {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #007bff;
        margin-bottom: 1rem;
    }
    
    .roi-positive { color: #28a745; font-weight: bold; }
    .roi-negative { color: #dc3545; font-weight: bold; }
    .roi-neutral { color: #6c757d; font-weight: bold; }
    
    .break-even-card {
        background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

class EconomicsDashboard:
    def __init__(self):
        self.load_data()
        self.init_cost_categories()
    
    def load_data(self):
        """Load economic data"""
        if 'economic_data' not in st.session_state:
            self.generate_sample_data()
    
    def init_cost_categories(self):
        """Initialize cost categories and their typical ranges"""
        self.cost_categories = {
            'Seeds & Seedlings': {
                'type': 'variable',
                'percentage_of_revenue': 0.15,
                'icon': '🌱',
                'description': 'Seeds, seedlings, and planting materials'
            },
            'Fertilizers': {
                'type': 'variable',
                'percentage_of_revenue': 0.12,
                'icon': '🧪',
                'description': 'Organic and synthetic fertilizers'
            },
            'Pesticides & Herbicides': {
                'type': 'variable',
                'percentage_of_revenue': 0.08,
                'icon': '🚫',
                'description': 'Plant protection chemicals'
            },
            'Water & Irrigation': {
                'type': 'variable',
                'percentage_of_revenue': 0.10,
                'icon': '💧',
                'description': 'Water costs and irrigation maintenance'
            },
            'Labor': {
                'type': 'variable',
                'percentage_of_revenue': 0.25,
                'icon': '👥',
                'description': 'Wages for farm workers'
            },
            'Equipment Fuel': {
                'type': 'variable',
                'percentage_of_revenue': 0.07,
                'icon': '⛽',
                'description': 'Fuel for tractors and equipment'
            },
            'Land Lease': {
                'type': 'fixed',
                'monthly_amount': 2000,
                'icon': '🏞️',
                'description': 'Land rental or lease payments'
            },
            'Equipment Maintenance': {
                'type': 'fixed',
                'monthly_amount': 800,
                'icon': '🔧',
                'description': 'Equipment repairs and maintenance'
            },
            'Insurance': {
                'type': 'fixed',
                'monthly_amount': 500,
                'icon': '🛡️',
                'description': 'Crop and liability insurance'
            },
            'Utilities': {
                'type': 'fixed',
                'monthly_amount': 600,
                'icon': '⚡',
                'description': 'Electricity and other utilities'
            }
        }
    
    def generate_sample_data(self):
        """Generate comprehensive economic sample data"""
        # Generate 24 months of data (12 historical + 12 projected)
        start_date = datetime.now().date() - timedelta(days=365)
        end_date = datetime.now().date() + timedelta(days=365)
        
        economic_records = []
        monthly_data = []
        
        current_date = start_date
        month_counter = 0
        
        while current_date <= end_date:
            month_start = current_date.replace(day=1)
            
            # Determine if this is historical or projected
            is_historical = current_date <= datetime.now().date()
            variation_factor = 1.0 if is_historical else np.random.uniform(0.9, 1.1)
            
            # Generate monthly revenue (seasonal variations)
            base_monthly_revenue = 25000
            seasonal_multiplier = 1 + 0.3 * np.sin(2 * np.pi * month_counter / 12)
            monthly_revenue = base_monthly_revenue * seasonal_multiplier * variation_factor
            
            # Generate income records
            revenue_sources = {
                'Crop Sales': monthly_revenue * 0.85,
                'Wholesale Contracts': monthly_revenue * 0.10,
                'Direct Sales': monthly_revenue * 0.05
            }
            
            total_monthly_costs = 0
            
            for source, amount in revenue_sources.items():
                economic_records.append({
                    'date': month_start,
                    'category': source,
                    'subcategory': 'Sales Revenue',
                    'amount': amount,
                    'type': 'income',
                    'description': f'{source} for {month_start.strftime("%B %Y")}',
                    'is_projected': not is_historical
                })
            
            # Generate expense records
            for category, info in self.cost_categories.items():
                if info['type'] == 'variable':
                    # Variable costs as percentage of revenue
                    cost_amount = monthly_revenue * info['percentage_of_revenue'] * np.random.uniform(0.8, 1.2)
                else:
                    # Fixed costs
                    cost_amount = info['monthly_amount'] * np.random.uniform(0.9, 1.1)
                
                total_monthly_costs += cost_amount
                
                economic_records.append({
                    'date': month_start,
                    'category': category,
                    'subcategory': 'Operating Expense',
                    'amount': cost_amount,
                    'type': 'expense',
                    'description': f'{category} for {month_start.strftime("%B %Y")}',
                    'is_projected': not is_historical
                })
            
            # Add some one-time expenses/income occasionally
            if np.random.random() < 0.2:  # 20% chance
                one_time_items = [
                    ('Equipment Purchase', 'expense', np.random.uniform(5000, 15000)),
                    ('Government Subsidy', 'income', np.random.uniform(2000, 8000)),
                    ('Insurance Claim', 'income', np.random.uniform(1000, 5000)),
                    ('Emergency Repairs', 'expense', np.random.uniform(1500, 4000))
                ]
                
                item_name, item_type, item_amount = np.random.choice(one_time_items)
                economic_records.append({
                    'date': month_start,
                    'category': item_name,
                    'subcategory': 'One-time' if item_type == 'expense' else 'Special Income',
                    'amount': item_amount,
                    'type': item_type,
                    'description': f'{item_name} in {month_start.strftime("%B %Y")}',
                    'is_projected': not is_historical
                })
                
                if item_type == 'expense':
                    total_monthly_costs += item_amount
            
            # Store monthly summary
            monthly_profit = monthly_revenue - total_monthly_costs
            monthly_data.append({
                'date': month_start,
                'revenue': monthly_revenue,
                'costs': total_monthly_costs,
                'profit': monthly_profit,
                'profit_margin': (monthly_profit / monthly_revenue) * 100 if monthly_revenue > 0 else 0,
                'is_projected': not is_historical
            })
            
            # Move to next month
            if current_date.month == 12:
                current_date = current_date.replace(year=current_date.year + 1, month=1)
            else:
                current_date = current_date.replace(month=current_date.month + 1)
            
            month_counter += 1
        
        # Store data
        st.session_state.economic_data = pd.DataFrame(economic_records)
        st.session_state.monthly_summary = pd.DataFrame(monthly_data)
        
        # Generate crop-specific profitability data
        self.generate_crop_profitability_data()
        
        # Generate ROI analysis data
        self.generate_roi_analysis_data()
    
    def generate_crop_profitability_data(self):
        """Generate crop-specific profitability analysis"""
        crops = ['Tomatoes', 'Lettuce', 'Peppers', 'Cucumber', 'Basil', 'Spinach']
        
        crop_data = []
        for crop in crops:
            # Generate quarterly data for each crop
            for quarter in range(1, 5):
                revenue = np.random.uniform(8000, 25000)
                costs = revenue * np.random.uniform(0.4, 0.7)
                area_harvested = np.random.uniform(1.5, 4.0)
                yield_per_ha = np.random.uniform(15000, 45000)
                
                crop_data.append({
                    'crop': crop,
                    'quarter': f'Q{quarter} 2024',
                    'revenue': revenue,
                    'costs': costs,
                    'profit': revenue - costs,
                    'area_hectares': area_harvested,
                    'yield_kg_per_ha': yield_per_ha,
                    'profit_per_ha': (revenue - costs) / area_harvested,
                    'profit_margin': ((revenue - costs) / revenue) * 100
                })
        
        st.session_state.crop_profitability = pd.DataFrame(crop_data)
    
    def generate_roi_analysis_data(self):
        """Generate ROI analysis data for different investments"""
        investments = [
            {
                'investment': 'Greenhouse Expansion',
                'initial_cost': 45000,
                'annual_revenue_increase': 18000,
                'annual_cost_increase': 6000,
                'payback_period': 3.75,
                'roi_5_year': 67.5
            },
            {
                'investment': 'Drip Irrigation System',
                'initial_cost': 12000,
                'annual_revenue_increase': 8000,
                'annual_cost_increase': 2000,
                'payback_period': 2.0,
                'roi_5_year': 125.0
            },
            {
                'investment': 'Automated Seeding Equipment',
                'initial_cost': 25000,
                'annual_revenue_increase': 5000,
                'annual_cost_increase': 1500,
                'payback_period': 7.1,
                'roi_5_year': 35.0
            },
            {
                'investment': 'Soil Health Program',
                'initial_cost': 8000,
                'annual_revenue_increase': 12000,
                'annual_cost_increase': 3000,
                'payback_period': 0.89,
                'roi_5_year': 281.25
            }
        ]
        
        st.session_state.roi_analysis = pd.DataFrame(investments)
    
    def create_financial_overview(self):
        """Create main financial overview dashboard"""
        st.subheader("💰 Financial Overview")
        
        monthly_data = st.session_state.monthly_summary
        current_data = monthly_data[~monthly_data['is_projected']]
        projected_data = monthly_data[monthly_data['is_projected']]
        
        # Current period metrics (last 12 months)
        last_12_months = current_data.tail(12)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_revenue = last_12_months['revenue'].sum()
            st.markdown(f"""
            <div class="metric-card">
                <h3>📈 Total Revenue (12M)</h3>
                <h2>${total_revenue:,.0f}</h2>
                <p>Average: ${total_revenue/12:,.0f}/month</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            total_costs = last_12_months['costs'].sum()
            st.markdown(f"""
            <div class="expense-card">
                <h3>💸 Total Costs (12M)</h3>
                <h2>${total_costs:,.0f}</h2>
                <p>Average: ${total_costs/12:,.0f}/month</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            total_profit = last_12_months['profit'].sum()
            profit_color = "profit-card" if total_profit >= 0 else "expense-card"
            st.markdown(f"""
            <div class="{profit_color}">
                <h3>🎯 Net Profit (12M)</h3>
                <h2>${total_profit:,.0f}</h2>
                <p>Margin: {(total_profit/total_revenue)*100:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            avg_margin = last_12_months['profit_margin'].mean()
            st.markdown(f"""
            <div class="metric-card">
                <h3>📊 Avg Profit Margin</h3>
                <h2>{avg_margin:.1f}%</h2>
                <p>Target: 25-35%</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Revenue and Profit Trends
        st.subheader("📈 Financial Trends")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Revenue vs Costs over time
            fig_trends = go.Figure()
            
            fig_trends.add_trace(go.Scatter(
                x=monthly_data['date'],
                y=monthly_data['revenue'],
                mode='lines+markers',
                name='Revenue',
                line=dict(color='#2E8B57', width=3)
            ))
            
            fig_trends.add_trace(go.Scatter(
                x=monthly_data['date'],
                y=monthly_data['costs'],
                mode='lines+markers',
                name='Costs',
                line=dict(color='#DC143C', width=3)
            ))
            
            # Add vertical line for current date
            fig_trends.add_vline(
                x=datetime.now(),
                line_dash="dash",
                line_color="gray",
                annotation_text="Current Date"
            )
            
            fig_trends.update_layout(
                title='Revenue vs Costs Trend',
                xaxis_title='Date',
                yaxis_title='Amount ($)',
                height=400
            )
            
            st.plotly_chart(fig_trends, use_container_width=True)
        
        with col2:
            # Profit margin trend
            fig_margin = px.line(
                monthly_data,
                x='date',
                y='profit_margin',
                title='Profit Margin Trend (%)',
                markers=True,
                color_discrete_sequence=['#4169E1']
            )
            
            # Add horizontal line for target margin
            fig_margin.add_hline(y=30, line_dash="dash", line_color="green", 
                                 annotation_text="Target: 30%")
            
            fig_margin.update_layout(height=400)
            st.plotly_chart(fig_margin, use_container_width=True)
    
    def create_cost_analysis(self):
        """Create detailed cost analysis"""
        st.subheader("💸 Cost Analysis")
        
        economic_data = st.session_state.economic_data
        expense_data = economic_data[economic_data['type'] == 'expense']
        
        # Time period selector
        col1, col2 = st.columns(2)
        
        with col1:
            analysis_period = st.selectbox(
                "Analysis Period",
                ["Last 3 Months", "Last 6 Months", "Last 12 Months", "All Data"]
            )
        
        with col2:
            cost_view = st.selectbox(
                "Cost View",
                ["By Category", "By Month", "Variable vs Fixed"]
            )
        
        # Filter data based on period
        if analysis_period == "Last 3 Months":
            cutoff_date = datetime.now().date() - timedelta(days=90)
        elif analysis_period == "Last 6 Months":
            cutoff_date = datetime.now().date() - timedelta(days=180)
        elif analysis_period == "Last 12 Months":
            cutoff_date = datetime.now().date() - timedelta(days=365)
        else:
            cutoff_date = expense_data['date'].min()
        
        filtered_expenses = expense_data[expense_data['date'] >= cutoff_date]
        
        if cost_view == "By Category":
            # Cost breakdown by category
            category_costs = filtered_expenses.groupby('category')['amount'].sum().sort_values(ascending=False)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                fig_category = px.bar(
                    x=category_costs.values,
                    y=category_costs.index,
                    orientation='h',
                    title=f'Costs by Category ({analysis_period})',
                    color=category_costs.values,
                    color_continuous_scale='Reds'
                )
                fig_category.update_layout(height=500)
                st.plotly_chart(fig_category, use_container_width=True)
            
            with col2:
                # Pie chart
                fig_pie = px.pie(
                    values=category_costs.values,
                    names=category_costs.index,
                    title='Cost Distribution'
                )
                fig_pie.update_layout(height=400)
                st.plotly_chart(fig_pie, use_container_width=True)
                
                # Cost summary
                st.markdown("### 📊 Cost Summary")
                total_costs = category_costs.sum()
                largest_cost = category_costs.index[0]
                largest_amount = category_costs.iloc[0]
                
                st.metric("Total Costs", f"${total_costs:,.0f}")
                st.metric("Largest Category", largest_cost)
                st.metric("Largest Amount", f"${largest_amount:,.0f}")
                st.metric("% of Total", f"{(largest_amount/total_costs)*100:.1f}%")
        
        elif cost_view == "By Month":
            # Monthly cost trends
            monthly_costs = filtered_expenses.groupby(filtered_expenses['date'].dt.to_period('M'))['amount'].sum()
            
            fig_monthly = px.line(
                x=monthly_costs.index.astype(str),
                y=monthly_costs.values,
                title='Monthly Cost Trends',
                markers=True
            )
            fig_monthly.update_layout(height=400)
            st.plotly_chart(fig_monthly, use_container_width=True)
            
            # Monthly breakdown by category
            monthly_category = filtered_expenses.groupby([
                filtered_expenses['date'].dt.to_period('M'), 'category'
            ])['amount'].sum().unstack(fill_value=0)
            
            fig_stacked = px.bar(
                monthly_category,
                title='Monthly Costs by Category',
                height=500
            )
            st.plotly_chart(fig_stacked, use_container_width=True)
        
        else:  # Variable vs Fixed
            # Categorize costs
            variable_categories = [cat for cat, info in self.cost_categories.items() if info['type'] == 'variable']
            fixed_categories = [cat for cat, info in self.cost_categories.items() if info['type'] == 'fixed']
            
            variable_costs = filtered_expenses[filtered_expenses['category'].isin(variable_categories)]['amount'].sum()
            fixed_costs = filtered_expenses[filtered_expenses['category'].isin(fixed_categories)]['amount'].sum()
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Variable vs Fixed pie chart
                fig_type = px.pie(
                    values=[variable_costs, fixed_costs],
                    names=['Variable Costs', 'Fixed Costs'],
                    title='Variable vs Fixed Costs',
                    color_discrete_sequence=['#FF6B6B', '#4ECDC4']
                )
                fig_type.update_layout(height=400)
                st.plotly_chart(fig_type, use_container_width=True)
            
            with col2:
                st.markdown("### 📊 Cost Structure Analysis")
                
                total_costs = variable_costs + fixed_costs
                
                st.metric("Variable Costs", f"${variable_costs:,.0f}")
                st.metric("Fixed Costs", f"${fixed_costs:,.0f}")
                st.metric("Variable %", f"{(variable_costs/total_costs)*100:.1f}%")
                st.metric("Fixed %", f"{(fixed_costs/total_costs)*100:.1f}%")
                
                st.markdown("### 💡 Insights")
                if variable_costs > fixed_costs * 2:
                    st.info("🔹 High variable cost ratio - costs scale with production")
                elif fixed_costs > variable_costs * 1.5:
                    st.warning("🔶 High fixed cost ratio - need to maximize production")
                else:
                    st.success("✅ Balanced cost structure")
    
    def create_profitability_analysis(self):
        """Create crop profitability analysis"""
        st.subheader("🌱 Crop Profitability Analysis")
        
        crop_data = st.session_state.crop_profitability
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            most_profitable = crop_data.groupby('crop')['profit_per_ha'].mean().idxmax()
            max_profit = crop_data.groupby('crop')['profit_per_ha'].mean().max()
            st.metric("Most Profitable Crop", most_profitable)
            st.caption(f"${max_profit:,.0f}/ha")
        
        with col2:
            highest_margin = crop_data.groupby('crop')['profit_margin'].mean().idxmax()
            max_margin = crop_data.groupby('crop')['profit_margin'].mean().max()
            st.metric("Highest Margin", highest_margin)
            st.caption(f"{max_margin:.1f}%")
        
        with col3:
            total_profit = crop_data['profit'].sum()
            st.metric("Total Crop Profit", f"${total_profit:,.0f}")
        
        with col4:
            avg_margin = crop_data['profit_margin'].mean()
            st.metric("Average Margin", f"{avg_margin:.1f}%")
        
        # Profitability charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Profit per hectare by crop
            profit_per_ha = crop_data.groupby('crop')['profit_per_ha'].mean().sort_values(ascending=False)
            
            fig_profit_ha = px.bar(
                x=profit_per_ha.values,
                y=profit_per_ha.index,
                orientation='h',
                title='Average Profit per Hectare by Crop',
                color=profit_per_ha.values,
                color_continuous_scale='Greens'
            )
            fig_profit_ha.update_layout(height=400)
            st.plotly_chart(fig_profit_ha, use_container_width=True)
        
        with col2:
            # Profit margin by crop
            margin_by_crop = crop_data.groupby('crop')['profit_margin'].mean().sort_values(ascending=False)
            
            fig_margin = px.bar(
                x=margin_by_crop.values,
                y=margin_by_crop.index,
                orientation='h',
                title='Average Profit Margin by Crop (%)',
                color=margin_by_crop.values,
                color_continuous_scale='Blues'
            )
            fig_margin.update_layout(height=400)
            st.plotly_chart(fig_margin, use_container_width=True)
        
        # Quarterly performance
        st.subheader("📊 Quarterly Performance")
        
        # Revenue by crop and quarter
        quarterly_revenue = crop_data.pivot(index='crop', columns='quarter', values='revenue')
        
        fig_quarterly = px.imshow(
            quarterly_revenue,
            title='Revenue by Crop and Quarter',
            color_continuous_scale='Greens',
            aspect='auto'
        )
        fig_quarterly.update_layout(height=400)
        st.plotly_chart(fig_quarterly, use_container_width=True)
        
        # Detailed crop performance table
        st.subheader("📋 Detailed Crop Performance")
        
        crop_summary = crop_data.groupby('crop').agg({
            'revenue': ['sum', 'mean'],
            'costs': ['sum', 'mean'],
            'profit': ['sum', 'mean'],
            'profit_margin': 'mean',
            'profit_per_ha': 'mean',
            'area_hectares': 'sum'
        }).round(2)
        
        # Flatten column names
        crop_summary.columns = ['_'.join(col).strip() for col in crop_summary.columns]
        
        st.dataframe(crop_summary, use_container_width=True)
    
    def create_roi_analysis(self):
        """Create ROI and investment analysis"""
        st.subheader("📊 ROI & Investment Analysis")
        
        roi_data = st.session_state.roi_analysis
        
        # Investment comparison
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # ROI comparison chart
            fig_roi = px.scatter(
                roi_data,
                x='payback_period',
                y='roi_5_year',
                size='initial_cost',
                color='investment',
                title='Investment ROI Analysis',
                labels={
                    'payback_period': 'Payback Period (years)',
                    'roi_5_year': '5-Year ROI (%)',
                    'initial_cost': 'Initial Cost ($)'
                }
            )
            
            # Add quadrant lines
            fig_roi.add_hline(y=50, line_dash="dash", line_color="gray", opacity=0.5)
            fig_roi.add_vline(x=3, line_dash="dash", line_color="gray", opacity=0.5)
            
            fig_roi.update_layout(height=500)
            st.plotly_chart(fig_roi, use_container_width=True)
        
        with col2:
            st.markdown("### 💡 Investment Recommendations")
            
            # Rank investments by ROI
            roi_ranked = roi_data.sort_values('roi_5_year', ascending=False)
            
            for idx, row in roi_ranked.iterrows():
                roi_class = "roi-positive" if row['roi_5_year'] > 50 else "roi-neutral"
                
                st.markdown(f"""
                **{row['investment']}**
                - Initial Cost: ${row['initial_cost']:,}
                - <span class="{roi_class}">5-Year ROI: {row['roi_5_year']:.1f}%</span>
                - Payback: {row['payback_period']:.1f} years
                """, unsafe_allow_html=True)
                st.markdown("---")
        
        # Break-even analysis
        st.subheader("⚖️ Break-even Analysis")
        
        col1, col2, col3 =
