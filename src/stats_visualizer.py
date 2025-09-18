# src/stats_visualizer.py
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class StatsVisualizer:
    def __init__(self, greenhouse_config: GreenhouseConfig):
        self.config = greenhouse_config
        self.style_config = self._setup_visualization_style()
    
    def _setup_visualization_style(self) -> Dict:
        """Setup visualization style based on operation scale"""
        if self.config.operation_scale == OperationScale.PERSONAL:
            return {
                'color_palette': 'Set2',
                'style': 'whitegrid',
                'figure_size': (10, 6),
                'title_size': 14,
                'label_size': 12
            }
        elif self.config.operation_scale == OperationScale.COMMERCIAL:
            return {
                'color_palette': 'viridis',
                'style': 'darkgrid',
                'figure_size': (12, 8),
                'title_size': 16,
                'label_size': 14
            }
        else:  # Educational
            return {
                'color_palette': 'bright',
                'style': 'whitegrid',
                'figure_size': (11, 7),
                'title_size': 15,
                'label_size': 13
            }
    
    def create_environmental_dashboard(self, environmental_data: List[EnvironmentalData]) -> plt.Figure:
        """Create comprehensive environmental monitoring dashboard"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Greenhouse Environmental Monitoring Dashboard', 
                    fontsize=self.style_config['title_size'], fontweight='bold')
        
        # Convert data to DataFrame for easier plotting
        df_data = []
        for data in environmental_data:
            df_data.append({
                'timestamp': data.timestamp,
                'temperature': data.temperature,
                'humidity': data.humidity,
                'soil_moisture': data.soil_moisture,
                'co2_level': data.co2_level,
                'light_intensity': data.light_intensity / 1000,  # Convert to k-lux
                'ph_level': data.ph_level
            })
        
        df = pd.DataFrame(df_data)
        
        # Temperature over time
        axes[0,0].plot(df['timestamp'], df['temperature'], color='red', linewidth=2)
        axes[0,0].set_title('Temperature (°C)', fontsize=self.style_config['label_size'])
        axes[0,0].set_ylabel('Temperature °C')
        axes[0,0].grid(True, alpha=0.3)
        
        # Humidity over time
        axes[0,1].plot(df['timestamp'], df['humidity'], color='blue', linewidth=2)
        axes[0,1].set_title('Humidity (%)', fontsize=self.style_config['label_size'])
        axes[0,1].set_ylabel('Humidity %')
        axes[0,1].grid(True, alpha=0.3)
        
        # Soil moisture over time
        axes[0,2].plot(df['timestamp'], df['soil_moisture'], color='brown', linewidth=2)
        axes[0,2].set_title('Soil Moisture (%)', fontsize=self.style_config['label_size'])
        axes[0,2].set_ylabel('Soil Moisture %')
        axes[0,2].grid(True, alpha=0.3)
        
        # CO2 levels over time
        axes[1,0].plot(df['timestamp'], df['co2_level'], color='green', linewidth=2)
        axes[1,0].set_title('CO₂ Levels (PPM)', fontsize=self.style_config['label_size'])
        axes[1,0].set_ylabel('CO₂ PPM')
        axes[1,0].grid(True, alpha=0.3)
        
        # Light intensity over time
        axes[1,1].plot(df['timestamp'], df['light_intensity'], color='orange', linewidth=2)
        axes[1,1].set_title('Light Intensity (k-lux)', fontsize=self.style_config['label_size'])
        axes[1,1].set_ylabel('Light k-lux')
        axes[1,# src/greenhouse_analyzer.py
import random
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from enum import Enum

class OperationScale(Enum):
    PERSONAL = "personal"  # Home gardening, small greenhouse
    COMMERCIAL = "commercial"  # Large-scale farming operations
    EDUCATIONAL = "educational"  # Schools, research, demonstrations

@dataclass
class EnvironmentalData:
    timestamp: datetime
    temperature: float  # Celsius
    humidity: float  # Percentage
    soil_moisture: float  # Percentage
    co2_level: float  # PPM
    ph_level: float  # 0-14 scale
    light_intensity: float  # Lux
    soil_nutrients: Dict[str, float]  # NPK levels

@dataclass
class GreenhouseConfig:
    size_sqft: int
    operation_scale: OperationScale
    automation_level: float  # 0-1, where 1 is fully automated
    climate_control: bool
    irrigation_system: bool
    location: str  # For weather simulation

class GreenhouseAnalyzer:
    def __init__(self, config: GreenhouseConfig):
        self.config = config
        self.environmental_history: List[EnvironmentalData] = []
        self.alerts: List[str] = []
        
    def simulate_environmental_conditions(self, days: int = 30) -> List[EnvironmentalData]:
        """Generate realistic environmental data based on operation scale"""
        data = []
        base_date = datetime.now()
        
        # Scale-specific parameters
        scale_params = self._get_scale_parameters()
        
        for day in range(days):
            for hour in range(0, 24, 4):  # Every 4 hours
                timestamp = base_date + timedelta(days=day, hours=hour)
                
                # Simulate daily and seasonal variations
                temp_variation = 5 * np.sin(2 * np.pi * hour / 24)  # Daily cycle
                seasonal_temp = 20 + 10 * np.sin(2 * np.pi * day / 365)  # Seasonal
                
                env_data = EnvironmentalData(
                    timestamp=timestamp,
                    temperature=seasonal_temp + temp_variation + random.gauss(0, scale_params['temp_noise']),
                    humidity=60 + 20 * np.sin(2 * np.pi * hour / 24) + random.gauss(0, scale_params['humidity_noise']),
                    soil_moisture=70 + random.gauss(0, scale_params['moisture_noise']),
                    co2_level=400 + 100 * random.random() + scale_params['co2_boost'],
                    ph_level=6.5 + random.gauss(0, 0.3),
                    light_intensity=max(0, 50000 * max(0, np.sin(np.pi * hour / 12)) + random.gauss(0, 5000)),
                    soil_nutrients={
                        'nitrogen': 50 + random.gauss(0, 10),
                        'phosphorus': 30 + random.gauss(0, 5),
                        'potassium': 40 + random.gauss(0, 8)
                    }
                )
                data.append(env_data)
        
        self.environmental_history.extend(data)
        return data
    
    def _get_scale_parameters(self) -> Dict:
        """Get simulation parameters based on operation scale"""
        if self.config.operation_scale == OperationScale.PERSONAL:
            return {
                'temp_noise': 2.0,
                'humidity_noise': 5.0,
                'moisture_noise': 8.0,
                'co2_boost': 0
            }
        elif self.config.operation_scale == OperationScale.COMMERCIAL:
            return {
                'temp_noise': 1.0,
                'humidity_noise': 3.0,
                'moisture_noise': 5.0,
                'co2_boost': 200  # CO2 supplementation
            }
        else:  # Educational
            return {
                'temp_noise': 1.5,
                'humidity_noise': 4.0,
                'moisture_noise': 6.0,
                'co2_boost': 100
            }
    
    def analyze_conditions(self) -> Dict[str, any]:
        """Analyze current environmental conditions and provide recommendations"""
        if not self.environmental_history:
            return {"error": "No environmental data available"}
        
        recent_data = self.environmental_history[-24:]  # Last 24 readings
        
        avg_temp = np.mean([d.temperature for d in recent_data])
        avg_humidity = np.mean([d.humidity for d in recent_data])
        avg_moisture = np.mean([d.soil_moisture for d in recent_data])
        avg_co2 = np.mean([d.co2_level for d in recent_data])
        
        analysis = {
            'current_conditions': {
                'temperature': round(avg_temp, 1),
                'humidity': round(avg_humidity, 1),
                'soil_moisture': round(avg_moisture, 1),
                'co2_level': round(avg_co2, 0)
            },
            'recommendations': [],
            'alerts': []
        }
        
        # Generate scale-appropriate recommendations
        if avg_temp < 15:
            analysis['recommendations'].append("Temperature low - consider heating system")
            analysis['alerts'].append("⚠️ Temperature below optimal range")
        elif avg_temp > 30:
            analysis['recommendations'].append("Temperature high - increase ventilation")
            analysis['alerts'].append("🌡️ High temperature alert")
        
        if avg_humidity < 40:
            analysis['recommendations'].append("Humidity low - add humidification")
        elif avg_humidity > 80:
            analysis['recommendations'].append("Humidity high - improve air circulation")
            analysis['alerts'].append("💧 High humidity risk of mold")
        
        if avg_moisture < 40:
            analysis['recommendations'].append("Soil moisture low - increase irrigation")
            analysis['alerts'].append("🏜️ Soil moisture critically low")
        
        if avg_co2 < 350:
            analysis['recommendations'].append("CO2 levels low - consider supplementation")
        
        return analysis

# src/vegetation_planner.py
from dataclasses import dataclass
from typing import List, Dict
import random

@dataclass
class PlantProfile:
    name: str
    space_required: float  # Square feet per plant
    grow_time_days: int
    optimal_temp_range: Tuple[float, float]  # Min, Max Celsius
    water_needs: str  # "low", "medium", "high"
    market_value_per_unit: float  # Dollar value
    difficulty_level: str  # "beginner", "intermediate", "advanced"
    seasonal_preference: str  # "spring", "summer", "fall", "winter", "all"

class VegetationPlanner:
    def __init__(self, greenhouse_config: GreenhouseConfig):
        self.config = greenhouse_config
        self.plant_database = self._initialize_plant_database()
    
    def _initialize_plant_database(self) -> List[PlantProfile]:
        """Initialize plant database with scale-appropriate options"""
        base_plants = [
            PlantProfile("Tomatoes", 4.0, 75, (18, 26), "high", 3.50, "intermediate", "summer"),
            PlantProfile("Lettuce", 1.0, 45, (15, 20), "medium", 2.00, "beginner", "all"),
            PlantProfile("Basil", 1.5, 60, (20, 25), "medium", 8.00, "beginner", "summer"),
            PlantProfile("Peppers", 3.0, 80, (20, 28), "medium", 4.00, "intermediate", "summer"),
            PlantProfile("Spinach", 0.75, 30, (10, 18), "medium", 2.50, "beginner", "spring"),
            PlantProfile("Cucumber", 6.0, 65, (20, 24), "high", 1.50, "intermediate", "summer"),
            PlantProfile("Herbs Mix", 0.5, 40, (18, 22), "low", 12.00, "beginner", "all"),
            PlantProfile("Strawberries", 2.0, 90, (15, 25), "medium", 6.00, "intermediate", "spring"),
        ]
        
        # Add scale-specific plants
        if self.config.operation_scale == OperationScale.COMMERCIAL:
            base_plants.extend([
                PlantProfile("Industrial Lettuce", 0.8, 35, (15, 20), "medium", 1.80, "beginner", "all"),
                PlantProfile("Cherry Tomatoes", 3.0, 70, (18, 26), "high", 5.00, "intermediate", "summer"),
                PlantProfile("Microgreens", 0.1, 14, (18, 22), "low", 25.00, "advanced", "all"),
            ])
        elif self.config.operation_scale == OperationScale.EDUCATIONAL:
            base_plants.extend([
                PlantProfile("Sunflowers", 4.0, 85, (18, 24), "medium", 2.00, "beginner", "summer"),
                PlantProfile("Bean Plants", 2.0, 50, (16, 22), "medium", 3.00, "beginner", "spring"),
                PlantProfile("Radishes", 0.25, 25, (12, 18), "low", 1.50, "beginner", "all"),
            ])
        
        return base_plants
    
    def generate_planting_plan(self, season: str = "current") -> Dict[str, any]:
        """Generate an optimal planting plan based on space and scale"""
        available_space = self.config.size_sqft * 0.8  # 80% usable space
        
        # Filter plants by season and difficulty based on scale
        suitable_plants = self._filter_plants_by_criteria(season)
        
        # Generate plan using different strategies by scale
        if self.config.operation_scale == OperationScale.PERSONAL:
            plan = self._generate_diverse_personal_plan(suitable_plants, available_space)
        elif self.config.operation_scale == OperationScale.COMMERCIAL:
            plan = self._generate_profit_focused_plan(suitable_plants, available_space)
        else:  # Educational
            plan = self._generate_educational_plan(suitable_plants, available_space)
        
        return plan
    
    def _filter_plants_by_criteria(self, season: str) -> List[PlantProfile]:
        """Filter plants based on season and operation scale"""
        filtered = []
        
        for plant in self.plant_database:
            # Season filter
            if season != "current" and plant.seasonal_preference != "all" and plant.seasonal_preference != season:
                continue
            
            # Difficulty filter based on scale
            if self.config.operation_scale == OperationScale.PERSONAL:
                if plant.difficulty_level == "advanced" and random.random() < 0.7:
                    continue  # 70% chance to skip advanced plants for personal use
            
            filtered.append(plant)
        
        return filtered
    
    def _generate_diverse_personal_plan(self, plants: List[PlantProfile], space: float) -> Dict:
        """Generate a diverse plan for personal gardening"""
        plan = {
            'strategy': 'Diverse Home Garden',
            'total_space': space,
            'plantings': [],
            'expected_harvest_timeline': [],
            'care_tips': []
        }
        
        used_space = 0
        selected_plants = []
        
        # Prioritize variety and beginner-friendly plants
        plants_by_difficulty = sorted(plants, key=lambda p: (p.difficulty_level == "advanced", -p.market_value_per_unit))
        
        for plant in plants_by_difficulty:
            if used_space >= space * 0.9:
                break
            
            max_plants = int((space - used_space) / plant.space_required)
            if max_plants > 0:
                # For personal use, don't plant too many of the same thing
                num_plants = min(max_plants, 6 if plant.space_required > 2 else 12)
                
                planting = {
                    'plant': plant.name,
                    'quantity': num_plants,
                    'space_used': num_plants * plant.space_required,
                    'harvest_date': f"{plant.grow_time_days} days",
                    'estimated_yield_value': num_plants * plant.market_value_per_unit
                }
                
                plan['plantings'].append(planting)
                used_space += planting['space_used']
        
        plan['space_utilization'] = f"{(used_space/space)*100:.1f}%"
        plan['care_tips'] = [
            "Start with easier plants like lettuce and herbs",
            "Monitor temperature and humidity daily",
            "Rotate crops for soil health",
            "Keep a garden journal to track progress"
        ]
        
        return plan
    
    def _generate_profit_focused_plan(self, plants: List[PlantProfile], space: float) -> Dict:
        """Generate a profit-optimized plan for commercial operations"""
        # Calculate profit per square foot per day for each plant
        plants_with_profit = []
        for plant in plants:
            daily_profit_per_sqft = (plant.market_value_per_unit / plant.space_required) / plant.grow_time_days
            plants_with_profit.append((plant, daily_profit_per_sqft))
        
        # Sort by profitability
        plants_with_profit.sort(key=lambda x: x[1], reverse=True)
        
        plan = {
            'strategy': 'Maximum Profitability',
            'total_space': space,
            'plantings': [],
            'projected_revenue': 0,
            'harvest_cycles_per_year': {}
        }
        
        used_space = 0
        
        # Focus on top 3-4 most profitable plants
        for plant, profit_rate in plants_with_profit[:4]:
            if used_space >= space * 0.95:
                break
            
            remaining_space = space - used_space
            num_plants = int(remaining_space / plant.space_required)
            
            if num_plants > 0:
                space_for_this_plant = min(remaining_space, space * 0.4)  # Max 40% per plant type
                num_plants = int(space_for_this_plant / plant.space_required)
                
                cycles_per_year = 365 / plant.grow_time_days
                annual_revenue = num_plants * plant.market_value_per_unit * cycles_per_year
                
                planting = {
                    'plant': plant.name,
                    'quantity': num_plants,
                    'space_used': num_plants * plant.space_required,
                    'annual_cycles': round(cycles_per_year, 1),
                    'annual_revenue': round(annual_revenue, 2),
                    'profit_per_sqft_per_day': round(profit_rate, 3)
                }
                
                plan['plantings'].append(planting)
                plan['projected_revenue'] += annual_revenue
                used_space += planting['space_used']
        
        plan['projected_revenue'] = round(plan['projected_revenue'], 2)
        plan['space_utilization'] = f"{(used_space/space)*100:.1f}%"
        
        return plan
    
    def _generate_educational_plan(self, plants: List[PlantProfile], space: float) -> Dict:
        """Generate an educational plan focusing on learning opportunities"""
        plan = {
            'strategy': 'Educational Demonstration',
            'total_space': space,
            'plantings': [],
            'learning_objectives': [],
            'experiment_suggestions': []
        }
        
        # Include plants with different characteristics for comparison
        selected_plants = []
        used_space = 0
        
        # Ensure variety in growth times, care needs, and difficulty
        fast_growing = [p for p in plants if p.grow_time_days < 40]
        medium_growing = [p for p in plants if 40 <= p.grow_time_days < 70]
        slow_growing = [p for p in plants if p.grow_time_days >= 70]
        
        categories = [
            ("Fast Growing", fast_growing, 0.3),
            ("Medium Growing", medium_growing, 0.4),
            ("Slow Growing", slow_growing, 0.3)
        ]
        
        for category_name, category_plants, space_allocation in categories:
            if not category_plants:
                continue
            
            allocated_space = space * space_allocation
            category_used = 0
            
            for plant in category_plants[:3]:  # Max 3 plants per category
                if category_used >= allocated_space * 0.9:
                    break
                
                num_plants = min(8, int((allocated_space - category_used) / plant.space_required))
                if num_plants > 0:
                    planting = {
                        'plant': plant.name,
                        'category': category_name,
                        'quantity': num_plants,
                        'space_used': num_plants * plant.space_required,
                        'growth_time': f"{plant.grow_time_days} days",
                        'difficulty': plant.difficulty_level,
                        'learning_focus': self._get_learning_focus(plant)
                    }
                    
                    plan['plantings'].append(planting)
                    category_used += planting['space_used']
                    used_space += planting['space_used']
        
        plan['space_utilization'] = f"{(used_space/space)*100:.1f}%"
        plan['learning_objectives'] = [
            "Compare growth rates between different plant types",
            "Observe water and nutrient requirements",
            "Study plant responses to environmental changes",
            "Practice harvest timing and techniques"
        ]
        
        plan['experiment_suggestions'] = [
            "Test different fertilizer levels on same plant species",
            "Compare growth under different light conditions",
            "Monitor pH effects on plant health",
            "Track harvest yields over multiple cycles"
        ]
        
        return plan
    
    def _get_learning_focus(self, plant: PlantProfile) -> str:
        """Generate learning focus points for educational plants"""
        focuses = {
            "fast_growth": "Rapid development observation",
            "water_needs": f"Watering technique ({plant.water_needs} needs)",
            "temperature": "Temperature sensitivity",
            "harvest_timing": "Optimal harvest identification"
        }
        
        if plant.grow_time_days < 30:
            return focuses["fast_growth"]
        elif plant.water_needs == "high":
            return focuses["water_needs"]
        elif plant.optimal_temp_range[1] - plant.optimal_temp_range[0] < 8:
            return focuses["temperature"]
        else:
            return focuses["harvest_timing"]

# src/economics_calculator.py
from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np

@dataclass
class CostStructure:
    initial_setup: float
    monthly_utilities: float
    seeds_per_cycle: float
    fertilizer_per_cycle: float
    labor_per_hour: float
    maintenance_monthly: float
    insurance_monthly: float

@dataclass
class EconomicProjection:
    total_investment: float
    monthly_costs: float
    projected_revenue: float
    break_even_months: float
    roi_annual: float
    profit_margin: float

class EconomicsCalculator:
    def __init__(self, greenhouse_config: GreenhouseConfig):
        self.config = greenhouse_config
        self.cost_structure = self._initialize_cost_structure()
    
    def _initialize_cost_structure(self) -> CostStructure:
        """Initialize costs based on operation scale and size"""
        size_factor = self.config.size_sqft / 1000  # Normalize to 1000 sqft
        
        if self.config.operation_scale == OperationScale.PERSONAL:
            return CostStructure(
                initial_setup=3000 + (size_factor * 1500),  # Basic greenhouse setup
                monthly_utilities=120 + (size_factor * 50),
                seeds_per_cycle=50 + (size_factor * 25),
                fertilizer_per_cycle=30 + (size_factor * 20),
                labor_per_hour=0,  # Personal labor not counted
                maintenance_monthly=100 + (size_factor * 30),
                insurance_monthly=50 + (size_factor * 20)
            )
        elif self.config.operation_scale == OperationScale.COMMERCIAL:
            automation_multiplier = 1 + (self.config.automation_level * 2)
            return CostStructure(
                initial_setup=15000 + (size_factor * 8000 * automation_multiplier),
                monthly_utilities=500 + (size_factor * 200),
                seeds_per_cycle=200 + (size_factor * 100),
                fertilizer_per_cycle=150 + (size_factor * 80),
                labor_per_hour=15.00,  # Minimum wage assumption
                maintenance_monthly=400 + (size_factor * 150),
                insurance_monthly=300 + (size_factor * 100)
            )
        else:  # Educational
            return CostStructure(
                initial_setup=8000 + (size_factor * 3000),
                monthly_utilities=200 + (size_factor * 80),
                seeds_per_cycle=75 + (size_factor * 40),
                fertilizer_per_cycle=50 + (size_factor * 30),
                labor_per_hour=12.00,  # Part-time student wages
                maintenance_monthly=150 + (size_factor * 60),
                insurance_monthly=100 + (size_factor * 40)
            )
    
    def calculate_economics(self, planting_plan: Dict, timeframe_months: int = 12) -> EconomicProjection:
        """Calculate complete economic analysis"""
        
        # Calculate initial investment
        total_investment = self.cost_structure.initial_setup
        
        # Calculate ongoing costs
        monthly_fixed_costs = (
            self.cost_structure.monthly_utilities +
            self.cost_structure.maintenance_monthly +
            self.cost_structure.insurance_monthly
        )
        
        # Calculate cycle-based costs and revenue
        total_cycles_per_year = 0
        annual_revenue = 0
        annual_variable_costs = 0
        
        for planting in planting_plan.get('plantings', []):
            if 'annual_cycles' in planting:
                cycles = planting['annual_cycles']
                revenue = planting.get('annual_revenue', planting.get('estimated_yield_value', 0) * cycles)
            else:
                # Estimate cycles for personal/educational plans
                plant_name = planting['plant']
                growth_days = self._estimate_growth_days(plant_name)
                cycles = 365 / growth_days if growth_days > 0 else 4
                revenue = planting.get('estimated_yield_value', 0) * cycles
            
            total_cycles_per_year += cycles
            annual_revenue += revenue
            
            # Variable costs per cycle
            cycle_cost = (
                self.cost_structure.seeds_per_cycle * 0.1 +  # 10% of seed cost per plant type
                self.cost_structure.fertilizer_per_cycle * 0.1
            )
            annual_variable_costs += cycle_cost * cycles
        
        # Labor costs
        if self.cost_structure.labor_per_hour > 0:
            if self.config.operation_scale == OperationScale.COMMERCIAL:
                monthly_labor_hours = 160 + (self.config.size_sqft / 100)  # Scale with size
            else:  # Educational
                monthly_labor_hours = 40 + (self.config.size_sqft / 200)
            
            monthly_labor_cost = monthly_labor_hours * self.cost_structure.labor_per_hour
            monthly_fixed_costs += monthly_labor_cost
        
        # Project over timeframe
        total_fixed_costs = monthly_fixed_costs * timeframe_months
        total_variable_costs = annual_variable_costs * (timeframe_months / 12)
        total_revenue = annual_revenue * (timeframe_months / 12)
        
        total_costs = total_investment + total_fixed_costs + total_variable_costs
        net_profit = total_revenue - total_costs
        
        # Calculate metrics
        break_even_months = float('inf')
        if annual_revenue > (monthly_fixed_costs * 12 + annual_variable_costs):
            monthly_net = (annual_revenue - annual_variable_costs) / 12 - monthly_fixed_costs
            if monthly_net > 0:
                break_even_months = total_investment / monthly_net
        
        roi_annual = (net_profit / total_investment) * (12 / timeframe_months) * 100 if total_investment > 0 else 0
        profit_margin = (net_profit / total_revenue) * 100 if total_revenue > 0 else -100
        
        return EconomicProjection(
            total_investment=round(total_investment, 2),
            monthly_costs=round(monthly_fixed_costs, 2),
            projected_revenue=round(total_revenue, 2),
            break_even_months=round(break_even_months, 1) if break_even_months != float('inf') else -1,
            roi_annual=round(roi_annual, 2),
            profit_margin=round(profit_margin, 2)
        )
    
    def _estimate_growth_days(self, plant_name: str) -> int:
        """Estimate growth days for common plants"""
        growth_estimates = {
            'tomatoes': 75, 'lettuce': 45, 'basil': 60, 'peppers': 80,
            'spinach': 30, 'cucumber': 65, 'herbs': 40, 'strawberries': 90,
            'sunflowers': 85, 'beans': 50, 'radishes': 25, 'microgreens': 14
        }
        
        for key, days in growth_estimates.items():
            if key in plant_name.lower():
                return days
        return 60  # Default estimate
    
    def generate_cost_breakdown(self) -> Dict[str, float]:
        """Generate detailed cost breakdown"""
        return {
            'Initial Setup': self.cost_structure.initial_setup,
            'Monthly Utilities': self.cost_structure.monthly_utilities,
            'Monthly Maintenance': self.cost_structure.maintenance_monthly,
            'Monthly Insurance': self.cost_structure.insurance_monthly,
            'Seeds per Cycle': self.cost_structure.seeds_per_cycle,
            'Fertilizer per Cycle': self.cost_structure.fertilizer_per_cycle,
            'Labor per Hour': self.cost_structure.labor_per_hour
        }

# src/sapling_manager.py
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from enum import Enum
import random

class SeedStage(Enum):
    SEED = "seed"
    GERMINATED = "germinated"
    SEEDLING = "seedling"
    SAPLING = "sapling"
    TRANSPLANT_READY = "transplant_ready"
    MATURE = "mature"

@dataclass
class SeedBatch:
    batch_id: str
    plant_type: str
    quantity: int
    start_date: datetime
    current_stage: SeedStage = SeedStage.SEED
    success_rate: float = 0.0
    days_in_stage: int = 0
    environmental_stress: float = 0.0
    notes: List[str] = field(default_factory=list)

@dataclass
class ProductionSchedule:
    target_date: datetime
    plant_type: str
    quantity_needed: int
    batch_ids: List[str] = field(default_factory=list)
    status: str = "planned"  # planned, in_progress, completed, delayed

class SaplingManager:
    def __init__(self, greenhouse_config: GreenhouseConfig):
        self.config = greenhouse_config
        self.active_batches: Dict[str, SeedBatch] = {}
        self.production_schedule: List[ProductionSchedule] = []
        self.stage_duration = self._get_stage_durations()
        self.success_rates = self._get_success_rates()
    
    def _get_stage_durations(self) -> Dict[SeedStage, int]:
        """Get typical duration for each growth stage in days"""
        base_durations = {
            SeedStage.SEED: 3,
            SeedStage.GERMINATED: 7,
            SeedStage.SEEDLING: 14,
            SeedStage.SAPLING: 21,
            SeedStage.TRANSPLANT_READY: 7
        }
        
        # Adjust based on operation scale (better equipment = faster growth)
        if self.config.operation_scale == OperationScale.COMMERCIAL:
            return {stage: max(1, int(days * 0.8)) for stage, days in base_durations.items()}
        elif self.config.operation_scale == OperationScale.EDUCATIONAL:
            return {stage: int(days * 1.1) for stage, days in base_durations.items()}
        return base_durations
    
    def _get_success_rates(self) -> Dict[SeedStage, float]:
        """Get expected success rates for each stage transition"""
        base_rates = {
            SeedStage.SEED: 0.85,  # Germination rate
            SeedStage.GERMINATED: 0.90,
            SeedStage.SEEDLING: 0.95,
            SeedStage.SAPLING: 0.98,
            SeedStage.TRANSPLANT_READY: 0.99
        }
        
        # Scale adjustments
        if self.config.operation_scale == OperationScale.COMMERCIAL:
            return {stage: min(0.98, rate + 0.05) for stage, rate in base_rates.items()}
        elif self.config.operation_scale == OperationScale.PERSONAL:
            return {stage: max(0.60, rate - 0.10) for stage, rate in base_rates.items()}
        return base_rates
    
    def start_seed_batch(self, plant_type: str, quantity: int, target_date: Optional[datetime] = None) -> str:
        """Start a new seed batch"""
        batch_id = f"{plant_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        batch = SeedBatch(
            batch_id=batch_id,
            plant_type=plant_type,
            quantity=quantity,
            start_date=datetime.now(),
            current_stage=SeedStage.SEED
        )
        
        self.active_batches[batch_id] = batch
        
        # Add to production schedule if target date provided
        if target_date:
            schedule = ProductionSchedule(
                target_date=target_date,
                plant_type=plant_type,
                quantity_needed=quantity,
                batch_ids=[batch_id],
                status="in_progress"
            )
            self.production_schedule.append(schedule)
        
        return batch_id
    
    def simulate_batch_progress(self, days: int = 1):
        """Simulate growth progress for all active batches"""
        for batch_id, batch in self.active_batches.items():
            if batch.current_stage == SeedStage.MATURE:
                continue
            
            # Simulate environmental stress (affects success rates)
            daily_stress = random.gauss(0.1, 0.05)  # Base stress
            if self.config.automation_level < 0.5:
                daily_stress += random.gauss(0.05, 0.02)  # More stress with less automation
            
            batch.environmental_stress = max(0, min(1, batch.environmental_stress + daily_stress - 0.08))
            batch.days_in_stage += days
            
            # Check for stage advancement
            expected_duration = self.stage_duration[batch.current_stage]
            if batch.days_in_stage >= expected_duration:
                self._advance_batch_stage(batch)
    
    def _advance_batch_stage(self, batch: SeedBatch):
        """Advance a batch to the next growth stage"""
        current_stage = batch.current_stage
        success_rate = self.success_rates[current_stage]
        
        # Apply environmental stress penalty
        stress_penalty = batch.environmental_stress * 0.2
        adjusted_success_rate = max(0.3, success_rate - stress_penalty)
        
        # Calculate survivors
        survivors = int(batch.quantity * (random.gauss(adjusted_success_rate, 0.05)))
        survivors = max(0, min(batch.quantity, survivors))
        
        loss_count = batch.quantity - survivors
        if loss_count > 0:
            batch.notes.append(f"Lost {loss_count} plants transitioning from {current_stage.value}")
        
        batch.quantity = survivors
        batch.success_rate = survivors / (survivors + loss_count) if (survivors + loss_count) > 0 else 0
        batch.days_in_stage = 0
        
        # Advance to next stage
        stage_order = [SeedStage.SEED, SeedStage.GERMINATED, SeedStage.SEEDLING, 
                      SeedStage.SAPLING, SeedStage.TRANSPLANT_READY, SeedStage.MATURE]
        
        current_index = stage_order.index(current_stage)
        if current_index < len(stage_order) - 1:
            batch.current_stage = stage_order[current_index + 1]
            batch.notes.append(f"Advanced to {batch.current_stage.value} stage with {survivors} plants")
    
    def get_production_status(self) -> Dict[str, any]:
        """Get current production status across all batches"""
        status = {
            'total_active_batches': len(self.active_batches),
            'stage_breakdown': {stage.value: 0 for stage in SeedStage},
            'plant_type_summary': {},
            'ready_for_transplant': 0,
            'production_alerts': []
        }
        
        for batch in self.active_batches.values():
            # Stage breakdown
            status['stage_breakdown'][batch.current_stage.value] += batch.quantity
            
            # Plant type summary
            if batch.plant_type not in status['plant_type_summary']:
                status['plant_type_summary'][batch.plant_type] = {
                    'total_plants': 0,
                    'batches': 0,
                    'avg_success_rate': 0
                }
            
            summary = status['plant_type_summary'][batch.plant_type]
            summary['total_plants'] += batch.quantity
            summary['batches'] += 1
            summary['avg_success_rate'] += batch.success_rate
            
            # Ready for transplant
            if batch.current_stage == SeedStage.TRANSPLANT_READY:
                status['ready_for_transplant'] += batch.quantity
            
            # Production alerts
            if batch.success_rate < 0.7:
                status['production_alerts'].append(
                    f"⚠️ Low success rate ({batch.success_rate:.1%}) in batch {batch.batch_id}"
                )
            
            if batch.environmental_stress > 0.6:
                status['production_alerts'].append(
                    f"🌡️ High environmental stress in batch {batch.batch_id}"
                )
        
        # Calculate average success rates
        for plant_type, summary in status['plant_type_summary'].items():
            if summary['batches'] > 0:
                summary['avg_success_rate'] = summary['avg_success_rate'] / summary['batches']
        
        return status
    
    def plan_production_schedule(self, planting_plan: Dict, weeks_ahead: int = 8) -> List[ProductionSchedule]:
        """Create production schedule based on planting plan"""
        schedule = []
        base_date = datetime.now()
        
        for planting in planting_plan.get('plantings', []):
            plant_type = planting['plant']
            quantity = planting['quantity']
            
            # Calculate when to start seeds (work backwards from transplant date)
            total_seed_time = sum(self.stage_duration.values())
            
            # Stagger production for continuous harvest
            cycles = 4 if self.config.operation_scale == OperationScale.COMMERCIAL else 2
            
            for cycle in range(cycles):
                cycle_date = base_date + timedelta(weeks=2 * cycle)
                target_transplant = cycle_date + timedelta(days=total_seed_time)
                
                schedule_item = ProductionSchedule(
                    target_date=target_transplant,
                    plant_type=plant_type,
                    quantity_needed=quantity // cycles,
                    status="planned"
                )
                schedule.append(schedule_item)
        
        self.production_schedule.extend(schedule)
        return schedule

# src/marketing_optimizer.py
from dataclasses import dataclass
from typing import Dict, List, Tuple
import random
import math

@dataclass
class MarketSegment:
    name: str
    size: int  # Number of potential customers
    price_sensitivity: float  # 0-1, where 1 is very sensitive
    quality_preference: float  # 0-1, where 1 prefers highest quality
    seasonal_demand: Dict[str, float]  # Demand multiplier by season
    preferred_products: List[str]

@dataclass
class MarketingStrategy:
    target_segments: List[str]
    pricing_strategy: str
    distribution_channels: List[str]
    promotional_activities: List[str]
    expected_reach: int
    estimated_conversion_rate: float
    projected_revenue_boost: float

class MarketingOptimizer:
    def __init__(self, greenhouse_config: GreenhouseConfig):
        self.config = greenhouse_config
        self.market_segments = self._initialize_market_segments()
        self.channels = self._get_available_channels()
    
    def _initialize_market_segments(self) -> List[MarketSegment]:
        """Initialize market segments based on operation scale"""
        segments = []
        
        if self.config.operation_scale == OperationScale.PERSONAL:
            segments = [
                MarketSegment(
                    name="Local Neighbors",
                    size=50,
                    price_sensitivity=0.6,
                    quality_preference=0.7,
                    seasonal_demand={"spring": 1.2, "summer": 1.5, "fall": 1.0, "winter": 0.6},
                    preferred_products=["tomatoes", "herbs", "lettuce"]
                ),
                MarketSegment(
                    name="Farmers Market Customers",
                    size=200,
                    price_sensitivity=0.4,
                    quality_preference=0.9,
                    seasonal_demand={"spring": 1.1, "summer": 1.3, "fall": 1.2, "winter": 0.4},
                    preferred_products=["herbs", "microgreens", "specialty vegetables"]
                )
            ]
        
        elif self.config.operation_scale == OperationScale.COMMERCIAL:
            segments = [
                MarketSegment(
                    name="Grocery Chains",
                    size=20,
                    price_sensitivity=0.8,
                    quality_preference=0.6,
                    seasonal_demand={"spring": 1.0, "summer": 1.1, "fall": 1.0, "winter": 0.9},
                    preferred_products=["lettuce", "tomatoes", "peppers", "cucumber"]
                ),
                MarketSegment(
                    name="Restaurants",
                    size=150,
                    price_sensitivity=0.5,
                    quality_preference=0.9,
                    seasonal_demand={"spring": 1.2, "summer": 1.4, "fall": 1.1, "winter": 0.8},
                    preferred_products=["herbs", "microgreens", "specialty produce"]
                ),
                MarketSegment(
                    name="Food Service",
                    size=80,
                    price_sensitivity=0.7,
                    quality_preference=0.5,
                    seasonal_demand={"spring": 1.0, "summer": 1.0, "fall": 1.0, "winter": 1.0},
                    preferred_products=["lettuce", "tomatoes", "basic vegetables"]
                )
            ]
        
        else:  # Educational
            segments = [
                MarketSegment(
                    name="School Community",
                    size=500,
                    price_sensitivity=0.7,
                    quality_preference=0.6,
                    seasonal_demand={"spring": 1.3, "summer": 0.3, "fall": 1.4, "winter": 1.0},
                    preferred_products=["educational plants", "herbs", "vegetables"]
                ),
                MarketSegment(
                    name="Local Families",
                    size=300,
                    price_sensitivity=0.6,
                    quality_preference=0.8,
                    seasonal_demand={"spring": 1.2, "summer": 1.0, "fall": 1.1, "winter": 0.7},
                    preferred_products=["fresh vegetables", "herbs"]
                )
            ]
        
        return segments
    
    def _get_available_channels(self) -> List[str]:
        """Get available distribution channels based on operation scale"""
        if self.config.operation_scale == OperationScale.PERSONAL:
            return ["Direct Sales", "Farmers Market", "Online Local", "Word of Mouth"]
        elif self.config.operation_scale == OperationScale.COMMERCIAL:
            return ["Wholesale", "Direct to Restaurant", "Farmers Markets", "Online B2B", "Retail Partnerships"]
        else:  # Educational
            return ["School Store", "Community Events", "Educational Programs", "Local Partnerships"]
    
    def optimize_pricing(self, planting_plan: Dict, current_prices: Dict[str, float]) -> Dict[str, Dict]:
        """Optimize pricing strategy based on market segments and demand"""
        pricing_recommendations = {}
        
        for planting in planting_plan.get('plantings', []):
            plant_type = planting['plant'].lower()
            current_price = current_prices.get(plant_type, 3.0)  # Default price
            
            # Analyze market segments for this plant
            interested_segments = [
                seg for seg in self.market_segments 
                if any(prod in plant_type for prod in seg.preferred_products)
            ]
            
            if not interested_segments:
                continue
            
            # Calculate optimal price based on segment analysis
            total_demand = sum(seg.size for seg in interested_segments)
            weighted_sensitivity = sum(
                seg.size * seg.price_sensitivity for seg in interested_segments
            ) / total_demand
            
            # Price optimization logic
            if weighted_sensitivity < 0.4:  # Low sensitivity - can charge premium
                optimal_price = current_price * 1.2
                strategy = "Premium Pricing"
            elif weighted_sensitivity > 0.7:  # High sensitivity - competitive pricing
                optimal_price = current_price * 0.9
                strategy = "Competitive Pricing"
            else:  # Medium sensitivity - value pricing
                optimal_price = current_price * 1.05
                strategy = "Value Pricing"
            
            # Calculate expected impact
            price_change = (optimal_price - current_price) / current_price
            demand_change = -price_change * weighted_sensitivity  # Price elasticity
            revenue_change = (1 + price_change) * (1 + demand_change) - 1
            
            pricing_recommendations[plant_type] = {
                'current_price': round(current_price, 2),
                'recommended_price': round(optimal_price, 2),
                'strategy': strategy,
                'expected_demand_change': f"{demand_change:+.1%}",
                'expected_revenue_change': f"{revenue_change:+.1%}",
                'target_segments': [seg.name for seg in interested_segments]
            }
        
        return pricing_recommendations
    
    def develop_marketing_strategy(self, planting_plan: Dict, budget: float) -> MarketingStrategy:
        """Develop comprehensive marketing strategy"""
        
        # Identify primary products and segments
        products = [planting['plant'] for planting in planting_plan.get('plantings', [])]
        primary_segments = self._select_target_segments(products)
        
        # Select distribution channels
        selected_channels = self._select_optimal_channels(primary_segments, budget)
        
        # Develop promotional activities
        promotional_activities = self._plan_promotional_activities(budget, primary_segments)
        
        # Calculate expected reach and conversion
        expected_reach = self._calculate_reach(primary_segments, selected_channels, budget)
        conversion_rate = self._estimate_conversion_rate(primary_segments, promotional_activities)
        
        # Estimate revenue boost
        revenue_boost = self._estimate_revenue_impact(expected_reach, conversion_rate, products)
        
        return MarketingStrategy(
            target_segments=[seg.name for seg in primary_segments],
            pricing_strategy=self._determine_pricing_strategy(primary_segments),
            distribution_channels=selected_channels,
            promotional_activities=promotional_activities,
            expected_reach=expected_reach,
            estimated_conversion_rate=conversion_rate,
            projected_revenue_boost=revenue_boost
        )
    
    def _select_target_segments(self, products: List[str]) -> List[MarketSegment]:
        """Select optimal target segments based on products"""
        segment_scores = {}
        
        for segment in self.market_segments:
            score = 0
            for product in products:
                if any(pref in product.lower() for pref in segment.preferred_products):
                    score += segment.size * (1 - segment.price_sensitivity) * segment.quality_preference
            segment_scores[segment] = score
        
        # Select top segments (max 3 for focus)
        sorted_segments = sorted(segment_scores.items(), key=lambda x: x[1], reverse=True)
        return [seg for seg, score in sorted_segments[:3] if score > 0]
    
    def _select_optimal_channels(self, segments: List[MarketSegment], budget: float) -> List[str]:
        """Select optimal distribution channels"""
        channel_effectiveness = {}
        
        for channel in self.channels:
            effectiveness = 0
            cost_factor = 1.0
            
            # Channel-specific logic
            if channel in ["Farmers Market", "Direct Sales"]:
                effectiveness = sum(seg.size * seg.quality_preference for seg in segments)
                cost_factor = 0.7  # Lower cost
            elif channel in ["Online", "Online B2B"]:
                effectiveness = sum(seg.size for seg in segments) * 1.2
                cost_factor = 1.2  # Higher cost
            elif channel in ["Wholesale", "Retail Partnerships"]:
                effectiveness = sum(seg.size for seg in segments if seg.price_sensitivity > 0.6)
                cost_factor = 1.5  # Much higher cost
            
            # Budget consideration
            if budget > 5000:
                cost_factor *= 0.8  # More budget flexibility
            elif budget < 1000:
                cost_factor *= 1.3  # Budget constraints
            
            channel_effectiveness[channel] = effectiveness / cost_factor
        
        # Select top channels within budget
        sorted_channels = sorted(channel_effectiveness.items(), key=lambda x: x[1], reverse=True)
        selected = []
        remaining_budget = budget
        
        for channel, effectiveness in sorted_channels:
            estimated_cost = self._estimate_channel_cost(channel, budget)
            if estimated_cost <= remaining_budget and len(selected) < 3:
                selected.append(channel)
                remaining_budget -= estimated_cost
        
        return selected or [self.channels[0]]  # At least one channel
    
    def _estimate_channel_cost(self, channel: str, total_budget: float) -> float:
        """Estimate cost for a distribution channel"""
        cost_ratios = {
            "Direct Sales": 0.1,
            "Farmers Market": 0.2,
            "Online Local": 0.3,
            "Word of Mouth": 0.05,
            "Wholesale": 0.4,
            "Direct to Restaurant": 0.25,
            "Online B2B": 0.35,
            "Retail Partnerships": 0.5,
            "School Store": 0.15,
            "Community Events": 0.2,
            "Educational Programs": 0.3,
            "Local Partnerships": 0.25
        }
        
        return total_budget * cost_ratios.get(channel, 0.3)
    
    def _plan_promotional_activities(self, budget: float, segments: List[MarketSegment]) -> List[str]:
        """Plan promotional activities based on budget and segments"""
        activities = []
        
        if self.config.operation_scale == OperationScale.PERSONAL:
            if budget > 500:
                activities.extend(["Social Media Posts", "Neighbor Sampling", "Garden Tours"])
            else:
                activities.extend(["Word of Mouth", "Social Media"])
        
        elif self.config.operation_scale == OperationScale.COMMERCIAL:
            if budget > 5000:
                activities.extend(["Trade Shows", "Professional Sales Team", "Quality Certifications", "Digital Marketing"])
            elif budget > 2000:
                activities.extend(["Local Advertising", "Samples to Restaurants", "Online Presence"])
            else:
                activities.extend(["Direct Outreach", "Quality Samples"])
        
        else:  # Educational
            if budget > 1000:
                activities.extend(["School Events", "Educational Workshops", "Community Partnerships", "Parent Engagement"])
            else:
                activities.extend(["Classroom Integration", "Harvest Festivals"])
        
        return activities
    
    def _calculate_reach(self, segments: List[MarketSegment], channels: List[str], budget: float) -> int:
        """Calculate expected customer reach"""
        base_reach = sum(seg.size for seg in segments)
        
        # Channel multipliers
        channel_multiplier = len(channels) * 0.3 + 0.4  # More channels = better reach
        
        # Budget multiplier
        if self.config.operation_scale == OperationScale.PERSONAL:
            budget_multiplier = min(2.0, (budget / 1000) * 0.5 + 0.3)
        elif self.config.operation_scale == OperationScale.COMMERCIAL:
            budget_multiplier = min(3.0, (budget / 10000) * 0.8 + 0.2)
        else:  # Educational
            budget_multiplier = min(2.5, (budget / 2000) * 0.6 + 0.4)
        
        return int(base_reach * channel_multiplier * budget_multiplier)
    
    def _estimate_conversion_rate(self, segments: List[MarketSegment], activities: List[str]) -> float:
        """Estimate conversion rate based on segments and activities"""
        base_conversion = 0.05  # 5% base conversion
        
        # Quality preference boost
        avg_quality_pref = sum(seg.quality_preference for seg in segments) / len(segments)
        quality_boost = avg_quality_pref * 0.03
        
        # Activity boost
        activity_boost = min(0.05, len(activities) * 0.01)
        
        # Scale-specific adjustments
        if self.config.operation_scale == OperationScale.COMMERCIAL:
            base_conversion *= 1.5  # B2B often has higher conversion
        elif self.config.operation_scale == OperationScale.EDUCATIONAL:
            base_conversion *= 1.2  # Community goodwill
        
        return min(0.25, base_conversion + quality_boost + activity_boost)
    
    def _estimate_revenue_impact(self, reach: int, conversion_rate: float, products: List[str]) -> float:
        """Estimate revenue boost from marketing efforts"""
        customers = reach * conversion_rate
        avg_purchase_value = self._estimate_avg_purchase_value(products)
        return customers * avg_purchase_value
    
    def _estimate_avg_purchase_value(self, products: List[str]) -> float:
        """Estimate average purchase value"""
        if self.config.operation_scale == OperationScale.PERSONAL:
            return 15.0  # Small personal purchases
        elif self.config.operation_scale == OperationScale.COMMERCIAL:
            return 200.0  # Bulk B2B orders
        else:  # Educational
            return 25.0  # Community purchases
    
    def _determine_pricing_strategy(self, segments: List[MarketSegment]) -> str:
        """Determine overall pricing strategy"""
        avg_sensitivity = sum(seg.price_sensitivity for seg in segments) / len(segments)
        avg_quality_pref = sum(seg.quality_preference for seg in segments) / len(segments)
        
        if avg_quality_pref > 0.7 and avg_sensitivity < 0.5:
            return "Premium Quality Pricing"
        elif avg_sensitivity > 0.7:
            return "Competitive Volume Pricing"
        else:
            return "Value-Based Pricing"
