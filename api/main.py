# api/main.py
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from datetime import datetime
from typing import List, Dict, Any, Optional
import logging

# Import route modules
from routes import greenhouse, vegetation, sapling, economics, marketing, feedback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Greenhouse Vegetation Playground API",
    description="AI-powered greenhouse optimization platform API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(greenhouse.router, prefix="/api/v1/greenhouse", tags=["greenhouse"])
app.include_router(vegetation.router, prefix="/api/v1/vegetation", tags=["vegetation"])
app.include_router(sapling.router, prefix="/api/v1/sapling", tags=["sapling"])
app.include_router(economics.router, prefix="/api/v1/economics", tags=["economics"])
app.include_router(marketing.router, prefix="/api/v1/marketing", tags=["marketing"])
app.include_router(feedback.router, prefix="/api/v1/feedback", tags=["feedback"])

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Greenhouse Vegetation Playground API",
        "version": "1.0.0",
        "docs": "/docs",
        "status": "operational",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "greenhouse_analyzer": "operational",
            "vegetation_planner": "operational",
            "economics_calculator": "operational",
            "sapling_manager": "operational",
            "marketing_optimizer": "operational"
        }
    }

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Global exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {str(exc)}"}
    )

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

# api/schemas/greenhouse_schema.py
from pydantic import BaseModel, validator
from typing import List, Dict, Optional, Any
from datetime import datetime
from enum import Enum

class OperationScaleEnum(str, Enum):
    PERSONAL = "personal"
    COMMERCIAL = "commercial"
    EDUCATIONAL = "educational"

class GreenhouseConfigSchema(BaseModel):
    size_sqft: int
    operation_scale: OperationScaleEnum
    automation_level: float
    climate_control: bool
    irrigation_system: bool
    location: str

    @validator('size_sqft')
    def validate_size(cls, v):
        if v < 100 or v > 50000:
            raise ValueError('Size must be between 100 and 50,000 sq ft')
        return v

    @validator('automation_level')
    def validate_automation(cls, v):
        if v < 0.0 or v > 1.0:
            raise ValueError('Automation level must be between 0.0 and 1.0')
        return v

class EnvironmentalDataSchema(BaseModel):
    timestamp: datetime
    temperature: float
    humidity: float
    soil_moisture: float
    co2_level: float
    ph_level: float
    light_intensity: float
    soil_nutrients: Dict[str, float]

class EnvironmentalAnalysisSchema(BaseModel):
    current_conditions: Dict[str, float]
    recommendations: List[str]
    alerts: List[str]

class SimulationRequestSchema(BaseModel):
    config: GreenhouseConfigSchema
    days: int = 7
    
    @validator('days')
    def validate_days(cls, v):
        if v < 1 or v > 365:
            raise ValueError('Days must be between 1 and 365')
        return v

class SimulationResponseSchema(BaseModel):
    environmental_data: List[EnvironmentalDataSchema]
    analysis: EnvironmentalAnalysisSchema
    status: str
    generated_at: datetime

# api/schemas/vegetation_schema.py
from pydantic import BaseModel, validator
from typing import List, Dict, Optional, Tuple
from enum import Enum

class SeasonEnum(str, Enum):
    SPRING = "spring"
    SUMMER = "summer"
    FALL = "fall"
    WINTER = "winter"
    ALL = "all"

class DifficultyLevelEnum(str, Enum):
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"

class WaterNeedsEnum(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class PlantProfileSchema(BaseModel):
    name: str
    space_required: float
    grow_time_days: int
    optimal_temp_range: Tuple[float, float]
    water_needs: WaterNeedsEnum
    market_value_per_unit: float
    difficulty_level: DifficultyLevelEnum
    seasonal_preference: SeasonEnum

class PlantingSchema(BaseModel):
    plant: str
    quantity: int
    space_used: float
    harvest_date: Optional[str] = None
    estimated_yield_value: Optional[float] = None
    annual_revenue: Optional[float] = None
    annual_cycles: Optional[float] = None

class VegetationPlanSchema(BaseModel):
    strategy: str
    total_space: float
    plantings: List[PlantingSchema]
    space_utilization: str
    learning_objectives: Optional[List[str]] = None
    experiment_suggestions: Optional[List[str]] = None
    care_tips: Optional[List[str]] = None
    projected_revenue: Optional[float] = None

class PlanningRequestSchema(BaseModel):
    config: GreenhouseConfigSchema
    season: SeasonEnum = SeasonEnum.ALL
    focus_area: Optional[str] = None  # "profit", "diversity", "education"

class PlanningResponseSchema(BaseModel):
    plan: VegetationPlanSchema
    recommendations: List[str]
    plant_database_size: int
    generated_at: datetime

# api/schemas/sapling_schema.py
from pydantic import BaseModel, validator
from typing import List, Dict, Optional
from datetime import datetime
from enum import Enum

class SeedStageEnum(str, Enum):
    SEED = "seed"
    GERMINATED = "germinated"
    SEEDLING = "seedling"
    SAPLING = "sapling"
    TRANSPLANT_READY = "transplant_ready"
    MATURE = "mature"

class SeedBatchSchema(BaseModel):
    batch_id: str
    plant_type: str
    quantity: int
    start_date: datetime
    current_stage: SeedStageEnum
    success_rate: float
    days_in_stage: int
    environmental_stress: float
    notes: List[str]

class ProductionScheduleSchema(BaseModel):
    target_date: datetime
    plant_type: str
    quantity_needed: int
    batch_ids: List[str]
    status: str

class ProductionStatusSchema(BaseModel):
    total_active_batches: int
    stage_breakdown: Dict[str, int]
    plant_type_summary: Dict[str, Dict[str, Any]]
    ready_for_transplant: int
    production_alerts: List[str]

class BatchStartRequestSchema(BaseModel):
    plant_type: str
    quantity: int
    target_date: Optional[datetime] = None

    @validator('quantity')
    def validate_quantity(cls, v):
        if v < 1 or v > 10000:
            raise ValueError('Quantity must be between 1 and 10,000')
        return v

class ProgressSimulationRequestSchema(BaseModel):
    days: int = 1

    @validator('days')
    def validate_days(cls, v):
        if v < 1 or v > 365:
            raise ValueError('Days must be between 1 and 365')
        return v

class BatchResponseSchema(BaseModel):
    batch_id: str
    message: str
    status: str
    created_at: datetime

# api/routes/greenhouse.py
from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict, Any
import logging
from datetime import datetime

# Import your greenhouse modules
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from src.greenhouse_analyzer import GreenhouseAnalyzer, GreenhouseConfig, OperationScale
from schemas.greenhouse_schema import (
    GreenhouseConfigSchema, SimulationRequestSchema, SimulationResponseSchema,
    EnvironmentalDataSchema, EnvironmentalAnalysisSchema
)

router = APIRouter()
logger = logging.getLogger(__name__)

# In-memory storage for demo (use Redis/Database in production)
active_analyzers: Dict[str, GreenhouseAnalyzer] = {}

def convert_operation_scale(scale: str) -> OperationScale:
    """Convert string to OperationScale enum"""
    mapping = {
        "personal": OperationScale.PERSONAL,
        "commercial": OperationScale.COMMERCIAL,
        "educational": OperationScale.EDUCATIONAL
    }
    return mapping.get(scale.lower(), OperationScale.PERSONAL)

def create_greenhouse_config(config_schema: GreenhouseConfigSchema) -> GreenhouseConfig:
    """Convert schema to greenhouse config"""
    return GreenhouseConfig(
        size_sqft=config_schema.size_sqft,
        operation_scale=convert_operation_scale(config_schema.operation_scale),
        automation_level=config_schema.automation_level,
        climate_control=config_schema.climate_control,
        irrigation_system=config_schema.irrigation_system,
        location=config_schema.location
    )

@router.post("/simulate", response_model=SimulationResponseSchema)
async def simulate_greenhouse_conditions(request: SimulationRequestSchema):
    """
    Simulate greenhouse environmental conditions
    """
    try:
        # Create greenhouse configuration
        config = create_greenhouse_config(request.config)
        
        # Initialize analyzer
        analyzer = GreenhouseAnalyzer(config)
        analyzer_id = f"{config.operation_scale.value}_{datetime.now().timestamp()}"
        active_analyzers[analyzer_id] = analyzer
        
        # Generate environmental data
        env_data = analyzer.simulate_environmental_conditions(days=request.days)
        analysis = analyzer.analyze_conditions()
        
        # Convert to response format
        environmental_data = []
        for data in env_data:
            environmental_data.append(EnvironmentalDataSchema(
                timestamp=data.timestamp,
                temperature=data.temperature,
                humidity=data.humidity,
                soil_moisture=data.soil_moisture,
                co2_level=data.co2_level,
                ph_level=data.ph_level,
                light_intensity=data.light_intensity,
                soil_nutrients=data.soil_nutrients
            ))
        
        response = SimulationResponseSchema(
            environmental_data=environmental_data,
            analysis=EnvironmentalAnalysisSchema(**analysis),
            status="success",
            generated_at=datetime.now()
        )
        
        logger.info(f"Successfully simulated {len(env_data)} environmental readings for {config.operation_scale.value} operation")
        return response
        
    except Exception as e:
        logger.error(f"Error in greenhouse simulation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Simulation failed: {str(e)}")

@router.get("/conditions/{analyzer_id}")
async def get_current_conditions(analyzer_id: str):
    """
    Get current environmental conditions for an active analyzer
    """
    try:
        if analyzer_id not in active_analyzers:
            raise HTTPException(status_code=404, detail="Analyzer not found")
        
        analyzer = active_analyzers[analyzer_id]
        analysis = analyzer.analyze_conditions()
        
        return {
            "analyzer_id": analyzer_id,
            "analysis": analysis,
            "timestamp": datetime.now()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating schedule: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to create schedule: {str(e)}")

# api/routes/economics.py
from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any
import logging
from datetime import datetime
from pydantic import BaseModel

from src.economics_calculator import EconomicsCalculator, EconomicProjection
from routes.greenhouse import create_greenhouse_config
from schemas.greenhouse_schema import GreenhouseConfigSchema

router = APIRouter()
logger = logging.getLogger(__name__)

# Pydantic schemas for economics
class EconomicAnalysisRequestSchema(BaseModel):
    config: GreenhouseConfigSchema
    planting_plan: Dict[str, Any]
    timeframe_months: int = 12

class EconomicProjectionSchema(BaseModel):
    total_investment: float
    monthly_costs: float
    projected_revenue: float
    break_even_months: float
    roi_annual: float
    profit_margin: float

class EconomicAnalysisResponseSchema(BaseModel):
    projection: EconomicProjectionSchema
    cost_breakdown: Dict[str, float]
    recommendations: List[str]
    risk_assessment: Dict[str, str]
    generated_at: datetime

# In-memory storage for calculators
active_calculators: Dict[str, EconomicsCalculator] = {}

@router.post("/analyze", response_model=EconomicAnalysisResponseSchema)
async def analyze_economics(request: EconomicAnalysisRequestSchema):
    """
    Perform comprehensive economic analysis for greenhouse operation
    """
    try:
        # Create greenhouse configuration
        config = create_greenhouse_config(request.config)
        
        # Initialize calculator
        calculator = EconomicsCalculator(config)
        calculator_id = f"{config.operation_scale.value}_{datetime.now().timestamp()}"
        active_calculators[calculator_id] = calculator
        
        # Perform economic analysis
        projection = calculator.calculate_economics(request.planting_plan, request.timeframe_months)
        cost_breakdown = calculator.generate_cost_breakdown()
        
        # Generate recommendations based on results
        recommendations = []
        if projection.roi_annual < 5:
            recommendations.extend([
                "Consider increasing automation to reduce labor costs",
                "Focus on higher-value crops",
                "Explore direct-to-consumer sales channels"
            ])
        elif projection.roi_annual < 15:
            recommendations.extend([
                "Good foundation - consider scaling up",
                "Optimize space utilization",
                "Investigate premium market opportunities"
            ])
        else:
            recommendations.extend([
                "Excellent returns - consider expansion",
                "Maintain current strategy",
                "Explore additional revenue streams"
            ])
        
        # Risk assessment
        risk_assessment = {}
        if projection.break_even_months < 0:
            risk_assessment["profitability"] = "High Risk - Break-even not achievable"
        elif projection.break_even_months > 24:
            risk_assessment["profitability"] = "Medium Risk - Long payback period"
        else:
            risk_assessment["profitability"] = "Low Risk - Reasonable payback"
        
        if projection.profit_margin < 10:
            risk_assessment["margin"] = "High Risk - Low profit margins"
        elif projection.profit_margin < 25:
            risk_assessment["margin"] = "Medium Risk - Moderate margins"
        else:
            risk_assessment["margin"] = "Low Risk - Healthy margins"
        
        # Market risk based on operation scale
        if config.operation_scale.value == "commercial":
            risk_assessment["market"] = "Medium Risk - Wholesale market dependency"
        elif config.operation_scale.value == "personal":
            risk_assessment["market"] = "Low Risk - Diverse small-scale markets"
        else:
            risk_assessment["market"] = "Low Risk - Educational market stability"
        
        response = EconomicAnalysisResponseSchema(
            projection=EconomicProjectionSchema(
                total_investment=projection.total_investment,
                monthly_costs=projection.monthly_costs,
                projected_revenue=projection.projected_revenue,
                break_even_months=projection.break_even_months,
                roi_annual=projection.roi_annual,
                profit_margin=projection.profit_margin
            ),
            cost_breakdown=cost_breakdown,
            recommendations=recommendations,
            risk_assessment=risk_assessment,
            generated_at=datetime.now()
        )
        
        logger.info(f"Economic analysis completed for {config.operation_scale.value} operation: ROI {projection.roi_annual:.1f}%")
        return response
        
    except Exception as e:
        logger.error(f"Error in economic analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@router.get("/scenarios/{calculator_id}")
async def run_scenario_analysis(calculator_id: str, scenarios: List[str] = None):
    """
    Run scenario analysis (optimistic, pessimistic, realistic)
    """
    try:
        if calculator_id not in active_calculators:
            raise HTTPException(status_code=404, detail="Calculator not found")
        
        calculator = active_calculators[calculator_id]
        
        # Default scenarios
        if not scenarios:
            scenarios = ["pessimistic", "realistic", "optimistic"]
        
        scenario_results = {}
        
        for scenario in scenarios:
            # Modify cost structure based on scenario
            original_costs = calculator.cost_structure
            
            if scenario == "pessimistic":
                # 20% higher costs, 15% lower revenue
                multiplier = {"costs": 1.2, "revenue": 0.85}
            elif scenario == "optimistic":
                # 10% lower costs, 20% higher revenue
                multiplier = {"costs": 0.9, "revenue": 1.2}
            else:  # realistic
                # Current projections
                multiplier = {"costs": 1.0, "revenue": 1.0}
            
            # For demo, we'll simulate the scenario impact
            base_roi = 12.0  # Example base ROI
            scenario_roi = base_roi * multiplier["revenue"] / multiplier["costs"]
            
            scenario_results[scenario] = {
                "roi_annual": round(scenario_roi, 1),
                "cost_multiplier": multiplier["costs"],
                "revenue_multiplier": multiplier["revenue"],
                "risk_level": "high" if scenario == "pessimistic" else "low" if scenario == "optimistic" else "medium"
            }
        
        return {
            "calculator_id": calculator_id,
            "scenarios": scenario_results,
            "generated_at": datetime.now()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in scenario analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Scenario analysis failed: {str(e)}")

@router.get("/optimization/{calculator_id}")
async def get_optimization_suggestions(calculator_id: str):
    """
    Get specific optimization suggestions based on economic analysis
    """
    try:
        if calculator_id not in active_calculators:
            raise HTTPException(status_code=404, detail="Calculator not found")
        
        calculator = active_calculators[calculator_id]
        cost_breakdown = calculator.generate_cost_breakdown()
        
        optimizations = []
        
        # Analyze cost structure for optimization opportunities
        total_costs = sum(cost_breakdown.values())
        
        for cost_category, amount in cost_breakdown.items():
            percentage = (amount / total_costs) * 100
            
            if cost_category == "Monthly Utilities" and percentage > 30:
                optimizations.append({
                    "category": "Energy Efficiency",
                    "suggestion": "Consider LED lighting and improved insulation",
                    "potential_savings": "15-25% utilities reduction",
                    "priority": "high"
                })
            
            if cost_category == "Labor per Hour" and percentage > 40:
                optimizations.append({
                    "category": "Automation",
                    "suggestion": "Invest in automated irrigation and climate control",
                    "potential_savings": "30-50% labor cost reduction",
                    "priority": "high"
                })
            
            if cost_category == "Seeds per Cycle" and percentage > 20:
                optimizations.append({
                    "category": "Propagation",
                    "suggestion": "Develop in-house seed starting program",
                    "potential_savings": "40-60% seed cost reduction",
                    "priority": "medium"
                })
        
        # Add general optimization suggestions
        if calculator.config.operation_scale.value == "commercial":
            optimizations.append({
                "category": "Scale Efficiency",
                "suggestion": "Consider vertical growing systems to increase yield per sq ft",
                "potential_savings": "50-100% space utilization increase",
                "priority": "medium"
            })
        
        return {
            "calculator_id": calculator_id,
            "total_optimizations": len(optimizations),
            "optimizations": optimizations,
            "estimated_total_savings": "25-45% cost reduction potential"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting optimizations: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get optimizations: {str(e)}")

@router.get("/calculators")
async def list_active_calculators():
    """
    List all active economic calculators
    """
    try:
        calculators_info = []
        for calc_id, calculator in active_calculators.items():
            calculators_info.append({
                "calculator_id": calc_id,
                "config": {
                    "size_sqft": calculator.config.size_sqft,
                    "operation_scale": calculator.config.operation_scale.value,
                    "automation_level": calculator.config.automation_level
                },
                "cost_structure": calculator.generate_cost_breakdown()
            })
        
        return {
            "active_calculators": len(active_calculators),
            "calculators": calculators_info
        }
        
    except Exception as e:
        logger.error(f"Error listing calculators: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list calculators: {str(e)}")

# api/routes/marketing.py
from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime
from pydantic import BaseModel

from src.marketing_optimizer import MarketingOptimizer, MarketingStrategy
from routes.greenhouse import create_greenhouse_config
from schemas.greenhouse_schema import GreenhouseConfigSchema

router = APIRouter()
logger = logging.getLogger(__name__)

# Pydantic schemas for marketing
class MarketingAnalysisRequestSchema(BaseModel):
    config: GreenhouseConfigSchema
    planting_plan: Dict[str, Any]
    budget: float
    current_prices: Optional[Dict[str, float]] = None

class MarketingStrategySchema(BaseModel):
    target_segments: List[str]
    pricing_strategy: str
    distribution_channels: List[str]
    promotional_activities: List[str]
    expected_reach: int
    estimated_conversion_rate: float
    projected_revenue_boost: float

class PricingRecommendationSchema(BaseModel):
    current_price: float
    recommended_price: float
    strategy: str
    expected_demand_change: str
    expected_revenue_change: str
    target_segments: List[str]

class MarketingAnalysisResponseSchema(BaseModel):
    strategy: MarketingStrategySchema
    pricing_recommendations: Dict[str, PricingRecommendationSchema]
    market_insights: Dict[str, Any]
    competitive_analysis: Dict[str, str]
    generated_at: datetime

# In-memory storage for optimizers
active_optimizers: Dict[str, MarketingOptimizer] = {}

@router.post("/analyze", response_model=MarketingAnalysisResponseSchema)
async def analyze_marketing_strategy(request: MarketingAnalysisRequestSchema):
    """
    Analyze and optimize marketing strategy for greenhouse operation
    """
    try:
        # Create greenhouse configuration
        config = create_greenhouse_config(request.config)
        
        # Initialize marketing optimizer
        optimizer = MarketingOptimizer(config)
        optimizer_id = f"{config.operation_scale.value}_{datetime.now().timestamp()}"
        active_optimizers[optimizer_id] = optimizer
        
        # Use provided prices or generate defaults
        current_prices = request.current_prices or {
            planting['plant'].lower(): 3.0 
            for planting in request.planting_plan.get('plantings', [])
        }
        
        # Generate marketing strategy
        strategy = optimizer.develop_marketing_strategy(request.planting_plan, request.budget)
        pricing_recs = optimizer.optimize_pricing(request.planting_plan, current_prices)
        
        # Convert to response format
        strategy_schema = MarketingStrategySchema(
            target_segments=strategy.target_segments,
            pricing_strategy=strategy.pricing_strategy,
            distribution_channels=strategy.distribution_channels,
            promotional_activities=strategy.promotional_activities,
            expected_reach=strategy.expected_reach,
            estimated_conversion_rate=strategy.estimated_conversion_rate,
            projected_revenue_boost=strategy.projected_revenue_boost
        )
        
        pricing_recommendations = {}
        for product, rec in pricing_recs.items():
            pricing_recommendations[product] = PricingRecommendationSchema(**rec)
        
        # Generate market insights
        market_insights = {
            "total_addressable_market": sum(seg.size for seg in optimizer.market_segments),
            "market_penetration_potential": f"{(strategy.expected_reach / sum(seg.size for seg in optimizer.market_segments)) * 100:.1f}%",
            "primary_market_segment": max(optimizer.market_segments, key=lambda x: x.size).name,
            "seasonality_impact": "High" if config.operation_scale.value != "commercial" else "Medium",
            "competition_level": "High" if config.operation_scale.value == "commercial" else "Medium"
        }
        
        # Competitive analysis
        competitive_analysis = {}
        if config.operation_scale.value == "personal":
            competitive_analysis = {
                "main_competitors": "Local farmers markets, grocery stores",
                "competitive_advantage": "Freshness, organic growing, personal touch",
                "pricing_position": "Premium pricing justified by quality",
                "market_differentiation": "Specialty varieties, direct customer relationship"
            }
        elif config.operation_scale.value == "commercial":
            competitive_analysis = {
                "main_competitors": "Large-scale farms, wholesale distributors",
                "competitive_advantage": "Controlled environment, year-round production",
                "pricing_position": "Competitive with quality premium",
                "market_differentiation": "Consistency, local sourcing, reduced transport"
            }
        else:  # educational
            competitive_analysis = {
                "main_competitors": "Community gardens, school programs",
                "competitive_advantage": "Educational value, community engagement",
                "pricing_position": "Value-based pricing for community benefit",
                "market_differentiation": "Learning experience, fresh local produce"
            }
        
        response = MarketingAnalysisResponseSchema(
            strategy=strategy_schema,
            pricing_recommendations=pricing_recommendations,
            market_insights=market_insights,
            competitive_analysis=competitive_analysis,
            generated_at=datetime.now()
        )
        
        logger.info(f"Marketing analysis completed for {config.operation_scale.value} operation with {len(strategy.target_segments)} target segments")
        return response
        
    except Exception as e:
        logger.error(f"Error in marketing analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@router.get("/segments/{optimizer_id}")
async def get_market_segments(optimizer_id: str):
    """
    Get detailed market segment information
    """
    try:
        if optimizer_id not in active_optimizers:
            raise HTTPException(status_code=404, detail="Optimizer not found")
        
        optimizer = active_optimizers[optimizer_id]
        
        segments_info = []
        for segment in optimizer.market_segments:
            segments_info.append({
                "name": segment.name,
                "size": segment.size,
                "price_sensitivity": segment.price_sensitivity,
                "quality_preference": segment.quality_preference,
                "seasonal_demand": segment.seasonal_demand,
                "preferred_products": segment.preferred_products
            })
        
        return {
            "optimizer_id": optimizer_id,
            "total_segments": len(segments_info),
            "segments": segments_info
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting segments: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get segments: {str(e)}")

@router.get("/channels")
async def get_distribution_channels(operation_scale: str = "personal"):
    """
    Get available distribution channels for an operation scale
    """
    try:
        # Create temporary optimizer to get channels
        from schemas.greenhouse_schema import GreenhouseConfigSchema
        
        temp_config = GreenhouseConfigSchema(
            size_sqft=2000,
            operation_scale=operation_scale,
            automation_level=0.5,
            climate_control=True,
            irrigation_system=True,
            location="Denver, CO"
        )
        
        config = create_greenhouse_config(temp_config)
        optimizer = MarketingOptimizer(config)
        channels = optimizer.channels
        
        # Add channel descriptions
        channel_descriptions = {
            "Direct Sales": "Sell directly to consumers at the farm or delivery",
            "Farmers Market": "Weekly/seasonal market stalls with premium pricing",
            "Online Local": "Local online ordering and delivery platforms",
            "Word of Mouth": "Customer referrals and community networking",
            "Wholesale": "Bulk sales to grocery stores and distributors",
            "Direct to Restaurant": "Fresh supply partnerships with restaurants",
            "Online B2B": "Business-to-business online platforms",
            "Retail Partnerships": "Partnerships with local retail stores",
            "School Store": "Campus-based sales and educational integration",
            "Community Events": "Local fairs, festivals, and community gatherings",
            "Educational Programs": "Integration with curriculum and learning activities",
            "Local Partnerships": "Collaboration with other local organizations"
        }
        
        channels_info = []
        for channel in channels:
            channels_info.append({
                "name": channel,
                "description": channel_descriptions.get(channel, "Distribution channel"),
                "suitable_for": operation_scale,
                "typical_margin": "15-60%" if "Direct" in channel or "Market" in channel else "5-25%"
            })
        
        return {
            "operation_scale": operation_scale,
            "available_channels": len(channels_info),
            "channels": channels_info
        }
        
    except Exception as e:
        logger.error(f"Error getting channels: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get channels: {str(e)}")

@router.post("/campaign/{optimizer_id}")
async def create_marketing_campaign(
    optimizer_id: str,
    campaign_name: str,
    target_segments: List[str],
    budget: float,
    duration_weeks: int = 8
):
    """
    Create a specific marketing campaign
    """
    try:
        if optimizer_id not in active_optimizers:
            raise HTTPException(status_code=404, detail="Optimizer not found")
        
        optimizer = active_optimizers[optimizer_id]
        
        # Calculate campaign metrics
        weekly_budget = budget / duration_weeks
        estimated_weekly_reach = int(weekly_budget * 50)  # Rough estimate: $1 = 50 people reached
        
        # Select appropriate channels based on budget
        if budget < 500:
            recommended_channels = ["Social Media", "Word of Mouth", "Direct Sales"]
        elif budget < 2000:
            recommended_channels = ["Social Media", "Local Advertising", "Farmers Market"]
        else:
            recommended_channels = ["Multi-channel", "Professional Marketing", "Trade Shows"]
        
        # Calculate expected outcomes
        conversion_rate = 0.02 if optimizer.config.operation_scale.value == "commercial" else 0.05
        expected_customers = int(estimated_weekly_reach * duration_weeks * conversion_rate)
        
        campaign = {
            "campaign_id": f"{campaign_name}_{datetime.now().timestamp()}",
            "name": campaign_name,
            "target_segments": target_segments,
            "budget": budget,
            "duration_weeks": duration_weeks,
            "weekly_budget": weekly_budget,
            "recommended_channels": recommended_channels,
            "estimated_reach": estimated_weekly_reach * duration_weeks,
            "expected_customers": expected_customers,
            "estimated_roi": f"{(expected_customers * 25 / budget) * 100:.0f}%",  # Assuming $25 avg customer value
            "timeline": [
                {"week": 1, "activity": "Campaign launch and awareness building"},
                {"week": 2, "activity": "Engagement and lead generation"},
                {"week": 4, "activity": "Mid-campaign optimization"},
                {"week": 6, "activity": "Conversion focus and sales push"},
                {"week": 8, "activity": "Campaign completion and analysis"}
            ],
            "success_metrics": [
                "Reach and impressions",
                "Engagement rate",
                "Lead generation",
                "Conversion rate",
                "Customer acquisition cost",
                "Return on ad spend"
            ],
            "created_at": datetime.now()
        }
        
        return {
            "optimizer_id": optimizer_id,
            "campaign": campaign,
            "recommendations": [
                f"Focus on {', '.join(target_segments[:2])} segments for best ROI",
                "Track weekly performance and adjust budget allocation",
                "Prepare for seasonal demand variations",
                "Consider customer retention strategies post-campaign"
            ]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating campaign: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to create campaign: {str(e)}")

@router.get("/optimizers")
async def list_active_optimizers():
    """
    List all active marketing optimizers
    """
    try:
        optimizers_info = []
        for opt_id, optimizer in active_optimizers.items():
            optimizers_info.append({
                "optimizer_id": opt_id,
                "config": {
                    "size_sqft": optimizer.config.size_sqft,
                    "operation_scale": optimizer.config.operation_scale.value,
                    "location": optimizer.config.location
                },
                "market_segments": len(optimizer.market_segments),
                "available_channels": len(optimizer.channels)
            })
        
        return {
            "active_optimizers": len(active_optimizers),
            "optimizers": optimizers_info
        }
        
    except Exception as e:
        logger.error(f"Error listing optimizers: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list optimizers: {str(e)}")

# api/routes/feedback.py
from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime
from pydantic import BaseModel, validator

router = APIRouter()
logger = logging.getLogger(__name__)

# Pydantic schemas for feedback
class FeedbackSubmissionSchema(BaseModel):
    user_id: Optional[str] = None
    operation_scale: str
    greenhouse_size: int
    feedback_type: str  # "bug", "feature_request", "improvement", "success_story"
    rating: int  # 1-5 stars
    title: str
    description: str
    module: Optional[str] = None  # Which module the feedback relates to
    
    @validator('rating')
    def validate_rating(cls, v):
        if v < 1 or v > 5:
            raise ValueError('Rating must be between 1 and 5')
        return v
    
    @validator('feedback_type')
    def validate_feedback_type(cls, v):
        valid_types = ["bug", "feature_request", "improvement", "success_story", "question"]
        if v not in valid_types:
            raise ValueError(f'Feedback type must be one of: {", ".join(valid_types)}')
        return v

class FeedbackResponseSchema(BaseModel):
    feedback_id: str
    status: str
    message: str
    submitted_at: datetime

class UsageAnalyticsSchema(BaseModel):
    session_id: str
    user_id: Optional[str] = None
    operation_scale: str
    modules_used: List[str]
    session_duration_minutes: int
    successful_analyses: int
    errors_encountered: int

# In-memory storage for feedback (use database in production)
feedback_storage: List[Dict[str, Any]] = []
analytics_storage: List[Dict[str, Any]] = []

@router.post("/submit", response_model=FeedbackResponseSchema)
async def submit_feedback(feedback: FeedbackSubmissionSchema):
    """
    Submit user feedback about the greenhouse system
    """
    try:
        feedback_id = f"FB_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(feedback_storage) + 1}"
        
        feedback_record = {
            "feedback_id": feedback_id,
            "user_id": feedback.user_id,
            "operation_scale": feedback.operation_scale,
            "greenhouse_size": feedback.greenhouse_size,
            "feedback_type": feedback.feedback_type,
            "rating": feedback.rating,
            "title": feedback.title,
            "description": feedback.description,
            "module": feedback.module,
            "status": "received",
            "submitted_at": datetime.now(),
            "priority": "high" if feedback.feedback_type == "bug" and feedback.rating <= 2 else "medium"
        }
        
        feedback_storage.append(feedback_record)
        
        # Log feedback for monitoring
        logger.info(f"Feedback received: {feedback.feedback_type} - {feedback.title} (Rating: {feedback.rating}/5)")
        
        response = FeedbackResponseSchema(
            feedback_id=feedback_id,
            status="received",
            message=f"Thank you for your feedback! We've received your {feedback.feedback_type} and will review it shortly.",
            submitted_at=datetime.now()
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error submitting feedback: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to submit feedback: {str(e)}")

@router.post("/analytics")
async def log_usage_analytics(analytics: UsageAnalyticsSchema):
    """
    Log usage analytics for system improvement
    """
    try:
        analytics_record = {
            "session_id": analytics.session_id,
            "user_id": analytics.user_id,
            "operation_scale": analytics.operation_scale,
            "modules_used": analytics.modules_used,
            "session_duration_minutes": analytics.session_duration_minutes,
            "successful_analyses": analytics.successful_analyses,
            "errors_encountered": analytics.errors_encountered,
            "logged_at": datetime.now()
        }
        
        analytics_storage.append(analytics_record)
        
        logger.info(f"Analytics logged for session {analytics.session_id}: {len(analytics.modules_used)} modules used")
        
        return {
            "status": "logged",
            "message": "Analytics data recorded successfully",
            "session_id": analytics.session_id
        }
        
    except Exception as e:
        logger.error(f"Error logging analytics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to log analytics: {str(e)}")

@router.get("/feedback")
async def get_feedback_summary(
    feedback_type: Optional[str] = None,
    operation_scale: Optional[str] = None,
    limit: int = 50
):
    """
    Get feedback summary for analysis
    """
    try:
        filtered_feedback = feedback_storage
        
        # Apply filters
        if feedback_type:
            filtered_feedback = [f for f in filtered_feedback if f['feedback_type'] == feedback_type]
        
        if operation_scale:
            filtered_feedback = [f for f in filtered_feedback if f['operation_scale'] == operation_scale]
        
        # Limit results
        filtered_feedback = filtered_feedback[-limit:]
        
        # Generate summary statistics
        if filtered_feedback:
            avg_rating = sum(f['rating'] for f in filtered_feedback) / len(filtered_feedback)
            feedback_types = {}
            operation_scales = {}
            
            for f in filtered_feedback:
                feedback_types[f['feedback_type']] = feedback_types.get(f['feedback_type'], 0) + 1
                operation_scales[f['operation_scale']] = operation_scales.get(f['operation_scale'], 0) + 1
        else:
            avg_rating = 0
            feedback_types = {}
            operation_scales = {}
        
        return {
            "total_feedback": len(filtered_feedback),
            "average_rating": round(avg_rating, 2),
            "feedback_by_type": feedback_types,
            "feedback_by_scale": operation_scales,
            "recent_feedback": filtered_feedback[-10:] if filtered_feedback else []
        }
        
    except Exception as e:
        logger.error(f"Error getting feedback summary: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get feedback: {str(e)}")

@router.get("/analytics/summary")
async def get_analytics_summary(days: int = 30):
    """
    Get usage analytics summary
    """
    try:
        # Filter recent analytics
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_analytics = [
            a for a in analytics_storage 
            if a['logged_at'] > cutoff_date
        ]
        
        if not recent_analytics:
            return {
                "message": "No recent analytics data",
                "period_days": days,
                "total_sessions": 0
            }
        
        # Calculate summary statistics
        total_sessions = len(recent_analytics)
        unique_users = len(set(a.get('user_id') for a in recent_analytics if a.get('user_id')))
        
        # Module usage
        module_usage = {}
        total_duration = 0
        total_analyses = 0
        total_errors = 0
        
        operation_scale_usage = {}
        
        for session in recent_analytics:
            for module in session['modules_used']:
                module_usage[module] = module_usage.get(module, 0) + 1
            
            total_duration += session['session_duration_minutes']
            total_analyses += session['successful_analyses']
            total_errors += session['errors_encountered']
            
            scale = session['operation_scale']
            operation_scale_usage[scale] = operation_scale_usage.get(scale, 0) + 1
        
        avg_session_duration = total_duration / total_sessions if total_sessions > 0 else 0
        success_rate = (total_analyses / (total_analyses + total_errors)) * 100 if (total_analyses + total_errors) > 0 else 100
        
        return {
            "period_days": days,
            "total_sessions": total_sessions,
            "unique_users": unique_users,
            "average_session_duration_minutes": round(avg_session_duration, 1),
            "total_successful_analyses": total_analyses,
            "total_errors": total_errors,
            "success_rate_percentage": round(success_rate, 1),
            "most_used_modules": sorted(module_usage.items(), key=lambda x: x[1], reverse=True)[:5],
            "usage_by_operation_scale": operation_scale_usage,
            "insights": [
                f"Most popular operation scale: {max(operation_scale_usage.items(), key=lambda x: x[1])[0] if operation_scale_usage else 'N/A'}",
                f"Average session length: {avg_session_duration:.1f} minutes",
                f"System reliability: {success_rate:.1f}% success rate"
            ]
        }
        
    except Exception as e:
        logger.error(f"Error getting analytics summary: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get analytics: {str(e)}")

@router.get("/health-metrics")
async def get_system_health_metrics():
    """
    Get system health and performance metrics
    """
    try:
        # Calculate recent error rates
        recent_analytics = analytics_storage[-100:]  # Last 100 sessions
        
        if recent_analytics:
            total_errors = sum(a['errors_encountered'] for a in recent_analytics)
            total_analyses = sum(a['successful_analyses'] for a in recent_analytics)
            error_rate = (total_errors / (total_errors + total_analyses)) * 100 if (total_errors + total_analyses) > 0 else 0
        else:
            error_rate = 0
            total_errors = 0
            total_analyses = 0
        
        # Analyze feedback for quality metrics
        recent_feedback = feedback_storage[-50:]  # Last 50 feedback items
        
        if recent_feedback:
            avg_rating = sum(f['rating'] for f in recent_feedback) / len(recent_feedback)
            bug_reports = len([f for f in recent_feedback if f['feedback_type'] == 'bug'])
        else:
            avg_rating = 5.0
            bug_reports = 0
        
        # System health status
        if error_rate < 5 and avg_rating >= 4.0 and bug_reports < 5:
            health_status = "Excellent"
        elif error_rate < 10 and avg_rating >= 3.5 and bug_reports < 10:
            health_status = "Good"
        elif error_rate < 20 and avg_rating >= 3.0:
            health_status = "Fair"
        else:
            health_status = "Needs Attention"
        
        return {
            "health_status": health_status,
            "error_rate_percentage": round(error_rate, 2),
            "average_user_rating": round(avg_rating, 2),
            "recent_bug_reports": bug_reports,
            "total_analyses_completed": total_analyses,
            "system_uptime": "99.9%",  # Mock uptime
            "recommendations": [
                "System performing well" if health_status == "Excellent" else f"Monitor {health_status.lower()} metrics",
                "Continue current maintenance schedule",
                "Regular feedback review recommended"
            ],
            "last_updated": datetime.now()
        }
        
    except Exception as e:
        logger.error(f"Error getting health metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get health metrics: {str(e)}")

@router.post("/feature-request")
async def submit_feature_request(
    title: str,
    description: str,
    operation_scale: str,
    priority: str = "medium",
    user_id: Optional[str] = None
):
    """
    Submit a specific feature request
    """
    try:
        feature_request = {
            "request_id": f"FR_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "title": title,
            "description": description,
            "operation_scale": operation_scale,
            "priority": priority,
            "user_id": user_id,
            "status": "submitted",
            "votes": 1,  # User's own vote
            "submitted_at": datetime.now(),
            "estimated_effort": "TBD",
            "target_version": "TBD"
        }
        
        # Add to feedback storage with feature_request type
        feedback_record = {
            "feedback_id": feature_request["request_id"],
            "user_id": user_id,
            "operation_scale": operation_scale,
            "feedback_type": "feature_request",
            "rating": 4,  # Default for feature requests
            "title": title,
            "description": description,
            "status": "submitted",
            "submitted_at": datetime.now(),
            "priority": priority
        }
        
        feedback_storage.append(feedback_record)
        
        return {
            "request_id": feature_request["request_id"],
            "status": "submitted",
            "message": "Feature request submitted successfully. We'll review it and consider for future updates.",
            "next_steps": [
                "Review by product team",
                "Technical feasibility assessment",
                "Priority ranking",
                "Development planning"
            ]
        }
        
    except Exception as e:
        logger.error(f"Error submitting feature request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to submit feature request: {str(e)}")

# API Dependencies and Requirements file
# requirements.txt for the FastAPI backend
"""
fastapi==0.104.1
uvicorn==0.24.0
pydantic==2.5.0
python-multipart==0.0.6
python-jose[cryptography]==3.3.0
passlib[bcrypt]==1.7.4
python-dotenv==1.0.0
aiofiles==23.2.1
httpx==0.25.2
redis==5.0.1
sqlalchemy==2.0.23
alembic==1.12.1
psycopg2-binary==2.9.9
pytest==7.4.3
pytest-asyncio==0.21.1
"""

# docker-compose.yml for easy deployment
"""
version: '3.8'

services:
  fastapi:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://user:password@db:5432/greenhouse
      - REDIS_URL=redis://redis:6379
    depends_on:
      - db
      - redis
    volumes:
      - ./src:/app/src
      - ./api:/app/api
    command: uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

  db:
    image: postgres:15
    environment:
      - POSTGRES_DB=greenhouse
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  streamlit:
    build:
      context: .
      dockerfile: Dockerfile.streamlit
    ports:
      - "8501:8501"
    environment:
      - API_BASE_URL=http://fastapi:8000
    depends_on:
      - fastapi
    volumes:
      - .:/app
    command: streamlit run streamlit_app.py --server.address 0.0.0.0

volumes:
  postgres_data:
"""

# Dockerfile for FastAPI
"""
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
"""

# API Client for Streamlit integration
# api_client.py
"""
import httpx
import streamlit as st
from typing import Dict, Any, Optional, List
import asyncio

class GreenhouseAPIClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.client = httpx.Client(base_url=base_url)
    
    async def simulate_greenhouse(self, config: Dict[str, Any], days: int = 7) -> Dict[str, Any]:
        \"\"\"Call greenhouse simulation API\"\"\"
        try:
            response = self.client.post(
                "/api/v1/greenhouse/simulate",
                json={"config": config, "days": days}
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            st.error(f"API Error: {str(e)}")
            return {}
    
    async def create_vegetation_plan(self, config: Dict[str, Any], season: str = "summer") -> Dict[str, Any]:
        \"\"\"Call vegetation planning API\"\"\"
        try:
            response = self.client.post(
                "/api/v1/vegetation/plan",
                json={"config": config, "season": season}
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            st.error(f"API Error: {str(e)}")
            return {}
    
    async def analyze_economics(self, config: Dict[str, Any], planting_plan: Dict[str, Any]) -> Dict[str, Any]:
        \"\"\"Call economics analysis API\"\"\"
        try:
            response = self.client.post(
                "/api/v1/economics/analyze",
                json={"config": config, "planting_plan": planting_plan}
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            st.error(f"API Error: {str(e)}")
            return {}
    
    async def start_seed_batch(self, plant_type: str, quantity: int, manager_id: str = "default") -> Dict[str, Any]:
        \"\"\"Start a seed batch\"\"\"
        try:
            response = self.client.post(
                f"/api/v1/sapling/start-batch?manager_id={manager_id}",
                json={"plant_type": plant_type, "quantity": quantity}
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            st.error(f"API Error: {str(e)}")
            return {}
    
    async def get_production_status(self, manager_id: str = "default") -> Dict[str, Any]:
        \"\"\"Get production status\"\"\"
        try:
            response = self.client.get(f"/api/v1/sapling/status/{manager_id}")
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            st.error(f"API Error: {str(e)}")
            return {}
    
    async def analyze_marketing(self, config: Dict[str, Any], planting_plan: Dict[str, Any], budget: float) -> Dict[str, Any]:
        \"\"\"Call marketing analysis API\"\"\"
        try:
            response = self.client.post(
                "/api/v1/marketing/analyze",
                json={"config": config, "planting_plan": planting_plan, "budget": budget}
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            st.error(f"API Error: {str(e)}")
            return {}
    
    async def submit_feedback(self, feedback_data: Dict[str, Any]) -> Dict[str, Any]:
        \"\"\"Submit user feedback\"\"\"
        try:
            response = self.client.post(
                "/api/v1/feedback/submit",
                json=feedback_data
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            st.error(f"API Error: {str(e)}")
            return {}
    
    def close(self):
        \"\"\"Close the HTTP client\"\"\"
        self.client.close()

# Usage in Streamlit app:
# 
# @st.cache_resource
# def get_api_client():
#     return GreenhouseAPIClient()
# 
# client = get_api_client()
# result = asyncio.run(client.simulate_greenhouse(config_data))
"""

# Testing utilities
# test_api.py
"""
import pytest
import httpx
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_greenhouse_simulation():
    config = {
        "size_sqft": 2000,
        "operation_scale": "personal",
        "automation_level": 0.5,
        "climate_control": True,
        "irrigation_system": True,
        "location": "Denver, CO"
    }
    
    response = client.post(
        "/api/v1/greenhouse/simulate",
        json={"config": config, "days": 7}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "environmental_data" in data
    assert "analysis" in data
    assert len(data["environmental_data"]) > 0

def test_vegetation_planning():
    config = {
        "size_sqft": 2000,
        "operation_scale": "commercial",
        "automation_level": 0.7,
        "climate_control": True,
        "irrigation_system": True,
        "location": "Denver, CO"
    }
    
    response = client.post(
        "/api/v1/vegetation/plan",
        json={"config": config, "season": "summer"}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "plan" in data
    assert "recommendations" in data
    assert len(data["plan"]["plantings"]) > 0

def test_feedback_submission():
    feedback = {
        "operation_scale": "personal",
        "greenhouse_size": 1000,
        "feedback_type": "feature_request",
        "rating": 4,
        "title": "Add mobile app support",
        "description": "Would love to monitor my greenhouse from my phone"
    }
    
    response = client.post("/api/v1/feedback/submit", json=feedback)
    
    assert response.status_code == 200
    data = response.json()
    assert "feedback_id" in data
    assert data["status"] == "received"

if __name__ == "__main__":
    pytest.main([__file__])
"""

# Deployment script
# deploy.sh
"""
#!/bin/bash

echo "🚀 Deploying Greenhouse Vegetation Playground API..."

# Build and start services
docker-compose build
docker-compose up -d

# Wait for services to be ready
echo "⏳ Waiting for services to start..."
sleep 10

# Run health checks
echo "🔍 Running health checks..."
curl -f http://localhost:8000/health || exit 1
curl -f http://localhost:8501 || exit 1

# Run tests
echo "🧪 Running API tests..."
python -m pytest test_api.py -v

echo "✅ Deployment complete!"
echo "📊 Streamlit Dashboard: http://localhost:8501"
echo "📚 API Documentation: http://localhost:8000/docs"
echo "🔧 API Health Check: http://localhost:8000/health"
"""

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ):
        raise
    except Exception as e:
        logger.error(f"Error getting conditions: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get conditions: {str(e)}")

@router.get("/analyzers")
async def list_active_analyzers():
    """
    List all active greenhouse analyzers
    """
    try:
        analyzers_info = []
        for analyzer_id, analyzer in active_analyzers.items():
            analyzers_info.append({
                "analyzer_id": analyzer_id,
                "config": {
                    "size_sqft": analyzer.config.size_sqft,
                    "operation_scale": analyzer.config.operation_scale.value,
                    "location": analyzer.config.location,
                    "automation_level": analyzer.config.automation_level
                },
                "data_points": len(analyzer.environmental_history)
            })
        
        return {
            "active_analyzers": len(active_analyzers),
            "analyzers": analyzers_info
        }
        
    except Exception as e:
        logger.error(f"Error listing analyzers: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list analyzers: {str(e)}")

@router.delete("/analyzers/{analyzer_id}")
async def delete_analyzer(analyzer_id: str):
    """
    Delete an active analyzer
    """
    try:
        if analyzer_id not in active_analyzers:
            raise HTTPException(status_code=404, detail="Analyzer not found")
        
        del active_analyzers[analyzer_id]
        
        return {
            "message": f"Analyzer {analyzer_id} deleted successfully",
            "remaining_analyzers": len(active_analyzers)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting analyzer: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete analyzer: {str(e)}")

@router.get("/environmental-data/{analyzer_id}")
async def get_environmental_history(analyzer_id: str, limit: int = 100):
    """
    Get environmental data history for an analyzer
    """
    try:
        if analyzer_id not in active_analyzers:
            raise HTTPException(status_code=404, detail="Analyzer not found")
        
        analyzer = active_analyzers[analyzer_id]
        history = analyzer.environmental_history[-limit:]  # Get last N records
        
        environmental_data = []
        for data in history:
            environmental_data.append(EnvironmentalDataSchema(
                timestamp=data.timestamp,
                temperature=data.temperature,
                humidity=data.humidity,
                soil_moisture=data.soil_moisture,
                co2_level=data.co2_level,
                ph_level=data.ph_level,
                light_intensity=data.light_intensity,
                soil_nutrients=data.soil_nutrients
            ))
        
        return {
            "analyzer_id": analyzer_id,
            "data_points": len(environmental_data),
            "environmental_data": environmental_data
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting environmental history: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get history: {str(e)}")

# api/routes/vegetation.py
from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict, Any
import logging
from datetime import datetime

from src.vegetation_planner import VegetationPlanner
from routes.greenhouse import create_greenhouse_config, convert_operation_scale
from schemas.vegetation_schema import (
    PlanningRequestSchema, PlanningResponseSchema, VegetationPlanSchema,
    PlantingSchema, PlantProfileSchema
)

router = APIRouter()
logger = logging.getLogger(__name__)

# In-memory storage for planners
active_planners: Dict[str, VegetationPlanner] = {}

def convert_planting_plan(plan: Dict) -> VegetationPlanSchema:
    """Convert internal plan format to schema"""
    plantings = []
    for planting in plan.get('plantings', []):
        plantings.append(PlantingSchema(**planting))
    
    plan_data = {
        "strategy": plan.get('strategy', ''),
        "total_space": plan.get('total_space', 0),
        "plantings": plantings,
        "space_utilization": plan.get('space_utilization', '0%'),
    }
    
    # Add optional fields if they exist
    optional_fields = ['learning_objectives', 'experiment_suggestions', 'care_tips', 'projected_revenue']
    for field in optional_fields:
        if field in plan:
            plan_data[field] = plan[field]
    
    return VegetationPlanSchema(**plan_data)

@router.post("/plan", response_model=PlanningResponseSchema)
async def create_vegetation_plan(request: PlanningRequestSchema):
    """
    Create an optimized vegetation plan based on greenhouse configuration
    """
    try:
        # Create greenhouse configuration
        config = create_greenhouse_config(request.config)
        
        # Initialize planner
        planner = VegetationPlanner(config)
        planner_id = f"{config.operation_scale.value}_{datetime.now().timestamp()}"
        active_planners[planner_id] = planner
        
        # Generate planting plan
        plan = planner.generate_planting_plan(request.season.value)
        
        # Generate recommendations based on operation scale
        recommendations = []
        if config.operation_scale.value == "personal":
            recommendations.extend([
                "Focus on high-value crops like herbs and microgreens",
                "Consider succession planting for continuous harvest",
                "Start with easier plants and gradually expand variety"
            ])
        elif config.operation_scale.value == "commercial":
            recommendations.extend([
                "Optimize for volume production and consistent quality",
                "Establish reliable wholesale partnerships early",
                "Invest in automation to reduce labor costs"
            ])
        else:  # educational
            recommendations.extend([
                "Include plants with different growth characteristics for learning",
                "Plan seasonal experiments and demonstrations",
                "Engage students in harvest and marketing activities"
            ])
        
        response = PlanningResponseSchema(
            plan=convert_planting_plan(plan),
            recommendations=recommendations,
            plant_database_size=len(planner.plant_database),
            generated_at=datetime.now()
        )
        
        logger.info(f"Successfully created vegetation plan for {config.operation_scale.value} operation with {len(plan['plantings'])} plant types")
        return response
        
    except Exception as e:
        logger.error(f"Error creating vegetation plan: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to create plan: {str(e)}")

@router.get("/plants")
async def get_plant_database(operation_scale: str = "personal"):
    """
    Get available plants for a specific operation scale
    """
    try:
        # Create a temporary config to get plant database
        from schemas.greenhouse_schema import GreenhouseConfigSchema, OperationScaleEnum
        
        temp_config = GreenhouseConfigSchema(
            size_sqft=2000,
            operation_scale=operation_scale,
            automation_level=0.5,
            climate_control=True,
            irrigation_system=True,
            location="Denver, CO"
        )
        
        config = create_greenhouse_config(temp_config)
        planner = VegetationPlanner(config)
        
        plants = []
        for plant in planner.plant_database:
            plants.append(PlantProfileSchema(
                name=plant.name,
                space_required=plant.space_required,
                grow_time_days=plant.grow_time_days,
                optimal_temp_range=plant.optimal_temp_range,
                water_needs=plant.water_needs,
                market_value_per_unit=plant.market_value_per_unit,
                difficulty_level=plant.difficulty_level,
                seasonal_preference=plant.seasonal_preference
            ))
        
        return {
            "operation_scale": operation_scale,
            "total_plants": len(plants),
            "plants": plants
        }
        
    except Exception as e:
        logger.error(f"Error getting plant database: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get plants: {str(e)}")

@router.get("/planners")
async def list_active_planners():
    """
    List all active vegetation planners
    """
    try:
        planners_info = []
        for planner_id, planner in active_planners.items():
            planners_info.append({
                "planner_id": planner_id,
                "config": {
                    "size_sqft": planner.config.size_sqft,
                    "operation_scale": planner.config.operation_scale.value,
                    "location": planner.config.location
                },
                "available_plants": len(planner.plant_database)
            })
        
        return {
            "active_planners": len(active_planners),
            "planners": planners_info
        }
        
    except Exception as e:
        logger.error(f"Error listing planners: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list planners: {str(e)}")

@router.get("/optimize/{planner_id}")
async def optimize_existing_plan(planner_id: str, focus: str = "profit"):
    """
    Re-optimize an existing plan with different focus
    """
    try:
        if planner_id not in active_planners:
            raise HTTPException(status_code=404, detail="Planner not found")
        
        planner = active_planners[planner_id]
        
        # Generate different plans based on focus
        if focus == "diversity":
            season = "all"
        elif focus == "profit":
            season = "summer"  # Typically highest value season
        else:
            season = "current"
        
        plan = planner.generate_planting_plan(season)
        
        return {
            "planner_id": planner_id,
            "optimization_focus": focus,
            "plan": convert_planting_plan(plan),
            "optimized_at": datetime.now()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error optimizing plan: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to optimize: {str(e)}")

# api/routes/sapling.py
from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any
import logging
from datetime import datetime

from src.sapling_manager import SaplingManager, SeedStage
from routes.greenhouse import create_greenhouse_config
from schemas.sapling_schema import (
    BatchStartRequestSchema, BatchResponseSchema, ProgressSimulationRequestSchema,
    ProductionStatusSchema, SeedBatchSchema, ProductionScheduleSchema
)

router = APIRouter()
logger = logging.getLogger(__name__)

# In-memory storage for sapling managers
active_managers: Dict[str, SaplingManager] = {}

def convert_seed_stage(stage: SeedStage) -> str:
    """Convert SeedStage enum to string"""
    return stage.value

@router.post("/start-batch", response_model=BatchResponseSchema)
async def start_seed_batch(request: BatchStartRequestSchema, manager_id: str = "default"):
    """
    Start a new seed batch for production
    """
    try:
        if manager_id not in active_managers:
            # Create default manager if none exists
            from schemas.greenhouse_schema import GreenhouseConfigSchema
            default_config = GreenhouseConfigSchema(
                size_sqft=2000,
                operation_scale="commercial",
                automation_level=0.7,
                climate_control=True,
                irrigation_system=True,
                location="Denver, CO"
            )
            config = create_greenhouse_config(default_config)
            active_managers[manager_id] = SaplingManager(config)
        
        manager = active_managers[manager_id]
        
        # Start the batch
        batch_id = manager.start_seed_batch(
            plant_type=request.plant_type,
            quantity=request.quantity,
            target_date=request.target_date
        )
        
        response = BatchResponseSchema(
            batch_id=batch_id,
            message=f"Started batch for {request.quantity} {request.plant_type} plants",
            status="started",
            created_at=datetime.now()
        )
        
        logger.info(f"Started seed batch {batch_id} with {request.quantity} {request.plant_type} plants")
        return response
        
    except Exception as e:
        logger.error(f"Error starting seed batch: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to start batch: {str(e)}")

@router.post("/simulate-progress")
async def simulate_batch_progress(request: ProgressSimulationRequestSchema, manager_id: str = "default"):
    """
    Simulate growth progress for all active batches
    """
    try:
        if manager_id not in active_managers:
            raise HTTPException(status_code=404, detail="Manager not found")
        
        manager = active_managers[manager_id]
        
        # Simulate progress
        manager.simulate_batch_progress(days=request.days)
        
        # Get updated status
        status = manager.get_production_status()
        
        return {
            "manager_id": manager_id,
            "simulated_days": request.days,
            "production_status": status,
            "simulated_at": datetime.now()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error simulating progress: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Simulation failed: {str(e)}")

@router.get("/status/{manager_id}")
async def get_production_status(manager_id: str):
    """
    Get current production status for a manager
    """
    try:
        if manager_id not in active_managers:
            raise HTTPException(status_code=404, detail="Manager not found")
        
        manager = active_managers[manager_id]
        status = manager.get_production_status()
        
        return {
            "manager_id": manager_id,
            "status": status,
            "timestamp": datetime.now()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get status: {str(e)}")

@router.get("/batches/{manager_id}")
async def get_active_batches(manager_id: str):
    """
    Get all active batches for a manager
    """
    try:
        if manager_id not in active_managers:
            raise HTTPException(status_code=404, detail="Manager not found")
        
        manager = active_managers[manager_id]
        
        batches = []
        for batch_id, batch in manager.active_batches.items():
            batches.append(SeedBatchSchema(
                batch_id=batch.batch_id,
                plant_type=batch.plant_type,
                quantity=batch.quantity,
                start_date=batch.start_date,
                current_stage=convert_seed_stage(batch.current_stage),
                success_rate=batch.success_rate,
                days_in_stage=batch.days_in_stage,
                environmental_stress=batch.environmental_stress,
                notes=batch.notes
            ))
        
        return {
            "manager_id": manager_id,
            "total_batches": len(batches),
            "batches": batches
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting batches: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get batches: {str(e)}")

@router.get("/managers")
async def list_active_managers():
    """
    List all active sapling managers
    """
    try:
        managers_info = []
        for manager_id, manager in active_managers.items():
            managers_info.append({
                "manager_id": manager_id,
                "config": {
                    "size_sqft": manager.config.size_sqft,
                    "operation_scale": manager.config.operation_scale.value,
                    "automation_level": manager.config.automation_level
                },
                "active_batches": len(manager.active_batches),
                "scheduled_items": len(manager.production_schedule)
            })
        
        return {
            "active_managers": len(active_managers),
            "managers": managers_info
        }
        
    except Exception as e:
        logger.error(f"Error listing managers: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list managers: {str(e)}")

@router.post("/schedule/{manager_id}")
async def create_production_schedule(manager_id: str, planting_plan: Dict[str, Any], weeks_ahead: int = 8):
    """
    Create production schedule based on planting plan
    """
    try:
        if manager_id not in active_managers:
            raise HTTPException(status_code=404, detail="Manager not found")
        
        manager = active_managers[manager_id]
        schedule = manager.plan_production_schedule(planting_plan, weeks_ahead)
        
        schedule_items = []
        for item in schedule:
            schedule_items.append(ProductionScheduleSchema(
                target_date=item.target_date,
                plant_type=item.plant_type,
                quantity_needed=item.quantity_needed,
                batch_ids=item.batch_ids,
                status=item.status
            ))
        
        return {
            "manager_id": manager_id,
            "schedule_items": len(schedule_items),
            "schedule": schedule_items,
            "weeks_ahead": weeks_ahead,
            "created_at": datetime.now()
        }
        
    except HTTPException
