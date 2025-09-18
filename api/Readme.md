🎯 API Endpoints Overview
Greenhouse Management (/api/v1/greenhouse)

POST /simulate - Generate environmental conditions
GET /conditions/{analyzer_id} - Get current conditions
GET /analyzers - List active analyzers
GET /environmental-data/{analyzer_id} - Historical data

Vegetation Planning (/api/v1/vegetation)

POST /plan - Create optimized planting plans
GET /plants - Browse plant database by scale
GET /optimize/{planner_id} - Re-optimize existing plans

Production Management (/api/v1/sapling)

POST /start-batch - Start seed batches
POST /simulate-progress - Simulate growth
GET /status/{manager_id} - Production status
GET /batches/{manager_id} - Active batches
POST /schedule/{manager_id} - Create schedules

Economic Analysis (/api/v1/economics)

POST /analyze - Complete financial analysis
GET /scenarios/{calculator_id} - Scenario modeling
GET /optimization/{calculator_id} - Cost optimizations

Marketing Intelligence (/api/v1/marketing)

POST /analyze - Marketing strategy analysis
GET /segments/{optimizer_id} - Market segments
GET /channels - Distribution channels
POST /campaign/{optimizer_id} - Create campaigns

Feedback & Analytics (/api/v1/feedback)

POST /submit - Submit user feedback
POST /analytics - Log usage data
GET /feedback - Feedback summaries
GET /health-metrics - System health

🔧 How to Run the API
Option 1: Quick Start
bash# Install dependencies
pip install fastapi uvicorn pydantic httpx

# Copy your greenhouse modules to src/
mkdir src
# Copy greenhouse_analyzer.py, vegetation_planner.py, etc.

# Run the API
uvicorn api.main:app --reload --port 8000
Option 2: Docker Deployment
bash# Using the provided docker-compose.yml
docker-compose up -d

# Services will start:
# - FastAPI: http://localhost:8000
# - PostgreSQL: localhost:5432
# - Redis: localhost:6379
# - Streamlit: http://localhost:8501
Option 3: Production Deployment
bash# Use the deployment script
chmod +x deploy.sh
./deploy.sh
📊 API Integration with Streamlit
Using the API Client
python# In your Streamlit app
from api_client import GreenhouseAPIClient

client = GreenhouseAPIClient("http://localhost:8000")

# Replace direct function calls with API calls
config_data = {...}
result = await client.simulate_greenhouse(config_data, days=14)
Benefits of API Architecture

🔄 Scalability - Independent scaling of frontend/backend
🛡️ Security - Centralized authentication and validation
📈 Analytics - Built-in usage tracking and feedback
🔧 Flexibility - Multiple frontends (web, mobile, desktop)
⚡ Performance - Caching and optimized data processing
🐛 Debugging - Centralized logging and error handling

🎮 Testing the API
Interactive Documentation

Swagger UI: http://localhost:8000/docs
ReDoc: http://localhost:8000/redoc

Health Check
bashcurl http://localhost:8000/health
Example API Call
pythonimport httpx

config = {
    "size_sqft": 2000,
    "operation_scale": "commercial",
    "automation_level": 0.7,
    "climate_control": True,
    "irrigation_system": True,
    "location": "Denver, CO"
}

response = httpx.post(
    "http://localhost:8000/api/v1/greenhouse/simulate",
    json={"config": config, "days": 7}
)

data = response.json()
print(f"Generated {len(data['environmental_data'])} readings")
🚀 Advanced Features
Real-time Monitoring

WebSocket support for live updates
Background task processing
Event-driven architecture

Data Persistence

PostgreSQL for relational data
Redis for caching and sessions
File storage for large datasets

Security & Authentication

JWT token authentication
Role-based access control
API rate limiting
Input validation and sanitization

Analytics & Insights

Usage tracking and metrics
Performance monitoring
User feedback analysis
A/B testing support

📈 Production Considerations
Performance

Response caching with Redis
Database query optimization
Async processing for heavy operations
Load balancing for high traffic

Monitoring

Health check endpoints
Metrics collection (Prometheus/Grafana)
Error tracking and alerting
Performance profiling

Deployment

Docker containerization
Kubernetes orchestration
CI/CD pipeline integration
Environment-specific configurations

This FastAPI backend transforms your greenhouse system into a production-ready, scalable API that can power multiple applications while providing robust analytics, user feedback, and system health monitoring. The architecture supports your current Streamlit frontend while enabling future expansion to mobile apps, IoT integrations, and enterprise solutions!
