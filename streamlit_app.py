# requirements.txt
streamlit>=1.28.0
pandas>=1.5.0
numpy>=1.24.0
matplotlib>=3.6.0
seaborn>=0.12.0
plotly>=5.15.0
dataclasses-json>=0.5.0

# setup.py (Optional - for package installation)
from setuptools import setup, find_packages

setup(
    name="greenhouse-vegetation-playground",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "streamlit>=1.28.0",
        "pandas>=1.5.0",
        "numpy>=1.24.0",
        "matplotlib>=3.6.0",
        "seaborn>=0.12.0",
        "plotly>=5.15.0",
    ],
    author="Greenhouse Developer",
    description="AI-powered greenhouse optimization platform",
    python_requires=">=3.8",
)

# run_app.sh (Linux/Mac startup script)
#!/bin/bash

# Create src directory if it doesn't exist
mkdir -p src

# Check if all required files exist
if [ ! -f "src/__init__.py" ]; then
    touch src/__init__.py
fi

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Check if greenhouse modules exist
if [ ! -f "src/greenhouse_analyzer.py" ]; then
    echo "⚠️  Warning: Greenhouse modules not found in src/ directory"
    echo "Please copy the greenhouse system code to the src/ folder"
    echo ""
fi

# Run the Streamlit app
echo "Starting Greenhouse Vegetation Playground..."
streamlit run streamlit_app.py --server.port 8501 --server.address localhost

# run_app.bat (Windows startup script)
@echo off

REM Create src directory if it doesn't exist
if not exist "src" mkdir src

REM Check if all required files exist
if not exist "src\__init__.py" (
    echo. > src\__init__.py
)

REM Install requirements
echo Installing requirements...
pip install -r requirements.txt

REM Check if greenhouse modules exist
if not exist "src\greenhouse_analyzer.py" (
    echo Warning: Greenhouse modules not found in src/ directory
    echo Please copy the greenhouse system code to the src/ folder
    echo.
)

REM Run the Streamlit app
echo Starting Greenhouse Vegetation Playground...
streamlit run streamlit_app.py --server.port 8501 --server.address localhost

# Quick Start Guide (README.md)
# Greenhouse Vegetation Playground - Streamlit Dashboard

🌱 **AI-Powered Greenhouse Optimization Platform**

A comprehensive system for optimizing greenhouse operations across personal, commercial, and educational scales.

## 🚀 Quick Start

### Option 1: Automatic Setup (Recommended)

**Windows:**
```bash
# Download and run
curl -o run_app.bat https://your-repo/run_app.bat
run_app.bat
```

**Linux/Mac:**
```bash
# Download and run
curl -o run_app.sh https://your-repo/run_app.sh
chmod +x run_app.sh
./run_app.sh
```

### Option 2: Manual Setup

1. **Install Dependencies:**
```bash
pip install streamlit pandas numpy matplotlib seaborn plotly
```

2. **Create Project Structure:**
```
greenhouse-playground/
├── streamlit_app.py          # Main Streamlit app
├── requirements.txt          # Dependencies
├── src/                      # Greenhouse modules
│   ├── __init__.py
│   ├── greenhouse_analyzer.py
│   ├── vegetation_planner.py
│   ├── economics_calculator.py
│   ├── sapling_manager.py
│   ├── marketing_optimizer.py
│   └── stats_visualizer.py
└── README.md
```

3. **Copy Greenhouse Modules:**
   - Copy the greenhouse system code into the `src/` directory
   - Each module should be a separate .py file

4. **Run the App:**
```bash
streamlit run streamlit_app.py
```

## 📱 Using the Dashboard

### 1. Configuration
- **Left Sidebar**: Set your greenhouse parameters
  - Operation Scale (Personal/Commercial/Educational)
  - Size, automation level, location
  - Analysis parameters

### 2. Analysis Tabs
- **🌡️ Environmental**: Real-time monitoring simulation
- **🌱 Planning**: Optimized vegetation plans
- **💰 Economics**: Financial analysis and ROI
- **🔬 Production**: Seed-to-harvest tracking
- **📈 Marketing**: Market optimization strategies
- **📊 Dashboard**: Executive summary

### 3. Interactive Features
- Real-time parameter adjustment
- Interactive charts and visualizations
- Downloadable reports and recommendations
- Scale-specific optimization

## 🎯 Key Features

### Environmental Monitoring
- Temperature, humidity, CO₂, pH tracking
- Smart alerts and recommendations
- Historical trend analysis
- Scale-appropriate variations

### Vegetation Planning
- **Personal**: Diverse home gardens, farmers market focus
- **Commercial**: Profit-optimized, volume production
- **Educational**: Learning-focused, experimental variety

### Economic Analysis
- Complete ROI and break-even calculations
- Cost optimization recommendations
- Revenue projections
- Risk assessments

### Production Management
- Seed-to-sapling workflow tracking
- Growth stage monitoring
- Success rate analysis
- Production scheduling

### Marketing Intelligence
- Market segment analysis
- Pricing optimization
- Distribution channel selection
- Revenue enhancement strategies

## 📊 Sample Results

### Personal Garden (2000 sq ft)
- **Setup Cost**: $3,000-5,000
- **Break-even**: 12-18 months
- **Focus**: High-value herbs, specialty vegetables
- **Revenue**: $2,000-4,000/year

### Commercial Operation (2000 sq ft)
- **Setup Cost**: $20,000-50,000
- **Break-even**: 8-12 months
- **Focus**: Volume production, wholesale
- **Revenue**: $15,000-30,000/year

### Educational Program (2000 sq ft)
- **Setup Cost**: $8,000-15,000
- **Break-even**: 18-24 months
- **Focus**: Learning, community engagement
- **Revenue**: $3,000-8,000/year + educational value

## 🔧 Customization

### Adding New Plants
Edit `vegetation_planner.py` to add new plant profiles:

```python
PlantProfile(
    name="Your Plant",
    space_required=2.0,  # sq ft per plant
    grow_time_days=60,
    optimal_temp_range=(18, 24),  # Celsius
    water_needs="medium",
    market_value_per_unit=4.00,
    difficulty_level="intermediate",
    seasonal_preference="summer"
)
```

### Adjusting Economic Parameters
Modify `economics_calculator.py` cost structures for your region.

### Custom Market Segments
Update `marketing_optimizer.py` with local market data.

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**
   - Ensure all modules are in `src/` directory
   - Check file names match exactly

2. **Slow Performance**
   - Reduce simulation days in sidebar
   - Use smaller greenhouse sizes for testing

3. **Visualization Issues**
   - Update plotly: `pip install --upgrade plotly`
   - Clear browser cache

4. **Memory Issues**
   - Restart Streamlit app
   - Reduce batch sizes in production simulation

## 🔗 API Integration Ideas

### Weather APIs
- OpenWeatherMap for real environmental data
- NOAA climate data integration

### Market Data
- USDA pricing information
- Local farmers market data

### IoT Integration
- Raspberry Pi sensor integration
- Arduino environmental monitoring

## 📈 Future Enhancements

- [ ] Machine learning yield prediction
- [ ] Automated irrigation control
- [ ] Mobile app companion
- [ ] Multi-location management
- [ ] Advanced financial modeling
- [ ] Integration with accounting systems

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## 📞 Support

- **Documentation**: Check the inline help in the Streamlit app
- **Issues**: Report bugs via GitHub issues
- **Discussions**: Join community discussions

---

**Happy Growing! 🌱**
