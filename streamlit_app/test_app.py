# test_app.py - Put this in streamlit_app/ folder for testing
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime

st.title("🧪 AgriTech System Test")

# Test all imports
try:
    st.success("✅ All imports successful!")
    st.write(f"Streamlit version: {st.__version__}")
    st.write(f"Pandas version: {pd.__version__}")
    st.write(f"NumPy version: {np.__version__}")
    
    # Test basic functionality
    sample_data = pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=10),
        'temperature': np.random.normal(25, 3, 10),
        'humidity': np.random.normal(65, 10, 10)
    })
    
    fig = px.line(sample_data, x='date', y=['temperature', 'humidity'], 
                  title='Sample Data Test')
    st.plotly_chart(fig)
    
    st.success("🚀 Ready to deploy full AgriTech system!")
    
except Exception as e:
    st.error(f"❌ Error: {e}")
    st.write("Check your requirements.txt and file structure")
