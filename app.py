import streamlit as st
import requests
import json
import time
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import base64
from bs4 import BeautifulSoup
import re

# Rest of your imports...
# Import des modèles multilingues avancés
from models import multilingual_models
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="Test App", layout="wide")

st.title("🧪 Test Application")
st.success("✅ App is running!")

# Test plotly
st.subheader("Plotly Test")
fig = px.bar(x=[1, 2, 3], y=[1, 2, 3], title="Test Chart")
st.plotly_chart(fig)

# Test models
try:
    from models import multilingual_models
    models = multilingual_models
    status = models.get_model_status()
    st.success(f"✅ Models loaded: {status['models_loaded']}")
    st.json(status)
except Exception as e:
    st.error(f"❌ Models error: {e}")

st.info("If you see this, the basic app works!")
