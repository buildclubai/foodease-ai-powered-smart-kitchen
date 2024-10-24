import streamlit as st
from recipe_tracker import create_health_visualizations

def main():
    st.set_page_config(layout="wide", page_title="FoodEase - Health Analytics")
    create_health_visualizations()

if __name__ == "__main__":
    main()