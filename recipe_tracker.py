import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os
import json

def save_recipe(recipe, recipe_details, meal_type):
    """Save recipe with selected meal type"""
    try:
        tracker = RecipeTracker()
        
        # Extract health data
        health_data = extract_health_data(recipe_details)
        
        # Prepare recipe data
        recipe_data = {
            'recipe_name': recipe['title'],
            'meal_type': meal_type,
            **health_data
        }
        
        # Save recipe
        if tracker.save_recipe(recipe_data):
            st.success(f"✅ Recipe '{recipe['title']}' saved successfully!")
            
            # Show saved data
            st.write("Saved recipe details:")
            st.json({
                'Recipe': recipe['title'],
                'Meal Type': meal_type,
                'Calories': health_data['calories'],
                'Protein': health_data['protein'],
                'Carbs': health_data['carbs'],
                'Fat': health_data['fat'],
                'Cuisine': health_data['cuisine'],
                'Dietary Type': health_data['dietary_type']
            })
        else:
            st.error("Failed to save recipe. Please try again.")
            
    except Exception as e:
        st.error(f"Error saving recipe: {str(e)}")
        st.error("Please try again or contact support if the issue persists.")

class RecipeTracker:
    def __init__(self):
        self.data_file = "recipe_history.csv"
        self.columns = [
            'date', 'recipe_name', 'calories', 'protein', 'carbs', 'fat',
            'cooking_time', 'cuisine', 'dietary_type', 'meal_type'
        ]
        self._initialize_data()

    def _initialize_data(self):
        """Initialize the CSV file if it doesn't exist"""
        try:
            if not os.path.exists(self.data_file):
                df = pd.DataFrame(columns=self.columns)
                df.to_csv(self.data_file, index=False)
            else:
                # Verify file is readable and has correct columns
                df = pd.read_csv(self.data_file)
                missing_columns = set(self.columns) - set(df.columns)
                if missing_columns:
                    for col in missing_columns:
                        df[col] = ''
                    df.to_csv(self.data_file, index=False)
        except Exception as e:
            st.error(f"Error initializing data file: {str(e)}")

    def save_recipe(self, recipe_data):
        """Save recipe data to CSV file"""
        try:
            df = pd.read_csv(self.data_file)
            recipe_data['date'] = datetime.now().strftime('%Y-%m-%d')
            
            # Ensure all required columns are present
            for col in self.columns:
                if col not in recipe_data:
                    recipe_data[col] = ''
            
            new_row = pd.DataFrame([recipe_data])
            df = pd.concat([df, new_row], ignore_index=True)
            df.to_csv(self.data_file, index=False)
            return True
        except Exception as e:
            st.error(f"Error saving recipe data: {str(e)}")
            return False

    def get_stats(self):
        if os.path.exists(self.data_file):
            df = pd.read_csv(self.data_file)
            df['date'] = pd.to_datetime(df['date'])
            return df
        return pd.DataFrame(columns=self.columns)

def extract_health_data(recipe_details):
    """Extract health-related data from recipe details"""
    health_data = {}
    
    # Extract calories
    calories_str = recipe_details.get('Calories', '0')
    calories = int(''.join(filter(str.isdigit, calories_str))) if any(c.isdigit() for c in calories_str) else 0
    health_data['calories'] = calories
    
    # Parse cooking time
    cooking_time = recipe_details.get('Cooking Time', 'N/A')
    health_data['cooking_time'] = cooking_time
    
    # Extract cuisine and dietary information
    health_data['cuisine'] = recipe_details.get('Cuisine', 'N/A')
    health_data['dietary_type'] = recipe_details.get('Dietary', 'N/A')
    
    # Estimate macronutrients based on calories
    total_calories = float(calories)
    health_data['protein'] = round((total_calories * 0.25) / 4)  # 25% of calories from protein
    health_data['carbs'] = round((total_calories * 0.45) / 4)    # 45% of calories from carbs
    health_data['fat'] = round((total_calories * 0.30) / 9)      # 30% of calories from fat
    
    return health_data

def add_save_recipe_button(recipe, recipe_details):
    """Add a save recipe button to recipe cards"""
    button_key = f"save_recipe_{recipe['title']}"
    
    # Create a meal type selector
    meal_type = st.selectbox(
        "Select meal type",
        options=['Breakfast', 'Lunch', 'Dinner', 'Snack'],
        key=f"meal_type_{recipe['title']}"
    )
    
    if st.button(f"💾 Save Recipe", key=button_key):
        tracker = RecipeTracker()
        
        # Extract health data
        health_data = extract_health_data(recipe_details)
        
        # Prepare recipe data for saving
        recipe_data = {
            'recipe_name': recipe['title'],
            'meal_type': meal_type,
            **health_data
        }
        
        # Save recipe
        if tracker.save_recipe(recipe_data):
            st.success(f"✅ Recipe '{recipe['title']}' saved successfully!")
        else:
            st.error("Failed to save recipe. Please try again.")

def create_health_visualizations():
    st.title("📊 Health Analytics Dashboard")
    
    tracker = RecipeTracker()
    df = tracker.get_stats()
    
    if df.empty:
        st.warning("No recipe data available yet. Start cooking and saving recipes!")
        return
    
    # Date range filter
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", min(df['date']))
    with col2:
        end_date = st.date_input("End Date", max(df['date']))
    
    mask = (df['date'] >= pd.Timestamp(start_date)) & (df['date'] <= pd.Timestamp(end_date))
    filtered_df = df.loc[mask]
    
    # Summary metrics
    st.subheader("📈 Summary Metrics")
    metrics_cols = st.columns(4)
    with metrics_cols[0]:
        st.metric("Total Recipes", len(filtered_df))
    with metrics_cols[1]:
        st.metric("Avg. Calories/Day", int(filtered_df.groupby('date')['calories'].sum().mean()))
    with metrics_cols[2]:
        st.metric("Most Common Cuisine", filtered_df['cuisine'].mode().iloc[0] if not filtered_df.empty else "N/A")
    with metrics_cols[3]:
        avg_cooking_time = filtered_df['cooking_time'].mode().iloc[0] if not filtered_df.empty else "N/A"
        st.metric("Avg. Cooking Time", avg_cooking_time)
    
    # Nutrition Overview
    st.subheader("🥗 Nutrition Overview")
    col1, col2 = st.columns(2)
    
    # Daily Macronutrient Distribution
    with col1:
        daily_macros = filtered_df.groupby('date')[['protein', 'carbs', 'fat']].mean()
        fig_macros = px.bar(daily_macros, 
                           title='Daily Macronutrient Distribution',
                           barmode='group')
        st.plotly_chart(fig_macros, use_container_width=True)
    
    # Meal Type Distribution
    with col2:
        meal_dist = filtered_df['meal_type'].value_counts()
        fig_meals = px.pie(values=meal_dist.values, 
                          names=meal_dist.index,
                          title='Meal Type Distribution')
        st.plotly_chart(fig_meals, use_container_width=True)
    
    # Calorie Trend
    st.subheader("🔥 Calorie Intake Trend")
    daily_calories = filtered_df.groupby(['date', 'meal_type'])['calories'].sum().reset_index()
    fig_calories = px.line(daily_calories, 
                          x='date', 
                          y='calories',
                          color='meal_type',
                          title='Daily Calorie Intake by Meal Type')
    st.plotly_chart(fig_calories, use_container_width=True)
    
    # Cuisine and Dietary Insights
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🌎 Cuisine Distribution")
        cuisine_counts = filtered_df['cuisine'].value_counts()
        fig_cuisine = px.pie(values=cuisine_counts.values,
                           names=cuisine_counts.index,
                           title='Cuisine Distribution')
        st.plotly_chart(fig_cuisine, use_container_width=True)
    
    with col2:
        st.subheader("🥗 Dietary Types")
        dietary_counts = filtered_df['dietary_type'].value_counts()
        fig_dietary = px.bar(x=dietary_counts.index,
                           y=dietary_counts.values,
                           title='Dietary Types Distribution')
        st.plotly_chart(fig_dietary, use_container_width=True)
    
    # Recipe History Table
    st.subheader("📝 Recipe History")
    st.dataframe(
        filtered_df[['date', 'recipe_name', 'meal_type', 'calories', 'cuisine', 'dietary_type']]
        .sort_values('date', ascending=False)
    )