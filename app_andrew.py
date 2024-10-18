import streamlit as st
import os
import cv2
from PIL import Image
import numpy as np
import base64
from dotenv import load_dotenv
import google.generativeai as genai
from groq import Groq
import re
import requests
import time

# Load environment variables
load_dotenv()

# Configure Google Generative AI
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize Groq client
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Spoonacular API configuration
SPOONACULAR_API_KEY = os.getenv("SPOONACULAR_API_KEY")

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def analyze_fridge_image(image_path):
    try:
        image = Image.open(image_path)
        model = genai.GenerativeModel(model_name="gemini-1.5-pro")

        prompt_items = """
        Analyze this image of a refrigerator's contents and list all the food items you can identify.
        For each item, provide the following information in this exact format:
        Item: [Item name]
        Quantity: [Estimated quantity or 'Not visible' if you can't determine]
        Location: [Bounding box coordinates as [ymin, xmin, ymax, xmax] where each coordinate is an integer between 0 and 1000]
        """

        for attempt in range(3):
            try:
                response_items = model.generate_content([prompt_items, image])
                break
            except Exception as e:
                if '429' in str(e):
                    if attempt < 2:
                        st.warning("Quota exceeded. Retrying in 5 seconds...")
                        time.sleep(5)
                    else:
                        st.error("Quota exceeded. Please try again later.")
                        return None, None, None
                else:
                    st.error(f"An error occurred: {str(e)}")
                    return None, None, None
        
        items_info = parse_item_info(response_items.text.strip())
        annotated_image = generate_annotated_image(image_path, items_info)

        analysis_result = "\n".join([f"{item}: {info['quantity']}" for item, info in items_info.items()])
        analysis_result += f"\n\nTotal number of distinct food items: {len(items_info)}"
        
        return analysis_result, annotated_image, items_info

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return None, None, None
    finally:
        image.close()

def parse_item_info(response_text):
    items_info = {}
    current_item = None
    for line in response_text.split('\n'):
        if line.startswith("Item:"):
            current_item = line.split(":", 1)[1].strip()
            items_info[current_item] = {}
        elif line.startswith("Quantity:") and current_item:
            items_info[current_item]['quantity'] = line.split(":", 1)[1].strip()
        elif line.startswith("Location:") and current_item:
            location = line.split(":", 1)[1].strip()
            box = parse_bounding_box(location)
            items_info[current_item]['box'] = box
            items_info[current_item]['location'] = location if box == [0, 0, 0, 0] else f"Coordinates: {box}"
    return items_info

def parse_bounding_box(text):
    match = re.search(r'\[?\s*(\d+)\s*,?\s*(\d+)\s*,?\s*(\d+)\s*,?\s*(\d+)\s*\]?', text)
    if match:
        return [int(match.group(i)) for i in range(1, 5)]
    return [0, 0, 0, 0]

def convert_coordinates(box, original_width, original_height):
    ymin, xmin, ymax, xmax = box
    return [
        int(ymin / 1000 * original_height),
        int(xmin / 1000 * original_width),
        int(ymax / 1000 * original_height),
        int(xmax / 1000 * original_width)
    ]

def generate_annotated_image(image_path, items_info):
    image = cv2.imread(image_path)
    height, width = image.shape[:2]

    for item, info in items_info.items():
        box = info['box']
        quantity = info['quantity']

        if box != [0, 0, 0, 0]:
            ymin, xmin, ymax, xmax = convert_coordinates(box, width, height)
            xmin = max(0, min(xmin, width - 1))
            ymin = max(0, min(ymin, height - 1))
            xmax = max(0, min(xmax, width - 1))
            ymax = max(0, min(ymax, height - 1))

            if xmin < xmax and ymin < ymax:
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                label = f"{item}: {quantity}"
                cv2.putText(image, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

def get_recipes_from_spoonacular(ingredients, number=4):
    url = "https://api.spoonacular.com/recipes/findByIngredients"
    params = {
        'apiKey': SPOONACULAR_API_KEY,
        'ingredients': ','.join(ingredients),
        'number': number,
        'ranking': 2,
        'ignorePantry': True
    }
    
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json()
    return []

def get_recipe_details(recipe_id):
    url = f"https://api.spoonacular.com/recipes/{recipe_id}/information"
    params = {
        'apiKey': SPOONACULAR_API_KEY,
        'includeNutrition': True
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json()
    return None

def generate_recipe_details_groq(recipe_name, ingredients, missed_ingredients, recipe_details):
    all_ingredients = ingredients + missed_ingredients
    prompt = f"""
    Generate a detailed recipe for "{recipe_name}" using these ingredients: {', '.join(all_ingredients)}.
    
    Use the following information in your response:
    Preparation Time: {recipe_details.get('readyInMinutes', 'N/A')} minutes
    Health Score: {recipe_details.get('healthScore', 'N/A')}
    Cuisine: {', '.join(recipe_details.get('cuisines', ['N/A']))}
    Diets: {', '.join(recipe_details.get('diets', ['N/A']))}
    
    Provide the following information in this exact format:
    
    Ingredients:
    - [Ingredient 1]: [Quantity]
    - [Ingredient 2]: [Quantity]
    ...
    
    Instructions:
    1. [Step 1]
    2. [Step 2]
    ...
    
    Equipment Needed:
    - [Equipment 1]
    - [Equipment 2]
    ...
    
    Nutrition Information (per serving):
    - Calories: [Calories]
    - Protein: [Protein]g
    - Fat: [Fat]g
    - Carbohydrates: [Carbs]g
    
    Cooking Time: [Time in minutes]
    Servings: [Number of servings]
    """
    
    response = groq_client.chat.completions.create(
        model="llama-3.1-70b-versatile",
        messages=[
            {"role": "system", "content": "You are a helpful culinary assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=1000
    )

    return response.choices[0].message.content

def create_recipe_card(recipe, recipe_details):
    missed_ingredients = ', '.join([ing['name'] for ing in recipe['missedIngredients']])
    card_html = f"""
    <div style="border: 1px solid #ddd; border-radius: 8px; padding: 16px; margin: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
        <img src="{recipe['image']}" style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px;">
        <h3 style="margin-top: 10px;">{recipe['title']}</h3>
        <p><strong>Missing ingredients:</strong> {missed_ingredients}</p>
        <div style="height: 300px; overflow-y: auto;">
            {recipe_details}
        </div>
    </div>
    """
    return card_html

def main():
    st.set_page_config(layout="wide")
    st.title("Fridge Food Item Detector and Recipe Generator")

    uploaded_image = st.file_uploader("Upload an image of your fridge", type=["jpg", "jpeg", "png"])

    if uploaded_image:
        col1, col2 = st.columns(2)
        with col1:
            st.image(uploaded_image, caption='Uploaded Image', use_column_width=True)
        
        temp_file_path = f"temp_{uploaded_image.name}"
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_image.getvalue())

        analysis_result, annotated_image, items_info = analyze_fridge_image(temp_file_path)

        if analysis_result and annotated_image and items_info:
            with col2:
                st.image(annotated_image, caption="Annotated Image", use_column_width=True)
            
            st.success("Fridge analysis completed!")
            st.text(analysis_result)

            ingredients = list(items_info.keys())
            if ingredients:
                st.subheader("Generated Recipes")
                recipes = get_recipes_from_spoonacular(ingredients)
                
                if recipes:
                    cols = st.columns(2)
                    for i, recipe in enumerate(recipes):
                        with cols[i % 2]:
                            recipe_details = get_recipe_details(recipe['id'])
                            missed_ingredients = [ing['name'] for ing in recipe['missedIngredients']]
                            detailed_recipe = generate_recipe_details_groq(recipe['title'], ingredients, missed_ingredients, recipe_details)
                            st.markdown(create_recipe_card(recipe, detailed_recipe), unsafe_allow_html=True)
                else:
                    st.warning("No recipes found for the given ingredients.")
            else:
                st.error("No ingredients were detected.")

        try:
            os.remove(temp_file_path)
        except PermissionError as e:
            st.warning(f"Could not delete temporary file: {e}")

if __name__ == "__main__":
    main()