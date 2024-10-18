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
import io

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

def display_ingredient_grid(items_info):
    st.subheader("🥕 Detected Ingredients")
    
    # Custom CSS for the ingredient grid
    st.markdown("""
    <style>
    .ingredient-card {
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 10px;
        margin: 5px;
        text-align: center;
        height: 100px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
    }
    .ingredient-name {
        font-weight: bold;
        margin-bottom: 5px;
    }
    .ingredient-quantity {
        font-size: 0.9em;
        color: #666;
    }
    </style>
    """, unsafe_allow_html=True)

    # Create a 4-column grid
    cols = st.columns(4)
    for i, (item, info) in enumerate(items_info.items()):
        with cols[i % 4]:
            st.markdown(f"""
                <div class="ingredient-card">
                    <div class="ingredient-name">{item}</div>
                    <div class="ingredient-quantity">{info['quantity']}</div>
                </div>
            """, unsafe_allow_html=True)

def get_recipes_from_spoonacular(ingredients, number=4):
    url = "https://api.spoonacular.com/recipes/findByIngredients"
    params = {
        'apiKey': SPOONACULAR_API_KEY,
        'ingredients': ','.join(ingredients),
        'number': number,
        'ranking': 2,
        'ignorePantry': True
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        
        recipes = response.json()
        
        if not recipes:
            st.warning(f"No recipes found for the given ingredients.")
            st.info("Try uploading a different image with more ingredients.")
        
        return recipes
    except requests.exceptions.RequestException as e:
        st.error(f"An error occurred while fetching recipes: {str(e)}")
        st.info("Response content:")
        st.code(response.text)
        return []

def generate_recipe_details_groq(recipe):
    prompt = f"""
    Generate a detailed recipe for "{recipe['title']}" based on the following information:
    
    Ingredients:
    {' '.join([f"- {ingredient['original']}" for ingredient in recipe.get('usedIngredients', []) + recipe.get('missedIngredients', [])])}
    
    Provide the following information in this exact format:
    
    Key Information:
    Calories: [Estimated calories per serving]
    Cooking Time: [Estimated total time in minutes]
    Price: [Estimated price per serving in USD]
    Dietary: [List any dietary categories this recipe fits, e.g., Vegetarian, Vegan, Gluten-Free, etc.]
    Cuisine: [Type of cuisine, e.g., Italian, Mexican, etc.]
    Difficulty: [Easy/Medium/Hard]

    Description:
    [Provide a brief, enticing description of the dish in 2-3 sentences]

    Instructions:
    1. [Step 1]
    2. [Step 2]
    ...
    
    Cooking Techniques:
    [List 2-3 main cooking techniques used in this recipe]

    Flavor Profile:
    [Describe the main flavors of the dish]

    Texture:
    [Describe the texture of the finished dish]

    Nutritional Highlights:
    [Mention 2-3 key nutritional benefits of the dish]

    Serving Suggestions:
    [Provide 1-2 suggestions for serving or pairing the dish]

    Tips:
    [Provide 1-2 cooking tips or variations for this recipe]
    """
    
    response = groq_client.chat.completions.create(
        model="mixtral-8x7b-32768",
        messages=[
            {"role": "system", "content": "You are a helpful culinary assistant with expertise in various cuisines and cooking techniques."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=1000
    )

    return response.choices[0].message.content

def parse_recipe_key_info(recipe_details):
    key_info = {}
    key_info_section = recipe_details.split("Key Information:")[1].split("\n\n")[0]
    for line in key_info_section.split("\n"):
        if ":" in line:
            key, value = line.split(":", 1)
            key_info[key.strip()] = value.strip()
    return key_info

def create_recipe_card(recipe, recipe_details):
    missed_ingredients = ', '.join([ing['name'] for ing in recipe.get('missedIngredients', [])])
    used_ingredients = ', '.join([ing['name'] for ing in recipe.get('usedIngredients', [])])
    key_info = parse_recipe_key_info(recipe_details)

    if 'Calories' not in key_info:
        key_info['Calories'] = "Not Found"
    if 'Cooking Time' not in key_info:
        key_info['Cooking Time'] = "Not Found"
    if 'Price' not in key_info:
        key_info['Price'] = "Not Found"
    if 'Dietary' not in key_info:
        key_info['Dietary'] = "Not Found"
    if 'Cuisine' not in key_info:
        key_info['Cuisine'] = "Not Found"
    if 'Difficulty' not in key_info:
        key_info['Difficulty'] = "Not Found"
    
    return f"""
    <div style="border: 1px solid #ddd; border-radius: 8px; padding: 16px; margin: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); height: 600px; overflow-y: auto;">
        <img src="{recipe['image']}" style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px;">
        <h3 style="margin-top: 10px;">{recipe['title']}</h3>
        <p>🔥 **Calories**: {key_info['Calories']}</p>
        <p>⏱️ **Cooking Time**: {key_info['Cooking Time']}</p>
        <p>💰 **Price**: {key_info['Price']}</p>
        <p>🥗 **Dietary**: {key_info['Dietary']}</p>
        <p>🌎 **Cuisine**: {key_info['Cuisine']}</p>
        <p>📊 **Difficulty**: {key_info['Difficulty']}</p>
        <p><strong>Used ingredients:</strong> {used_ingredients}</p>
        <p><strong>Missing ingredients:</strong> {missed_ingredients}</p>
        <div>
            {recipe_details}
        </div>
    </div>
    """

def main():
    st.set_page_config(layout="wide", page_title="FoodEase")
    st.title("🍽️ FoodEase: AI Family Hub")

    # Initialize session state
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False
    if 'items_info' not in st.session_state:
        st.session_state.items_info = {}
    if 'original_image' not in st.session_state:
        st.session_state.original_image = None
    if 'annotated_image' not in st.session_state:
        st.session_state.annotated_image = None

    uploaded_image = st.file_uploader("📸 Upload an image of your fridge", type=["jpg", "jpeg", "png"])

    if uploaded_image and not st.session_state.analysis_complete:
        st.session_state.original_image = Image.open(uploaded_image)
        
        temp_file_path = f"temp_{uploaded_image.name}"
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_image.getvalue())

        with st.spinner("🔍 Analyzing fridge contents..."):
            analysis_result, annotated_image, items_info = analyze_fridge_image(temp_file_path)

        if analysis_result and annotated_image and items_info:
            st.session_state.annotated_image = annotated_image
            st.session_state.analysis_result = analysis_result
            st.session_state.items_info = items_info
            st.session_state.analysis_complete = True

        try:
            os.remove(temp_file_path)
        except PermissionError as e:
            st.warning(f"Could not delete temporary file: {e}")

    # Always display images if they exist in session state
    if st.session_state.original_image and st.session_state.annotated_image:
        col1, col2 = st.columns(2)
        with col1:
            st.image(st.session_state.original_image, caption='Uploaded Image', use_column_width=True)
        with col2:
            st.image(st.session_state.annotated_image, caption="Annotated Image", use_column_width=True)

    if st.session_state.analysis_complete:
        st.success("✅ Fridge analysis completed!")
        # st.text(st.session_state.analysis_result)

        # Display ingredient grid
        display_ingredient_grid(st.session_state.items_info)

        # Add a top margin to the button
        st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)
        
        if st.button("🍳 Generate Recipes", key="generate_recipes"):
            st.subheader("🍽️ Generated Recipes")
            with st.spinner("🔍 Searching for delicious recipes..."):
                recipes = get_recipes_from_spoonacular(list(st.session_state.items_info.keys()))
            
            if recipes:
                for i in range(0, len(recipes), 2):
                    cols = st.columns(2)
                    for j in range(2):
                        if i + j < len(recipes):
                            recipe = recipes[i + j]
                            with cols[j]:
                                with st.spinner(f"✨ Generating details for {recipe['title']}..."):
                                    recipe_details = generate_recipe_details_groq(recipe)
                                recipe_card = create_recipe_card(recipe, recipe_details)
                                st.markdown(recipe_card, unsafe_allow_html=True)
            else:
                st.warning("😕 No recipes found. Try uploading a different image with more ingredients.")
    
    # Add a top margin to the button
    st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)

    # if st.button("🔄 Reset"):
    #     for key in list(st.session_state.keys()):
    #         del st.session_state[key]
    #     st.rerun()

if __name__ == "__main__":
    main()