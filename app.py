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
import matplotlib.pyplot as plt
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

# Encode the image in base64
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

        # Retry logic for quota errors
        for attempt in range(3):  # Retry up to 3 times
            try:
                response_items = model.generate_content([prompt_items, image])
                break  # Break the loop if successful
            except Exception as e:
                if '429' in str(e):
                    if attempt < 2:  # If not the last attempt
                        st.warning("Quota exceeded. Retrying in 5 seconds...")
                        time.sleep(5)  # Wait before retrying
                    else:
                        st.error("Quota exceeded. Please try again later.")
                        return None, None, None
                else:
                    st.error(f"An error occurred: {str(e)}")
                    return None, None, None
        
        items_info = parse_item_info(response_items.text.strip())
        annotated_image = generate_annotated_image(image_path, items_info)

        # Modify this line to exclude location details
        analysis_result = "\n".join([f"{item}: {info['quantity']}" for item, info in items_info.items()])
        analysis_result += f"\n\nTotal number of distinct food items: {len(items_info)}"
        
        return analysis_result, annotated_image, items_info

    except Exception as e:  # General exception handling for errors in the AI service
        st.error(f"An error occurred: {str(e)}")
        return None, None, None
    finally:
        # Ensure the image file is closed properly
        image.close()

        
        
# Parse item info from the model's response
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

# Parse bounding box information
def parse_bounding_box(text):
    match = re.search(r'\[?\s*(\d+)\s*,?\s*(\d+)\s*,?\s*(\d+)\s*,?\s*(\d+)\s*\]?', text)
    if match:
        return [int(match.group(i)) for i in range(1, 5)]
    return [0, 0, 0, 0]

# Convert bounding box coordinates
def convert_coordinates(box, original_width, original_height):
    ymin, xmin, ymax, xmax = box
    return [
        int(ymin / 1000 * original_height),
        int(xmin / 1000 * original_width),
        int(ymax / 1000 * original_height),
        int(xmax / 1000 * original_width)
    ]

# Generate annotated image with bounding boxes
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

# Generate recipes using Groq model
def generate_recipes_groq(ingredients):
    prompt = f"""
    Based on the following ingredients found in a refrigerator, suggest 2 recipes:  # Changed to 2 recipes
    {', '.join(ingredients)}

    For each recipe, provide:
    1. Recipe name
    2. List of ingredients with quantities
    3. Cooking instructions
    4. Estimated calories per serving
    5. Cuisine type

    Present the information in a clear, structured format.
    """
    
    response = groq_client.chat.completions.create(
        model="mixtral-8x7b-32768",
        messages=[
            {"role": "system", "content": "You are a helpful culinary assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=1000
    )

    return response.choices[0].message.content

# Spoonacular API recipe search function
def search_recipes_spoonacular(ingredients, cuisine=None, diet=None, intolerances=None, recipe_type=None):
    url = "https://api.spoonacular.com/recipes/findByIngredients"
    params = {
        'apiKey': SPOONACULAR_API_KEY,
        'ingredients': ','.join(ingredients),
        'number': 2,  # Limit to 2 recipes
        'ranking': 1
    }
    # Add additional parameters if provided
    if cuisine and cuisine != "Any":
        params['cuisine'] = cuisine
    if diet and diet != "Any":
        params['diet'] = diet
    if intolerances:
        params['intolerances'] = ','.join(intolerances)
    if recipe_type and recipe_type != "Any":
        params['type'] = recipe_type

    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Error: {response.status_code} - {response.text}")
        return []

def generate_nutritional_info(ingredients):
    prompt = f"""
    Provide the nutritional information for the following ingredients:
    {', '.join(ingredients)}

    Include total calories, protein, total fat, carbohydrates, and health score.
    Present the information in a clear format.
    """

    response = groq_client.chat.completions.create(
        model="mixtral-8x7b-32768",
        messages=[
            {"role": "system", "content": "You are a helpful nutritional assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=300
    )

    return response.choices[0].message.content
       
# Get recipe details from Spoonacular API
def get_recipe_details(recipe_id, ingredients):
    url = f"https://api.spoonacular.com/recipes/{recipe_id}/information"
    params = {
        'apiKey': SPOONACULAR_API_KEY
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        # If Spoonacular fails, generate details using Groq
        st.warning("Generating recipe details using Groq...")
        return generate_recipes_groq(ingredients)
    
def plot_nutrition_pie_chart(calories, protein, total_fat, carbs):
    labels = ['Calories', 'Protein', 'Total Fat', 'Carbs']
    sizes = [calories, protein, total_fat, carbs]
    colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99']
    
    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    # Save it to a BytesIO object to use in Streamlit
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return buf

    
def main():
    st.title("Fridge Food Item Detector and Recipe Generator")

    st.sidebar.subheader("Search for Recipes")
    
    # Sidebar for recipe search options
    cuisine = st.sidebar.selectbox("Cuisine", ["Any", "Italian", "Chinese", "Indian", "Mexican", "American", "French", "Thai"])
    diet = st.sidebar.selectbox("Diet", ["Any", "Vegetarian", "Vegan", "Keto", "Paleo"])
    intolerances = st.sidebar.multiselect("Intolerances", ["Dairy", "Gluten", "Peanut", "Seafood", "Soy", "Tree Nut", "Wheat"])
    recipe_type = st.sidebar.selectbox("Type of Meal", ["Any", "Breakfast", "Lunch", "Dinner", "Snack", "Dessert"])

    # File uploader for fridge image
    uploaded_image = st.file_uploader("Upload an image of your fridge", type=["jpg", "jpeg", "png"])

    if uploaded_image:
        st.image(uploaded_image, caption='Uploaded Image', use_column_width=True)
        temp_file_path = f"temp_{uploaded_image.name}"
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_image.getvalue())

        # Analyze the fridge image
        analysis_result, annotated_image, items_info = analyze_fridge_image(temp_file_path)

        if analysis_result and annotated_image and items_info:
            st.success("Fridge analysis completed!")
            st.text(analysis_result)

            # Display the annotated image
            st.image(annotated_image, caption="Annotated Image", use_column_width=True)

            # Extract ingredients for recipe generation
            ingredients = list(items_info.keys())
            if ingredients:
                # Search recipes based on sidebar selections
                recipes = search_recipes_spoonacular(ingredients, cuisine if cuisine != "Any" else None, 
                                                      diet if diet != "Any" else None, 
                                                      intolerances, 
                                                      recipe_type if recipe_type != "Any" else None)

                if recipes:
                    # Create two columns for displaying recipes side by side
                    col1, col2 = st.columns(2)

                    # Loop through the recipes
                    for i, recipe in enumerate(recipes):
                        with col1 if i % 2 == 0 else col2:  # Use col1 for even indices, col2 for odd indices
                            st.subheader(recipe['title'])
                            st.image(recipe['image'], width=200)
                            st.markdown(f"[Link to recipe](https://spoonacular.com/recipes/{recipe['title'].replace(' ', '-').lower()}-{recipe['id']})")

                            # Get detailed recipe info
                            recipe_details = get_recipe_details(recipe['id'], ingredients)

                            # Display recipe instructions and nutritional info
                            if isinstance(recipe_details, dict):  # Check if recipe details are a dictionary
                                st.markdown("**Instructions:**")
                                instructions = recipe_details.get("instructions", "Instructions not available.")
                                instructions_clean = re.sub(r'<.*?>', '', instructions)  # Remove HTML tags
                                st.write(instructions_clean)

                                # Display nutritional information in a table-like format
                                if 'nutrition' in recipe_details:
                                    nutritional_info = recipe_details['nutrition']
                                    calories = nutritional_info.get("nutrients", [{}])[0].get("amount", 0)
                                    protein = nutritional_info.get("nutrients", [{}])[1].get("amount", 0)
                                    total_fat = nutritional_info.get("nutrients", [{}])[2].get("amount", 0)
                                    carbs = nutritional_info.get("nutrients", [{}])[3].get("amount", 0)
                                    health_score = recipe_details.get("healthScore", "N/A")

                                    # Display nutritional information in a table-like format
                                    st.markdown(f"**Quickview**  \n"
                                                f"|{calories} Calories|  |{protein}g Protein|  |{total_fat}g Total Fat|  |{carbs}g Carbs|  |{health_score}% Health Score|")

                                    # Generate and display the pie chart
                                    pie_chart_buf = plot_nutrition_pie_chart(calories, protein, total_fat, carbs)
                                    st.image(pie_chart_buf, caption='Nutritional Breakdown', use_column_width=True)
                                else:
                                    # If no nutrition data from Spoonacular, generate it using Groq
                                    st.warning("Generating nutritional information using Groq...")
                                    nutritional_info = generate_nutritional_info([recipe['title']])  # Use recipe name only
                                    st.markdown("**Quickview**  \n" + nutritional_info)




                else:
                    st.warning("No recipes found for the provided ingredients.")
            else:
                st.error("No ingredients were detected.")

        # Ensure file is closed before deletion
        try:
            os.remove(temp_file_path)
        except PermissionError as e:
            st.warning(f"Could not delete temporary file: {e}")

if __name__ == "__main__":
    main()
