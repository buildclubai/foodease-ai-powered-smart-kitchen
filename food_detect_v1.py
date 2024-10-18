import base64
import os
import requests
import random
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
import google.generativeai as genai
from groq import Groq
from google.api_core import exceptions as google_exceptions
import re

# Load environment variables
load_dotenv()

# Configure Google Generative AI
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize Groq client (if still needed)
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Spoonacular API key
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
        Location: [Bounding box coordinates as [ymin, xmin, ymax, xmax] if possible, or describe the location in the fridge]

        Be as specific as possible about the item and its quantity. If you can't determine the exact quantity, provide an estimate or range.
        For the location, always try to provide bounding box coordinates. If you can't determine precise coordinates, provide an approximate location.
        After listing all items, on a new line, state the total number of distinct food items.
        """
        
        response_items = model.generate_content([prompt_items, image])
        items_info = parse_item_info(response_items.text.strip())
        
        # Generate annotated image
        annotated_image = generate_annotated_image(image_path, items_info)
        
        # Prepare analysis result
        analysis_result = "\n".join([f"{item}: {info['quantity']} - {info['location']}" for item, info in items_info.items()])
        analysis_result += f"\n\nTotal number of distinct food items: {len(items_info)}"
        
        return analysis_result, annotated_image, items_info

    except google_exceptions.ResourceExhausted:
        print("API quota exceeded. Please try again later.")
        return None, None, None
    except Exception as e:
        print(f"An error occurred during image analysis: {str(e)}")
        return None, None, None

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
    # Try to find a list of four numbers in the text
    match = re.search(r'\[?\s*(\d+(?:\.\d+)?)\s*,?\s*(\d+(?:\.\d+)?)\s*,?\s*(\d+(?:\.\d+)?)\s*,?\s*(\d+(?:\.\d+)?)\s*\]?', text)
    if match:
        return [float(match.group(i)) for i in range(1, 5)]
    else:
        # If no valid bounding box is found, return default
        return [0, 0, 0, 0]

def generate_annotated_image(image_path, items_info):
    image = cv2.imread(image_path)
    height, width = image.shape[:2]

    for item, info in items_info.items():
        box = info['box']
        quantity = info['quantity']
        
        if box != [0, 0, 0, 0]:
            ymin, xmin, ymax, xmax = map(int, box)
            
            # Ensure coordinates are within image boundaries
            xmin = max(0, min(xmin, width - 1))
            ymin = max(0, min(ymin, height - 1))
            xmax = max(0, min(xmax, width - 1))
            ymax = max(0, min(ymax, height - 1))
            
            # Only draw rectangle if coordinates are valid
            if xmin < xmax and ymin < ymax:
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                label = f"{item}: {quantity}"
                cv2.putText(image, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            else:
                print(f"Invalid bounding box for {item}: {box}")
        else:
            print(f"No bounding box for {item}")

    # Convert back to PIL Image for consistency
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

def generate_recipes_spoonacular(ingredients):
    base_url = "https://api.spoonacular.com/recipes/findByIngredients"
    params = {
        "apiKey": SPOONACULAR_API_KEY,
        "ingredients": ",".join(ingredients),
        "number": 3,
        "ranking": 1,
        "ignorePantry": True
    }
    
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        recipes = response.json()
        
        if not recipes:
            print("No recipes found.")
            return []
        
        detailed_recipes = []
        for recipe in recipes:
            recipe_id = recipe['id']
            detailed_recipe = get_recipe_details(recipe_id)
            if detailed_recipe:
                detailed_recipes.append(detailed_recipe)
        
        return detailed_recipes
    except requests.exceptions.RequestException as e:
        print(f"Error calling Spoonacular API: {e}")
        return []

def get_recipe_details(recipe_id):
    base_url = f"https://api.spoonacular.com/recipes/{recipe_id}/information"
    params = {
        "apiKey": SPOONACULAR_API_KEY,
        "includeNutrition": True
    }
    
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        recipe_info = response.json()
        
        return {
            'name': recipe_info['title'],
            'ingredients': recipe_info['extendedIngredients'],
            'instructions': recipe_info['instructions'],
            'calories': next((nutrient['amount'] for nutrient in recipe_info['nutrition']['nutrients'] if nutrient['name'] == 'Calories'), None)
        }
    except requests.exceptions.RequestException as e:
        print(f"Error getting recipe details: {e}")
        return None

def main():
    image_path = "images/Fridge1.jpg"
    
    # Analyze the fridge image
    analysis_result, annotated_image, items_info = analyze_fridge_image(image_path)
    
    if analysis_result and annotated_image and items_info:
        print("Fridge Analysis Result:")
        print(analysis_result)
        
        # Save the annotated image
        annotated_image.save("annotated_fridge.jpg")
        print("Annotated image saved as 'annotated_fridge.jpg'")
        
        # Extract ingredients from the items_info
        ingredients = list(items_info.keys())
        print("\nExtracted Ingredients:")
        print(ingredients)
        
        if ingredients:
            # Generate recipes using Spoonacular
            recipes = generate_recipes_spoonacular(ingredients)
            
            if recipes:
                print("\nGenerated Recipes:")
                for i, recipe in enumerate(recipes, 1):
                    print(f"\nRecipe {i}: {recipe['name']}")
                    print("Ingredients used:")
                    for ingredient in recipe['ingredients']:
                        print(f"- {ingredient['name']}")
                    print("Instructions:")
                    print(recipe['instructions'])
                    print(f"Approximate calories: {recipe['calories']}")
            else:
                print("No recipes could be generated.")
        else:
            print("No ingredients were extracted. Please check the image analysis result.")
    else:
        print("Image analysis failed. Please try again later.")

if __name__ == "__main__":
    main()