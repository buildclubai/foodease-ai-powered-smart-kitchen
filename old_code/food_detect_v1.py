import base64
from groq import Groq
from dotenv import load_dotenv
import os
import requests
import random

# Load environment variables
load_dotenv()

# Initialize Groq client
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Spoonacular API key
SPOONACULAR_API_KEY = os.getenv("SPOONACULAR_API_KEY")

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return f"data:image/jpeg;base64,{base64.b64encode(image_file.read()).decode('utf-8')}"

def analyze_fridge_image(image_path):
    image_data_url = encode_image(image_path)
    
    completion = groq_client.chat.completions.create(
        model="llama-3.2-11b-vision-preview",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Analyze this image of a fridge and list all the food items you can identify in detail with their quantity. For example: Tomatoes (3). Show each ingredient and its quantity on a separate line. Then, count the total number of distinct food items."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_data_url
                        }
                    }
                ]
            }
        ],
        temperature=0.7,
        max_tokens=1024,
        top_p=1,
        stream=False,
        stop=None
    )
    
    return completion.choices[0].message.content

def extract_ingredients(analysis_result):
    lines = analysis_result.split('\n')
    ingredients = []
    for line in lines:
        if '(' in line and ')' in line:
            item = line.split('(')[0].strip()
            if item and not any(word in item.lower() for word in ['unfortunately', 'however', 'please', 'total']):
                ingredients.append(item)
    return ingredients

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
        response.raise_for_status()  # Raises an HTTPError for bad responses
        recipes = response.json()
        
        if not recipes:
            print("No recipes found. Using fallback method.")
            return generate_fallback_recipes(ingredients)
        
        detailed_recipes = []
        for recipe in recipes:
            recipe_id = recipe['id']
            detailed_recipe = get_recipe_details(recipe_id)
            if detailed_recipe:
                detailed_recipes.append(detailed_recipe)
        
        return detailed_recipes if detailed_recipes else generate_fallback_recipes(ingredients)
    except requests.exceptions.RequestException as e:
        print(f"Error calling Spoonacular API: {e}")
        return generate_fallback_recipes(ingredients)

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

def generate_fallback_recipes(ingredients):
    print("Using fallback recipe generation method.")
    recipes = []
    for i in range(3):
        recipe = {
            'name': f"Simple {random.choice(ingredients)} Recipe {i+1}",
            'ingredients': [{'name': ing} for ing in random.sample(ingredients, min(len(ingredients), 5))],
            'instructions': "Mix all ingredients and cook to your liking.",
            'calories': random.randint(200, 800)
        }
        recipes.append(recipe)
    return recipes

def main():
    image_path = "images/Fridge1.jpg"
    
    # Analyze the fridge image
    analysis_result = analyze_fridge_image(image_path)
    print("Fridge Analysis Result:")
    print(analysis_result)
    
    # Extract ingredients from the analysis result
    ingredients = extract_ingredients(analysis_result)
    print("\nExtracted Ingredients:")
    print(ingredients)
    
    if ingredients:
        # Generate recipes using Spoonacular or fallback method
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

if __name__ == "__main__":
    main()