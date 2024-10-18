import requests
import base64
from groq import Groq
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Initialize Groq client
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def analyze_fridge_image(image_path):
    base64_image = encode_image(image_path)
    
    prompt = f"""
    Analyze the following image of a fridge and list all the food items you can identify.
    Then, count the total number of distinct food items.
    
    Image: data:image/jpeg;base64,{base64_image}
    """
    
    response = groq_client.chat.completions.create(
        model="llama-3.2-3b-vision-preview",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that analyzes images of fridges."},
            {"role": "user", "content": prompt}
        ]
    )
    
    return response.choices[0].message.content

def generate_recipes(ingredients):
    prompt = f"""
    Generate 3 recipe ideas using some or all of the following ingredients:
    {', '.join(ingredients)}
    
    For each recipe, provide:
    1. Recipe name
    2. Ingredients used (from the list)
    3. Brief cooking instructions
    """
    
    response = groq_client.chat.completions.create(
        model="llama-3.2-90b-chat",  # Using a text-based model for recipe generation
        messages=[
            {"role": "system", "content": "You are a helpful assistant that generates recipes."},
            {"role": "user", "content": prompt}
        ]
    )
    
    return response.choices[0].message.content

def get_calorie_info(recipe):
    # This is a placeholder function. In a real application, you would integrate
    # with a nutrition API to get accurate calorie information.
    # For demonstration purposes, we'll return a random number.
    import random
    return random.randint(300, 800)

def main():
    image_path = "images/Fridge1.jpg"
    
    # Analyze the fridge image
    analysis_result = analyze_fridge_image(image_path)
    print("Fridge Analysis Result:")
    print(analysis_result)
    
    # Extract ingredients from the analysis result
    # This is a simplification. In a real application, you'd need more robust parsing.
    ingredients = [item.strip() for item in analysis_result.split('\n') if item.strip()]
    
    # Generate recipes
    recipes = generate_recipes(ingredients)
    print("\nGenerated Recipes:")
    print(recipes)
    
    # Get calorie information for each recipe
    # This is a simplification. In a real application, you'd parse the recipes and get accurate calorie info.
    print("\nCalorie Information:")
    for i, recipe in enumerate(recipes.split('\n\n'), 1):
        calories = get_calorie_info(recipe)
        print(f"Recipe {i}: Approximately {calories} calories")

if __name__ == "__main__":
    main()