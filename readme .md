# FoodEase: AI-Powered Smart Kitchen Assistant 🍽️

FoodEase is an intelligent kitchen management system that helps you make the most of your ingredients and maintain healthy eating habits. Using advanced AI, it analyzes your refrigerator contents, suggests recipes, and tracks your nutritional intake over time.

## 🌟 Features

### 1. Ingredient Recognition
- 📸 Upload photos of your refrigerator contents
- 🤖 AI-powered ingredient detection and classification
- 📊 Visual ingredient mapping with bounding boxes
- 📝 Automatic quantity estimation

### 2. Recipe Generation
- 🍳 Get personalized recipe suggestions based on available ingredients
- 🔍 Smart recipe matching using the Spoonacular API
- 📖 Detailed recipe information including:
  - Cooking instructions
  - Nutritional information
  - Cooking time
  - Difficulty level
  - Price estimates
  - Dietary categories

### 3. Health Analytics
- 📈 Track daily calorie intake
- 🥗 Monitor macronutrient distribution
- 🍽️ Meal type analysis
- 🌎 Cuisine variety tracking
- 📊 Interactive data visualizations
- 📅 Historical recipe tracking

## 🚀 Installation

### Prerequisites
- Python 3.8+
- pip package manager
- Virtual environment (recommended)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/foodease.git
cd foodease
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the project root with your API keys:
```env
GOOGLE_API_KEY=your_google_api_key
GROQ_API_KEY=your_groq_api_key
SPOONACULAR_API_KEY=your_spoonacular_api_key
```

## 🔧 Configuration

### API Keys Required:
1. **Google API Key**: For Gemini Pro Vision API (ingredient detection)
   - Get it from [Google AI Studio](https://makersuite.google.com/app/apikey)

2. **Groq API Key**: For recipe detail generation
   - Get it from [Groq Cloud](https://console.groq.com/)

3. **Spoonacular API Key**: For recipe matching
   - Get it from [Spoonacular](https://spoonacular.com/food-api)

## 📁 Project Structure
```
foodease/
├── main.py              # Main application file
├── recipe_tracker.py    # Recipe tracking and analytics
├── requirements.txt     # Python dependencies
├── .env                # API keys and configuration
├── recipe_history.csv   # Saved recipe data
└── pages/
    └── health_analytics.py  # Health analytics dashboard
```

## 🎯 Usage

1. Start the application:
```bash
streamlit run main.py
```

2. Upload a photo of your refrigerator contents:
   - Click the upload button
   - Wait for AI analysis
   - Review detected ingredients

3. Generate recipes:
   - Click "Generate Recipes"
   - Browse suggested recipes
   - Select meal types
   - Save recipes you plan to cook

4. Track health metrics:
   - Navigate to the Health Analytics page
   - View nutritional trends
   - Filter by date range
   - Analyze eating patterns

## 📊 Data Storage

Recipe and nutritional data is stored in `recipe_history.csv` with the following columns:
- date
- recipe_name
- calories
- protein
- carbs
- fat
- cooking_time
- cuisine
- dietary_type
- meal_type

## 🛠️ Dependencies

Main libraries used:
- `streamlit`: Web application framework
- `google.generativeai`: Gemini Pro Vision API
- `groq`: Groq LLM API
- `opencv-python`: Image processing
- `pandas`: Data manipulation
- `plotly`: Data visualization
- `python-dotenv`: Environment configuration
- `pillow`: Image handling
- `requests`: HTTP requests

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Gemini Pro Vision API for image analysis
- Groq API for natural language processing
- Spoonacular API for recipe database
- Streamlit for the web framework
- All contributors and users of the application

## 📮 Contact

For questions and support, please open an issue in the GitHub repository or contact the maintainers directly.

## 🔮 Future Features

Planned enhancements:
- [ ] Multiple image upload support
- [ ] Ingredient expiration tracking
- [ ] Shopping list generation
- [ ] Meal planning calendar
- [ ] Recipe rating system
- [ ] Social sharing features
- [ ] Custom recipe addition
- [ ] Dietary restriction filters
- [ ] Nutritional goal setting
- [ ] Mobile app version