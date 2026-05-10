import google.generativeai as genai
import sys

api_key = 'AIzaSyC0bU_7hhtcdSdJuMBKomZ6uiOgIMnRMks'
genai.configure(api_key=api_key)

models_to_test = [
    'gemini-1.5-flash',
    'gemini-1.5-flash-latest',
    'gemini-1.5-pro',
    'gemini-1.0-pro',
    'gemini-pro'
]

print(f"Testing models with API Key: {api_key[:10]}...")

for model_name in models_to_test:
    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content("Hello, reply with 'OK'")
        print(f"SUCCESS: {model_name} -> {response.text.strip()}")
        sys.exit(0)
    except Exception as e:
        print(f"FAIL: {model_name} -> {str(e)}")

print("\nAll models failed. Checking available models...")
try:
    available = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
    print(f"Available models: {available}")
except Exception as e:
    print(f"Could not list models: {e}")
