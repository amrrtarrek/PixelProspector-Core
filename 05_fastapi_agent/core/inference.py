import joblib
import os
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
MODELS_PATH = os.path.join(current_dir, '..', '03_supervised_ml', 'models')

def get_game_prediction(features_list):
    model = joblib.load(os.path.join(MODELS_PATH, 'game_svm.joblib'))
    prediction = model.predict([features_list])
    return int(prediction[0])

def get_user_prediction(features_list):
    model = joblib.load(os.path.join(MODELS_PATH, 'user_svm.joblib'))
    prediction = model.predict([features_list])
    return int(prediction[0])
if __name__ == "__main__":
    game_test = [0.90, 0.85, 0.88, 0.82, 0.80, 0.70]
    game_res = get_game_prediction(game_test)
    print(f"✅ Game Class: {game_res}") 

    
    user_test = [0.85, 0.05, 0.90, 0.88]
    user_res = get_user_prediction(user_test)
    print(f"✅ User Class: {user_res}") 