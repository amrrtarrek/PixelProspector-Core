from api.schemas import GameMLFeatures, UserReviewFeatures
from pydantic import ValidationError

def test_validation():
    print("--- Testing Schema Validation ---")
    
    # 1. تجربة بيانات صحيحة (المفروض تعدي)
    try:
        valid_data = {
            "gameplay_addictiveness": 0.9,
            "technical_polish": 0.8,
            "aesthetic_appeal": 0.7,
            "narrative_depth": 0.6,
            "replayability": 0.5,
            "viral_momentum": 0.4  # الـ Feature السادسة
        }
        GameMLFeatures(**valid_data)
        print("✅ Correct data passed!")
    except ValidationError:
        print("❌ Failed: Correct data should have passed.")

    # 2. تجربة بيانات ناقصة (المفروض تطلع Error)
    try:
        invalid_data = {"gameplay_addictiveness": 0.9} # ناقصة بقية الـ features
        GameMLFeatures(**invalid_data)
        print("❌ Failed: Missing data should have triggered an error.")
    except ValidationError as e:
        print(f"✅ Caught expected error: {len(e.errors())} missing fields found.")

if __name__ == "__main__":
    test_validation()