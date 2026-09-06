
from src.vision_nlp.image_quality import ImageQualityAnalyzer
import os

analyzer = ImageQualityAnalyzer(verbose=True)
path = "examples/cat-and-dog/dog.4001.jpg"
print(f"Checking {path}...")
if not os.path.exists(path):
    print("FILE NOT FOUND")
else:
    result = analyzer.analyze_image(path)
    print("Result keys:", result.keys())
    if "error" in result:
        print("Error:", result["error"])
    print("Overall Score:", result.get("overall_score"))
    print("Verdict:", result.get("verdict"))
