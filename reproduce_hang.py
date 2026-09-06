
import pandas as pd
import numpy as np
from src.vision_nlp.label_quality import LabelQualityAnalyzer

def reproduce():
    print("Creating tiny dataset...")
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4], "y": [0, 1]})
    
    print("Initializing LabelQualityAnalyzer...")
    analyzer = LabelQualityAnalyzer(verbose=True)
    
    print("Running analyze()...")
    try:
        report = analyzer.analyze(df, label_col="y")
        print("Success!")
        print(f"Score: {report.label_quality_score}")
    except Exception as e:
        print(f"Failed with error: {e}")

if __name__ == "__main__":
    reproduce()
