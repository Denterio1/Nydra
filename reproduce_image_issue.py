
import os
import pandas as pd
from src.vision_nlp.image_loader import ImageLoader
from src.vision_nlp.image_quality import analyze_dataset_quality

def main():
    target_dir = "examples/cat-and-dog"
    if not os.path.exists(target_dir):
        print(f"Directory {target_dir} not found.")
        return

    print(f"Scanning {target_dir}...")
    loader = ImageLoader(target_dir, verbose=True)
    loader_df, _ = loader.load()
    
    if loader_df is None or loader_df.empty:
        print("No images found.")
        return

    # Only take a few images for speed
    loader_df = loader_df.head(5)
    print(f"Analyzing {len(loader_df)} images...")
    
    df_q, report = analyze_dataset_quality(loader_df, verbose=True)
    
    print("\nResults:")
    for i, row in df_q.iterrows():
        print(f"File: {row['file_name']}, Score: {row['overall_score']}, Verdict: {row['verdict']}")
        print(f"  Sharpness: {row.get('sharpness_score')}")
        print(f"  Exposure:  {row.get('exposure_score')}")
        print(f"  Noise:     {row.get('noise_score')}")
        print(f"  Contrast:  {row.get('contrast_score')}")
        print(f"  Artifacts: {row.get('artifact_score')}")

if __name__ == "__main__":
    main()
