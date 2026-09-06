
import pandas as pd
from src.core.agent import Nydra
import logging

def test_debug():
    logging.basicConfig(level=logging.INFO)
    print("Initializing Nydra...")
    doctor = Nydra(verbose=True)
    
    print("Step 1: audit_text...")
    text_res = doctor.audit_text("Sample data analysis text.")
    print(f"audit_text status: {text_res.get('status')}")
    
    print("Step 2: audit_training_data...")
    df = pd.DataFrame({"a": [1, 2, 3, 4, 5, 6], "b": [3, 4, 5, 6, 7, 8], "y": [0, 1, 0, 1, 0, 1]})
    # Increased sample size to avoid CV error
    train_res = doctor.audit_training_data(df, target_col="y")
    print("audit_training_data complete!")
    print(f"Keys in result: {train_res.keys()}")

if __name__ == "__main__":
    test_debug()

