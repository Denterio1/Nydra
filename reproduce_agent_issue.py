import pandas as pd
from src.core.agent import Nydra

def test_no_dedup():
    df = pd.DataFrame({
        "a": [1, 1, 2],
        "b": [4, 4, 5],
        "c": [None, 7, 8]
    })
    doctor = Nydra(remove_dupes=False)
    result = doctor.inspect_df(df)
    print("Cleaning log:", result.get("cleaning_log"))
    if "duplicates_removed" in result.get("cleaning_log", {}):
        print("FAIL: duplicates_removed is in cleaning_log")
    else:
        print("PASS: duplicates_removed is NOT in cleaning_log")

if __name__ == "__main__":
    test_no_dedup()

