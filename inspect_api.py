import re

with open("api.py", "r", encoding="utf-8", errors="ignore") as f:
    content = f.read()

# Replace hashed_password=None or missing hashed_password for google oauth with a dummy hash
# Or make hashed_password nullable in DBUser if possible.
# Let's search where DBUser or google user is created.
print("Length of api.py:", len(content))
