
import sys
import os
sys.path.append(os.path.abspath("."))
try:
    from src.images.document_reader import DocumentReader
    print("DocumentReader imported successfully!")
except SyntaxError as e:
    print(f"SyntaxError: {e}")
    print(f"Line: {e.lineno}, Offset: {e.offset}, Text: {e.text}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
