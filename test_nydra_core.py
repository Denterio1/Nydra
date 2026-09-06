
import sys
import os
# Ensure src is in path
sys.path.append(os.path.join(os.getcwd(), "src"))

try:
    import nydra_core
    print(f"NYDRA CORE STATUS:")
    print(f"  - Version:  {nydra_core.__version__}")
    print(f"  - Has Rust: {nydra_core.__has_rust__}")
    
    # Test basic math (Fallback or Rust)
    data = [1.0, 2.0, 3.0, 4.0, 5.0]
    mean = nydra_core.mean(data)
    std = nydra_core.std_dev(data)
    
    print(f"  - Test Mean: {mean}")
    print(f"  - Test Std:  {std}")
    
    if nydra_core.__has_rust__:
        print("\n🚀 RUST ENGINE ACTIVE")
    else:
        print("\n🐍 PYTHON FALLBACK ACTIVE (System stable)")
        
except Exception as e:
    print(f"❌ Integration Error: {e}")
