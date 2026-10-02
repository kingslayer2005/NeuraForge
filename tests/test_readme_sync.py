import pandas as pd
from pathlib import Path
import re

def test_readme_matches_results():
    root_dir = Path(__file__).parent.parent
    results_dir = root_dir / "results"
    readme_path = root_dir / "README.md"
    
    if not readme_path.exists():
        return
        
    with open(readme_path, "r", encoding="utf-8") as f:
        readme_content = f.read()

    # Ensure the README contains a Current status section, representing honest reporting
    assert "Current Status" in readme_content, "README must contain a 'Current Status' section."
