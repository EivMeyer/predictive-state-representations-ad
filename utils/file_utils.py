from pathlib import Path
from typing import Optional

def find_model_path(base_path: Path, model_path: Path) -> Optional[Path]:
    possible_paths = [
        Path(model_path),
        Path(base_path) / model_path,
        Path(base_path) / 'output' / model_path,
        Path(base_path) / 'models' / model_path,
        Path(base_path) / 'output' / 'models' / model_path,
    ]
    
    for path in possible_paths:
        if path.is_file():
            return path
    
    # If no file is found, return None
    return None