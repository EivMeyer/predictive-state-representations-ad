import argparse
import os
import shutil
from pathlib import Path
import random
from typing import List, Optional

def get_files(directory: Path, extensions: Optional[List[str]] = None) -> List[Path]:
    """
    Get all files in directory matching specified extensions.
    
    Args:
        directory: Directory to search
        extensions: List of file extensions to include (e.g. ['.xml', '.txt'])
                   If None, include all files
    """
    files = []
    for f in directory.iterdir():
        if f.is_file():
            if extensions is None or f.suffix.lower() in extensions:
                files.append(f)
    return sorted(files)

def split_dataset(input_dir: str, train_ratio: float = 0.8, extensions: Optional[List[str]] = None, 
                 seed: int = 42, dry_run: bool = False):
    """
    Split files between train and test subdirectories.
    
    Args:
        input_dir: Directory containing the files
        train_ratio: Ratio of files to use for training (default 0.8)
        extensions: List of file extensions to include (e.g. ['.xml', '.txt'])
        seed: Random seed for reproducibility
        dry_run: If True, print what would be done without moving files
    """
    input_path = Path(input_dir)
    if not input_path.exists():
        raise ValueError(f"Input directory does not exist: {input_dir}")
    
    # Create train and test subdirectories
    train_dir = input_path / 'train'
    test_dir = input_path / 'test'
    
    if not dry_run:
        train_dir.mkdir(parents=True, exist_ok=True)
        test_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all matching files
    files = get_files(input_path, extensions)
    if not files:
        ext_str = f" with extensions {extensions}" if extensions else ""
        print(f"No files found in {input_dir}{ext_str}")
        return
    
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Shuffle and split files
    random.shuffle(files)
    split_idx = int(len(files) * train_ratio)
    train_files = files[:split_idx]
    test_files = files[split_idx:]
    
    # Print summary
    print(f"Found {len(files)} total files")
    print(f"Selected extensions: {extensions if extensions else 'all'}")
    print(f"Files per extension:")
    ext_counts = {}
    for f in files:
        ext = f.suffix.lower()
        ext_counts[ext] = ext_counts.get(ext, 0) + 1
    for ext, count in sorted(ext_counts.items()):
        print(f"  {ext}: {count} files")
    print(f"\nSplitting with {train_ratio:.0%} train ratio:")
    print(f"  Moving {len(train_files)} files to train/")
    print(f"  Moving {len(test_files)} files to test/")
    
    if dry_run:
        print("\nDRY RUN - no files will be moved")
        print("\nFiles that would be moved to train/:")
        for f in train_files[:5]:
            print(f"  {f.name}")
        if len(train_files) > 5:
            print(f"  ... and {len(train_files)-5} more")
        print("\nFiles that would be moved to test/:")
        for f in test_files[:5]:
            print(f"  {f.name}")
        if len(test_files) > 5:
            print(f"  ... and {len(test_files)-5} more")
        return

    # Move files
    print("\nMoving files...")
    for f in train_files:
        shutil.move(str(f), str(train_dir / f.name))
    for f in test_files:
        shutil.move(str(f), str(test_dir / f.name))

def main():
    parser = argparse.ArgumentParser(description='Split dataset files into train and test sets')
    parser.add_argument('input_dir', type=str, help='Input directory containing files')
    parser.add_argument('--train-ratio', type=float, default=0.8,
                      help='Ratio of files to use for training (default: 0.8)')
    parser.add_argument('--extensions', type=str, nargs='+',
                      help='File extensions to include (e.g. .xml .txt). Default: all files')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed (default: 42)')
    parser.add_argument('--dry-run', action='store_true',
                      help='Print what would be done without moving files')
    args = parser.parse_args()

    # Normalize extensions to lowercase with leading dot
    extensions = None
    if args.extensions:
        extensions = [ext.lower() if ext.startswith('.') else f'.{ext.lower()}' 
                     for ext in args.extensions]

    try:
        split_dataset(args.input_dir, args.train_ratio, extensions, args.seed, args.dry_run)
        if not args.dry_run:
            print("\nDataset split successfully!")
    except Exception as e:
        print(f"Error splitting dataset: {str(e)}")
        raise

if __name__ == "__main__":
    main()