import torch
import numpy as np
from pathlib import Path
import argparse
import shutil
from tqdm import tqdm

def to_tensor(data):
    if isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data, torch.Tensor):
        return data
    else:
        raise TypeError(f"Unsupported data type: {type(data)}")

def process_batch(batch, new_batch_size):
    new_batches = []
    for i in range(0, len(batch['observations']), new_batch_size):
        new_batch = {key: value[i:i+new_batch_size] for key, value in batch.items()}
        new_batches.append(new_batch)
    return new_batches

def change_batch_size(input_dir: Path, output_dir: Path, new_batch_size: int):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_files = sorted(input_dir.glob('batch_*.pt'))
    total_samples = 0
    new_batch_index = 0
    current_batch = {key: [] for key in ['observations', 'actions', 'ego_states', 'next_observations', 'next_actions', 'dones']}

    for file in tqdm(input_files, desc="Processing batches"):
        batch = torch.load(file)
        total_samples += len(batch['observations'])
        
        new_batches = process_batch(batch, new_batch_size)
        
        for new_batch in new_batches:
            for key in current_batch.keys():
                current_batch[key].extend(new_batch[key])
            
            while len(current_batch['observations']) >= new_batch_size:
                output_batch = {key: torch.stack([to_tensor(item) for item in value[:new_batch_size]]) 
                                for key, value in current_batch.items()}
                torch.save(output_batch, output_dir / f'batch_{new_batch_index}.pt')
                new_batch_index += 1
                
                # Remove the saved data from current_batch
                for key in current_batch.keys():
                    current_batch[key] = current_batch[key][new_batch_size:]

    # Save any remaining data
    if len(current_batch['observations']) > 0:
        output_batch = {key: torch.stack([to_tensor(item) for item in value]) 
                        for key, value in current_batch.items()}
        torch.save(output_batch, output_dir / f'batch_{new_batch_index}.pt')
        new_batch_index += 1

    print(f"Converted {total_samples} samples into {new_batch_index} batches of size {new_batch_size}")

def main():
    parser = argparse.ArgumentParser(description="Change the batch size of a dataset")
    parser.add_argument("input_dir", type=str, help="Input directory containing the original dataset")
    parser.add_argument("output_dir", type=str, help="Output directory for the new dataset")
    parser.add_argument("new_batch_size", type=int, help="New batch size")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if input_dir == output_dir:
        print("Input and output directories must be different.")
        return

    if output_dir.exists():
        user_input = input(f"Output directory {output_dir} already exists. Do you want to overwrite it? (y/n): ")
        if user_input.lower() != 'y':
            print("Operation cancelled.")
            return
        shutil.rmtree(output_dir)

    change_batch_size(input_dir, output_dir, args.new_batch_size)

if __name__ == "__main__":
    main()