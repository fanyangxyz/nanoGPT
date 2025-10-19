"""
Prepare the Su Shi poems dataset for character-level language modeling.
Collects all poems from the source directory and encodes them character-by-character.
Will save train.bin, val.bin containing the ids, and meta.pkl containing the
encoder and decoder and some other related info.
"""
import os
import pickle
import numpy as np
from pathlib import Path

# Path to the Su Shi poems data
source_data_path = '/Users/fanyang/code/classical-modern/reproduce/su-shi-poems'

# Collect all poems
print("Collecting all Su Shi poems...")
all_text = []
poem_count = 0

# Iterate through all directories in the source path
for poem_dir in sorted(Path(source_data_path).iterdir()):
    if poem_dir.is_dir():
        text_file = poem_dir / 'text.txt'
        if text_file.exists():
            with open(text_file, 'r', encoding='utf-8') as f:
                poem_text = f.read().strip()
                if poem_text:
                    all_text.append(poem_text)
                    poem_count += 1

# Join all poems with newlines to separate them
data = '\n\n'.join(all_text)
print(f"Collected {poem_count} poems")
print(f"Total length in characters: {len(data):,}")

# Get all the unique characters that occur in this text
chars = sorted(list(set(data)))
vocab_size = len(chars)
print(f"Unique characters: {vocab_size:,}")
print("Sample characters:", ''.join(chars[:20]) + '...' if len(chars) > 20 else ''.join(chars))

# Create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }

def encode(s):
    return [stoi[c] for c in s]  # encoder: take a string, output a list of integers

def decode(l):
    return ''.join([itos[i] for i in l])  # decoder: take a list of integers, output a string

# Create the train and test splits (90/10)
n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

# Encode both to integers
train_ids = encode(train_data)
val_ids = encode(val_data)
print(f"Train has {len(train_ids):,} tokens")
print(f"Val has {len(val_ids):,} tokens")

# Export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

# Save the meta information as well, to help us encode/decode later
meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
}
with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)

print("Done! Files saved:")
print("  - train.bin")
print("  - val.bin")
print("  - meta.pkl")
