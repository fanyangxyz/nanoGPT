# Su Shi Poems Dataset

This dataset contains 298 classical Chinese poems by Su Shi (苏轼), a famous Song Dynasty poet.

## Dataset Statistics

- **Total poems**: 298
- **Total characters**: 40,132
- **Vocabulary size**: 2,852 unique characters
- **Training tokens**: 36,118 (90%)
- **Validation tokens**: 4,014 (10%)

## Data Source

Source directory: `/Users/fanyang/code/classical-modern/reproduce/su-shi-poems`

Each poem is stored in a separate directory with a `text.txt` file containing the poem content.

## Preparation

Run the preparation script to generate the binary training files:

```bash
python data/su_shi_poems/prepare.py
```

This will create:
- `train.bin` - Training data (binary format)
- `val.bin` - Validation data (binary format)
- `meta.pkl` - Metadata including vocabulary and character mappings

## Character Encoding

The dataset uses character-level encoding, mapping each unique Chinese character (and punctuation) to an integer index. This is ideal for learning the structure and style of classical Chinese poetry.
