import json
from textblob import TextBlob
from nltk.corpus import wordnet
from itertools import chain
import random

# Load the dataset
file_path = './data/dataset.jsonl'
data = []

with open(file_path, 'r') as f:
    for line in f:
        data.append(json.loads(line))

# Function to get synonyms
def get_synonyms(word):
    synonyms = wordnet.synsets(word)
    lemmas = set(chain.from_iterable([word.lemma_names() for word in synonyms]))
    if word in lemmas:
        lemmas.remove(word)
    return list(lemmas)

# Function to augment a sentence
def augment_sentence(sentence):
    words = sentence.split()
    new_sentence = []
    for word in words:
        if random.random() > 0.8:  # Replace word with a synonym 20% of the time
            synonyms = get_synonyms(word)
            if synonyms:
                new_word = random.choice(synonyms)
                new_sentence.append(new_word)
            else:
                new_sentence.append(word)
        else:
            new_sentence.append(word)
    return ' '.join(new_sentence)

# Augment the dataset
augmented_data = []

for item in data:
    augmented_item = item.copy()
    for msg in augmented_item['messages']:
        if msg['role'] == 'user' or msg['role'] == 'assistant':
            msg['content'] = augment_sentence(msg['content'])
    augmented_data.append(augmented_item)

# Save the augmented dataset
augmented_file_path = './data/preprocessing/augmented_data.jsonl'
with open(augmented_file_path, 'w') as f:
    for item in augmented_data:
        json.dump(item, f)
        f.write('\n')

print(f"Augmented data saved to {augmented_file_path}")