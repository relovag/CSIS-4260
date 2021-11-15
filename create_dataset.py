import os
import json
import random
import pickle

data_dir = 'data/'
files = os.listdir(data_dir)


def generate_positives():
    data = []
    for file in files:
        full_path = os.path.join(data_dir, file)
        with open(full_path, 'rb') as f:
            for line in f:
                if line:
                    entry = json.loads(line)
                    datum = {}
                    datum['caption_title_and_reference_description'] = []
                    for el in entry['wit_features']:
                        if el.get('caption_title_and_reference_description'):
                            datum['caption_title_and_reference_description'].append(
                                el['caption_title_and_reference_description'])

                    datum['b64_bytes'] = entry['b64_bytes']
                    datum['target'] = 1
                    if datum['caption_title_and_reference_description'] and datum['b64_bytes'] != "":
                        data.append(datum)


def generate_negatives(data):
    neg_data = []
    for datum in data:
        new = {}
        new['b64_bytes'] = datum['b64_bytes']
        new['target'] = -1
        neg_caption = random.choice(data)
        new['caption_title_and_reference_description'] = neg_caption['caption_title_and_reference_description']
        if neg_caption['b64_bytes'] != datum['b64_bytes']:
            neg_data.append(new)
    data.extend(neg_data)
    return data


if __name__ == "__main__":
    data = generate_positives()
    final_data = generate_negatives(data)
    with open('data/wiki_data.pkl', 'wb') as f:
        pickle.dump(final_data, f)
