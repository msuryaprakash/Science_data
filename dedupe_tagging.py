# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 23:47:22 2019

@author: ayrus
"""
import pandas as pd
df_with_truth = pd.read_csv('D:/Downloads/Programs/Compressed/deduplication-slides-master/deduplication-slides-master/restaurant.csv', skip_blank_lines=True)

df_with_truth.head(9)

df = df_with_truth.drop(columns=['cluster', 'phone', 'type'])
df.head(9)



import re

irrelevant_regex = re.compile(r'[^a-z0-9/s]')
multispace_regex = re.compile(r'/s/s+')

def assign_no_symbols_name(df):
    return df.assign(
        name=df['name']
             .str.replace(irrelevant_regex, ' ')
             .str.replace(multispace_regex, ' '))

df = assign_no_symbols_name(df)
df.head(9)

import numpy as np

all_addresses = df['addr'].str.cat(df['city'], sep=', ').values
unique_addresses = np.unique(all_addresses)
print(len(all_addresses), len(unique_addresses))

819

import os.path
import json

geocoding_filename = 'D:/Downloads/Programs/Compressed/deduplication-slides-master/deduplication-slides-master/address_to_geocoding.json'

def geocode_addresses(address_to_geocoding):
    remaining_addresses = (
        set(unique_addresses) -
        set(k for k, v in address_to_geocoding.items() if v is not None and 'lat' in v))

    with requests.Session() as session:
        for i, address in enumerate(remaining_addresses):
            print(f"Geocoding {i + 1}/{len(remaining_addresses)}")
            geocode_result = geocoder.google(address, session=session)
            address_to_geocoding[address] = geocode_result.json

        with open(geocoding_filename, 'w') as f:
            json.dump(address_to_geocoding, f, indent=4)

if not os.path.exists(geocoding_filename):
    address_to_geocoding = {}
    geocode_addresses(address_to_geocoding)
else:
    with open(geocoding_filename) as f:
        address_to_geocoding = json.load(f)
    geocode_addresses(address_to_geocoding)
 
address_to_postal = {
    k: v['postal']
    for k, v in address_to_geocoding.items()
    if v is not None and 'postal' in v
}
address_to_latlng = {
    k: (v['lat'], v['lng'])
    for k, v in address_to_geocoding.items()
    if v is not None
}
print(f"Failed to get postal from {len(address_to_geocoding) - len(address_to_postal)}")
print(f"Failed to get latlng from {len(address_to_geocoding) - len(address_to_latlng)}")




def assign_postal_lat_lng(df):
    addresses = df['addr'].str.cat(df['city'], sep=', ')
    addresses_to_postal = [address_to_postal.get(a) for a in addresses]
    addresses_to_lat = [address_to_latlng[a][0] if a in address_to_latlng else None for a in addresses]
    addresses_to_lng = [address_to_latlng[a][1] if a in address_to_latlng else None for a in addresses]

    return df.assign(postal=addresses_to_postal, lat=addresses_to_lat, lng=addresses_to_lng)

df = assign_postal_lat_lng(df)
df.head(6)

import recordlinkage as rl
from recordlinkage.index import Full

full_indexer = Full()
pairs = full_indexer.index(df)

print(f"Full index: {len(df)} records, {len(pairs)} pairs")



from recordlinkage.index import Block

postal_indexer = Block('postal')
pairs = postal_indexer.index(df)

print(f"Postal index: {len(pairs)} pairs")

pairs.to_frame()[:10].values

pd.DataFrame([[0.5, 0.8, 0.9, 1]],
             columns=['name', 'addr', 'postal', 'latlng'],
             index=pd.MultiIndex.from_arrays([[100], [200]]))



comp = rl.Compare()
comp.string('name', 'name', method='jarowinkler', label='name')
comp.string('addr', 'addr', method='jarowinkler', label='addr')
comp.string('postal', 'postal', method='jarowinkler', label='postal')
comp.geo('lat', 'lng', 'lat', 'lng', method='exp', scale=0.1, offset=0.01, label='latlng');



comparison_vectors = comp.compute(pairs, df)
comparison_vectors.head(5)


scores = np.average(
    comparison_vectors.values,
    axis=1,
    weights=[50, 30, 10, 20])
scored_comparison_vectors = comparison_vectors.assign(score=scores)
scored_comparison_vectors.head(5)




df.head(5)



matches = scored_comparison_vectors[
    scored_comparison_vectors['score'] >= 0.9]
matches.head(5)



golden_pairs = Block('cluster').index(df_with_truth)
golden_pairs = golden_pairs.swaplevel().sortlevel()[0]
print("Golden pairs:", len(golden_pairs))



found_pairs_set = set(matches.index)

golden_pairs_set = set(golden_pairs)

true_positives = golden_pairs_set & found_pairs_set
false_positives = found_pairs_set - golden_pairs_set
false_negatives = golden_pairs_set - found_pairs_set

print('true_positives total:', len(true_positives))
print('false_positives total:', len(false_positives))
print('false_negatives total:', len(false_negatives))


print(f"False positives:")
for false_positive_pair in false_positives:
    print(df.loc[list(false_positive_pair)][['name', 'addr', 'postal', 'lat', 'lng']])
    
print(f"False negatives (sample 10 of {len(false_negatives)}):")
for false_negative_pair in list(false_negatives)[:10]:
    display(df.loc[list(false_negative_pair)][['name', 'addr', 'postal', 'lat', 'lng']])
    




df_training = pd.read_csv('D:/Downloads/Programs/Compressed/deduplication-slides-master/deduplication-slides-master/restaurant-training.csv', skip_blank_lines=True)
df_training = df_training.drop(columns=['type', 'phone'])
df_training



df_training = assign_no_symbols_name(df_training)
df_training = assign_postal_lat_lng(df_training)
df_training.head(5)




all_training_pairs = Full().index(df_training)
matches_training_pairs = Block('cluster').index(df_training)

training_vectors = comp.compute(all_training_pairs, df_training)

svm = rl.SVMClassifier()
svm.fit(training_vectors, matches_training_pairs);

svm_pairs = svm.predict(comparison_vectors)
svm_found_pairs_set = set(svm_pairs)

svm_true_positives = golden_pairs_set & svm_found_pairs_set
svm_false_positives = svm_found_pairs_set - golden_pairs_set
svm_false_negatives = golden_pairs_set - svm_found_pairs_set

print('true_positives total:', len(true_positives))
print('false_positives total:', len(false_positives))
print('false_negatives total:', len(false_negatives))
print()
print('svm_true_positives total:', len(svm_true_positives))
print('svm_false_positives total:', len(svm_false_positives))
print('svm_false_negatives total:', len(svm_false_negatives))

import logging; logging.disable(level=logging.NOTSET)



from svm_dedupe import SVMDedupe
import dedupe

fields = [
    {
        'field': 'name',
        'variable name': 'name',
        'type': 'JaroWinkler',
    },
    {
        'field': 'addr',
        'variable name': 'addr',
        'type': 'JaroWinkler',
    },
    {
        'field': 'postal',
        'variable name': 'postal',
        'type': 'JaroWinkler'
    },
    {
        'field': 'latlng',
        'variable name': 'latlng',
        'type': 'ExpLatLong'
    },
]

deduper = SVMDedupe(fields)







