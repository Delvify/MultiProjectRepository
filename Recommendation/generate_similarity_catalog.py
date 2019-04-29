import tensorflow as tf
import tensorflow_hub as hub
import json
import os
import numpy as np
from numpy import dot
from numpy.linalg import norm
from itertools import islice
import operator


fp = "Resources/floraDataSE.json"


with open(fp, "r") as f:
  data = json.load(f)


def cos_sim(a, b):
  return dot(a, b)/(norm(a)*norm(b))


datasize = len(data)


# get n number of items from the dictionary
def get_top5(mydict, n=5):
  return list(islice(mydict, n))


# sorting dictionary in descending order based on value, 0 for key
def sort_dict(mydict):
 return sorted(mydict.items(), key = operator.itemgetter(1), reverse=True)


sku_score = dict()
for i, item in enumerate(data):
 for j, itm in enumerate(data):
   if item.items() != itm.items():
     score = cos_sim(np.array(item['embeddings']) , np.array(itm['embeddings']))
     sku_score[itm['SKU']] = score
 sorted_sku = sort_dict(sku_score)
 item['sim_item'] = get_top5(sorted_sku)


output_fname = "Resources/floralSimilarSKU.json"
with open(output_fname, "w") as f:
 json.dump(data, f)