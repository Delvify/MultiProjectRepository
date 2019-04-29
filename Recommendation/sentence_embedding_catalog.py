import tensorflow as tf
import tensorflow_hub as hub
import json
import os

fp = "/home/jugs/nlp/catalogs/resources/DataFeedInterfloraProductFeed.json"


with open(fp,"r") as f:
 data = json.load(f)


item_desc = []
for item in data:
 item_desc.append(item["Description"])


module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/3"  #args.module_url
embed = hub.Module(module_url, trainable=True)
tf.logging.set_verbosity(tf.logging.ERROR)


similarity_input_placeholder = tf.placeholder(tf.string, shape=(None))
similarity_message_encodings = embed(similarity_input_placeholder)


with tf.Session() as session:
  session.run(tf.global_variables_initializer())
  session.run(tf.tables_initializer())
  message_embedding = session.run(similarity_message_encodings, feed_dict={similarity_input_placeholder : item_desc})


for i, item in enumerate(data):
 item["embeddings"] = message_embedding[i].tolist()      # np array to list


# To save a file
with open("floraDataSE.json", "w") as f:
 json.dump(data, f)


# To Open a File
# with open("floraDataSE.json", "r") as f:
#  data = json.load(f)

