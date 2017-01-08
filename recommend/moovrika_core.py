import itertools
import numpy as np
import scipy.sparse as sp


def _read_raw_data(path):


        return (open(path + '/dataset_ratings_train.csv').read().splitlines(),
                open(path + '/dataset_ratings_test.csv').read().splitlines(),
                open(path + '/dataset_movies.csv').read().splitlines())


def _parse(data):


  for line in data:


    if not line:
      continue


    uid, iid, rating = [x for x in line.split('|')]


    # Subtract one from ids to shift
    # to zero-based indexing
    yield int(uid) - 1, int(iid) - 1, rating


def _get_dimensions(train_data, test_data):


  uids = set()
  iids = set()


  for uid, iid, _ in itertools.chain(train_data, test_data):
    uids.add(uid)
    iids.add(iid)


  rows = max(uids) + 1
  cols = max(iids) + 1


  return rows, cols


def _build_interaction_matrix(rows, cols, data):


  mat = sp.lil_matrix((rows, cols), dtype=np.float32)


  for uid, iid, rating in data:
    mat[uid, iid] = rating


  return mat.tocoo()


def _parse_item_metadata(num_items, item_metadata_raw):


  id_feature_labels = np.empty(num_items, dtype=np.object)


  id_features = sp.identity(num_items, format='csr', dtype=np.float32)


  for line in item_metadata_raw:


    if not line:
      continue


    splt = line.split('|')


    # Zero-based indexing
    #print(splt[0])
    iid = int(splt[0]) - 1
    title = splt[1]


    id_feature_labels[iid] = title


  return (id_features, id_feature_labels)


def fetch_dataset(data_dir=None):

  # Load raw data
  (train_raw, test_raw, item_metadata_raw) = _read_raw_data(data_dir)


  # Figure out the dimensions
  num_users, num_items = _get_dimensions(_parse(train_raw), _parse(test_raw))


  # Load train interactions
  train = _build_interaction_matrix(num_users, num_items, _parse(train_raw))


  # Load test interactions
  test = _build_interaction_matrix(num_users, num_items, _parse(test_raw))


  # Load metadata features
  (id_features, id_feature_labels) = _parse_item_metadata(num_items, item_metadata_raw)


  data = {'train': train, 'test': test, 'item_labels': id_feature_labels}


  return data


def sample_recommendation(model, data, user_ids):


  n_users, n_items = data['train'].shape


  for user_id in user_ids:
    known_positives = data['item_labels'][data['train'].tocsr()[user_id].indices]


    scores = model.predict(user_id, np.arange(n_items))
    top_items = data['item_labels'][np.argsort(-scores)]


    print("User %s" % user_id)
    #print("   Known positives:")


    #for x in known_positives[:20]:
    #  print("      %s" % x)


    print("   Recommended:")


    for x in top_items[:20]:
      print("        %s" % x)






''' This is to see how AUC is changing increasing training iterations
import datetime
datetime.datetime.now()
from moovrika import fetch_dataset_v2
movielens = fetch_dataset_v2.fetch_dataset('./moovrika/movies_after_2005')
from lightfm import LightFM
from lightfm.evaluation import auc_score
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
alpha = 1e-05
epochs = 3000
num_components = 32
#                    max_sampled=3,
model = LightFM(no_components=num_components,
                    loss='warp',
                    learning_schedule='adagrad',
                    user_alpha=alpha,
                    item_alpha=alpha)
warp_auc = []
for epoch in range(epochs):
    model.fit_partial(movielens['train'], epochs=1)
    curr_auc = auc_score(model, movielens['test'], train_interactions=movielens['train']).mean()
    print('epoch', epoch, ':', curr_auc)
    warp_auc.append(curr_auc)


x = np.arange(epochs)
plt.plot(x, np.array(warp_auc))
plt.legend(['WARP AUC'], loc='upper right')
plt.show()
'''


''' This is to generate recommendations for user [0]
import datetime
datetime.datetime.now()
from moovrika import fetch_dataset_v2
movielens = fetch_dataset_v2.fetch_dataset('./moovrika/movies_after_2005')
from lightfm import LightFM
alpha = 1e-05
epochs = 1000
model = LightFM(loss='warp', learning_schedule='adagrad',
                    user_alpha=alpha,
                    item_alpha=alpha)
model.fit(movielens['train'], epochs=epochs, num_threads=2)
fetch_dataset_v2.sample_recommendation(model, movielens, [0])
datetime.datetime.now()
from lightfm.evaluation import auc_score
print(auc_score(model, movielens['test'], train_interactions=movielens['train']).mean())
'''

