import datetime
datetime.datetime.now()
import moovrika_core
movielens = moovrika_core.fetch_dataset('./')
from lightfm import LightFM
alpha = 1e-05
epochs = 500
model = LightFM(loss='warp', learning_schedule='adagrad',
                    user_alpha=alpha,
                    item_alpha=alpha)
model.fit(movielens['train'], epochs=epochs, num_threads=8)
moovrika_core.sample_recommendation(model, movielens, [1])
datetime.datetime.now()
from lightfm.evaluation import auc_score
print(auc_score(model, movielens['test'], train_interactions=movielens['train']).mean())
import pickle
pickle.dump(model, open("model.p", "wb"))
pickle.dump(movielens, open("movielens.p", "wb"))
#model = pickle.load(open("model.p", "rb"))
