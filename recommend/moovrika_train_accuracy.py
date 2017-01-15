from lightfm.evaluation import auc_score
import datetime
import pickle
datetime.datetime.now()
import moovrika_core
movielens = moovrika_core.fetch_dataset('./')
movielens_new = moovrika_core.fetch_dataset('./online_training/')
print(movielens['train'].shape)
from lightfm import LightFM
alpha = 1e-05
epochs = 500
model = LightFM(loss='warp', learning_schedule='adagrad',
                    user_alpha=alpha,
                    item_alpha=alpha)

for epoch in range(epochs):
    model.fit_partial(movielens['train'], epochs=1)
    print('epoch', epoch)
print(auc_score(model, movielens['test'], train_interactions=movielens['train']).mean())
moovrika_core.sample_recommendation(model, movielens, [1])
moovrika_core.sample_recommendation(model, movielens, [0])
pickle.dump(model, open("model.p", "wb"))
pickle.dump(movielens, open("movielens.p", "wb"))

for epoch in range(epochs):
    model.fit_partial(movielens_new['train'], epochs=1)
    print('epoch', epoch)
    if epoch%50 == 0:
        moovrika_core.sample_recommendation(model, movielens, [0])

moovrika_core.sample_recommendation(model, movielens, [1])
moovrika_core.sample_recommendation(model, movielens, [0])

datetime.datetime.now()
print(auc_score(model, movielens['test'], train_interactions=movielens['train']).mean())
import pickle
pickle.dump(model, open("model.p", "wb"))
pickle.dump(movielens, open("movielens.p", "wb"))
#model = pickle.load(open("model.p", "rb"))

