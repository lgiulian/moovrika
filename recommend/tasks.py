from kuyruk import Kuyruk
import subprocess
import sys
sys.path.insert(0, '../')
import numpy as np
import scipy.sparse as sp
import pickle
import mysql.connector
import moovrika_core

kuyruk = Kuyruk()

@kuyruk.task
def echo(message):
    print(message)

@kuyruk.task
def train_by_request_id(request_id):

  if not request_id.isnumeric():
    raise NameError('Request ID is not a number')

  con = mysql.connector.connect(user='recommendation', password='2yNnQSxpAJG!uPTe', host='localhost',database='recommend')
  cursor = con.cursor(buffered=True)

  try:
    query = ("select u.idx, m.idx, rat.rating "
      "from recommend.recom_requests req "
      "inner join recommend.users u on req.iduser = u.id "
      "inner join recommend.rates rat on req.iduser = rat.userid "
      "inner join recommend.movies m on rat.movieid = m.id "
      "where req.idrequest = %(request_id)s")

    cursor.execute(query, {'request_id':request_id})

    movielens = pickle.load(open("../movielens.p", "rb"))
    model = pickle.load(open("../model.p", "rb"))
    n_users = movielens['train'].shape[0]
    n_items = movielens['train'].shape[1]
    mat = sp.lil_matrix((n_users, n_items), dtype=np.float32)

    for (userindex, movieindex, rating) in cursor:
      mat[userindex-1, movieindex-1] = rating
      print("{}, {}, {}".format(
        userindex-1, movieindex-1, rating))

    train_data = mat.tocoo()
    epochs = 50
    for epoch in range(epochs):
      model.fit_partial(train_data, epochs=1)
      print('epoch', epoch)

    scores = model.predict(userindex, np.arange(n_items))
    top_items = movielens['item_labels'][np.argsort(-scores)]
    order = 1;
    for x in top_items[:20]:
      print("        %s" % x)
      cursor.execute("INSERT INTO recommend.recom_result values (%(idrequest)s, %(identity)s, %(order)s, now())", {'idrequest':request_id, 'identity':x, 'order':order})
      order = order + 1
    con.commit()

    pickle.dump(model, open("../model.p", "wb"))
  finally:
    try:
      cursor.close()
    finally:
      print("")
    con.close()

def findProcess():
  ps= subprocess.Popen("ps -ef | grep kuyruk", shell=True, stdout=subprocess.PIPE)
  output = ps.stdout.read()
  ps.stdout.close()
  ps.wait()
  return output
def isProcessRunning():
  output = findProcess()
  if re.search(processId, output) is None:
    return true
  else:
    return False


