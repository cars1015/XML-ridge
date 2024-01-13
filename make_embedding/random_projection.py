from sklearn.random_projection import SparseRandomProjection
from scipy.sparse import csr_matrix
import numpy as np
from scipy.sparse import csr_matrix

def projection(save_dir):
    tr_path=save_dir+'/X.trn.npz'
    te_path=save_dir+'/X.tst.npz'
    data=np.load(tr_path)
    tfidf_tr=csr_matrix((data['data'], data['indices'], data['indptr']), shape=data['shape'])
    data=np.load(te_path)
    tfidf_te=csr_matrix((data['data'], data['indices'], data['indptr']), shape=data['shape'])

    transformer=SparseRandomProjection()
    matrix_tr=transformer.fit_transform(tfidf_tr)
    matrix_te = transformer.transform(tfidf_te)
    np.save(save_dir+'/X.trn_projection.npy',matrix_tr)
    np.save(save_dir+'/X.tst_projection.npy',matrix_te)
    

data='Wiki10-31K'
#data='Eurlex-4K'
#data='AmazonCat-13K'
#data='Wiki-500K'
#data_dir='./data_dir/'+data
projection(data_dir)