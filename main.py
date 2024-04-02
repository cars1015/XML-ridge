from scipy.sparse import csr_matrix
import numpy as np
from tabulate import tabulate
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import eval
import argparse
def display_metrics(metrics, k=5):
    final_metrics=metrics
    # Dataset metrics.
    print("----------Tests with Ordered Retrieval------------")
    table = [['Precision@k'] + [i * 100  for i in final_metrics[0]]]
    table.append(['nDCG@k'] + [i * 100  for i in final_metrics[1]])
    table.append(['PSprec@k'] + [i * 100  for i in final_metrics[2]])
    table.append(['PSnDCG@k'] + [i * 100  for i in final_metrics[3]])
    print(tabulate(table, headers=[i+1 for i in range(0, k)],
                   floatfmt=".3f"))

class AE:
    def fit(self,X,Y,lambda_,flag=True):
        if flag==True:
            G=X.T.dot(X).toarray()
            diagIndices = np.diag_indices(G.shape[0])
            G[diagIndices] += lambda_
            self.Y=Y
            P=np.linalg.inv(G)
            W=P@X.T@Y
            self.W=W
        else :
            G=X.dot(X.T).toarray()
            print(G.shape)
            diagIndices = np.diag_indices(G.shape[0])
            G[diagIndices] += lambda_
            print(G.shape)
            self.Y=Y
            P=np.linalg.inv(G)
            W=X.T@P@Y
            self.W=W
def load_data(directory, file_name):
    data = np.load(directory + file_name)
    return csr_matrix((data['data'], data['indices'], data['indptr']), shape=data['shape'])

def main():
    parser = argparse.ArgumentParser(description='Run AE model with custom parameters.')
    parser.add_argument('--data', type=str, required=True, help='Dataset name')
    parser.add_argument('--lambda_', type=float, default=0.1, help='Lambda value')
    parser.add_argument('--A', type=float, default=0.55, help='A value for inv propensity')
    parser.add_argument('--B', type=float, default=1.5, help='B value for inv propensity')
    parser.add_argument('--w_flg', action='store_true', help='Weight flag')
    parser.add_argument('--c_flg',action='store_true',help='Concat flg')
    parser.add_argument('--flag', action='store_true', help='Flag for fit function')
    args = parser.parse_args()

    dir = f"./data_dir/{args.data}/"

    Y_train = load_data(dir, 'Y.trn.npz')
    if args.w_flg:
        inv_propen = eval.compute_inv_propensity(Y_train, args.A, args.B)
        Y_train = Y_train.multiply(inv_propen)
    
    if args.data == "delicious200k":
        X_train=np.load(dir+'X.trn.npy')
        X_test=np.load(dir+'X.tst.npy')
    else:
        X_train = load_data(dir, 'X.trn.npz')
        X_test = load_data(dir, 'X.tst.npz')
    if args.c_flg==True:
        if args.data == "wiki10-31k":
            X_tr=np.load(dir+'X.bert.trn.npy')
            X_te=np.load(dir+'X.bert.tst.npy')
        else:
            X_1=np.load(dir+'X.bert.trn.npy')
            X_2=np.load(dir+'X.roberta.trn.npy')
            X_3=np.load(dir+'X.xlnet.trn.npy')
            X_tr=
            X_1=np.load(dir+'X.bert.tst.npy')
            X_2=np.load(dir+'X.roberta.tst.npy')
            X_3=np.load(dir+'X.xlnet.tst.npy')

        if scipy.sparse.isspmatrix_csr(X):
            X_train.toarray()
            X_test.toarray()
        X_train=csr_matrix(np.vstack(X_train,X_tr))
        X_test=csr_matrix(np.vstack(X_test,X_te))

    y_true = load_data(dir, 'Y.tst.npz')

    model = AE()
    model.fit(X_train, Y_train, lambda_=args.lambda_, flag=args.flag)
    if args.data == "amazoncat13k" or args.data == "delicious200K":
        N_test=X_train.shape[0]
        batch_size=30000
        W=model.W
        acc_sum = None
        for st_idx in range(0, N_test, batch_size):
            end_idx = min(st_idx + batch_size, N_test)
            print(end_idx)
            y_pred = X[st_idx:end_idx] @ W
            y_true_b = y_true[st_idx:end_idx, :]
            acc = eval.Metrics(y_true_b, inv_psp=inv_propen, remove_invalid=False)
            acc = acc.eval(y_pred, 5)
            acc_weighted = [x * (end_idx - st_idx) for x in acc]
            if acc_sum is None:
                acc_sum = acc_weighted
            else:
                acc_sum = [sum(x) for x in zip(acc_sum, acc_weighted)]
        # normalize
        acc_normalized = [x / N_test for x in acc_sum]
        display_metrics(acc_normalized)    
    else:        
        y_pred = X_test.dot(model.W)
        acc = eval.Metrics(y_true, inv_psp=inv_propen,
                                    remove_invalid=False)
        acc = acc.eval(y_pred, 5)
        display_metrics(acc)
if __name__ == '__main__':
    main()