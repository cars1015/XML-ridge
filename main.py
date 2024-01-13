from scipy.sparse import csr_matrix
import numpy as np
from tabulate import tabulate
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import eval
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
    
    X_train = load_data(dir, 'X.trn.npz')
    X_test = load_data(dir, 'X.tst.npz')
    if args.c_flg==True:
        X_tr=np.load(dir+'X.trn.finetune.xlnet.npy')
        X_te=np.load(dir+'X.tst.finetune.xlnet.npy')
        X_train.toarray()
        X_test.toarray()
        X_train=csr_matrix(np.vstack(X_train,X_tr))
        X_test=csr_matrix(np.vstack(X_test,X_te))

    y_true = load_data(dir, 'Y.tst.npz')

    model = AE()
    model.fit(X_train, Y_train, lambda_=args.lambda_, flag=args.flag)
    y_pred = X_test.dot(model.W)
    acc = eval.Metrics(y_true, inv_psp=inv_propen,
                                remove_invalid=False)
    acc = acc.eval(y_pred, 5)
    display_metrics(acc)
if __name__ == '__main__':
    main()