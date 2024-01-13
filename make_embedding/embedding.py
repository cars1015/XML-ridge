import numpy as np
import argparse
from gensim.models import KeyedVectors
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import save_npz
from tqdm import tqdm

class TextEmbedder:
    def __init__(self, w2v_path):
        self.w2v_model = KeyedVectors.load(w2v_path)
        self.tfidf_vectorizer = TfidfVectorizer(analyzer='word', max_df=0.98, stop_words='english')

    def compute_tfidf(self, documents):
        return self.tfidf_vectorizer.fit_transform(documents)

    def transform_tfidf(self, documents):
        return self.tfidf_vectorizer.transform(documents)

    def compute_document_embeddings(self, documents, tfidf_matrix):
        document_embeddings = []
        analyzer = self.tfidf_vectorizer.build_analyzer()
        for doc_index, doc in tqdm(enumerate(documents), desc="Computing document embeddings", total=len(documents)):
            tokens = set(analyzer(doc))
            doc_embedding = np.zeros(self.w2v_model.vector_size, dtype=np.float32)
            for token in tokens:
                if token in self.w2v_model and token in self.tfidf_vectorizer.vocabulary_:
                    token_index = self.tfidf_vectorizer.vocabulary_[token]
                    tfidf_value = tfidf_matrix[doc_index, token_index]
                    doc_embedding += tfidf_value * self.w2v_model[token]
            document_embeddings.append(doc_embedding)
        return np.array(document_embeddings)

def read_corpus(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.readlines()

def main(args,data_dir):
    corpus_tr_path = f'{data_dir}{args.data}/{args.train_path}'
    corpus_te_path = f'{data_dir}{args.data}/{args.test_path}'

    embedder = TextEmbedder(args.w2v_model)

    # For training data
    documents_tr = read_corpus(corpus_tr_path)
    tfidf_matrix_tr = embedder.compute_tfidf(documents_tr)
    document_embeddings_tr = embedder.compute_document_embeddings(documents_tr, tfidf_matrix_tr)
    print("Training TF-IDF matrix shape:", tfidf_matrix_tr.shape)
    save_npz(f'{data_dir}{args.data}/X.trn_My.npz', tfidf_matrix_tr)
    np.save(f'{data_dir}{args.data}/X_.bag_trn.npy', document_embeddings_tr)

    # For test data
    documents_te = read_corpus(corpus_te_path)
    tfidf_matrix_te = embedder.transform_tfidf(documents_te)
    document_embeddings_te = embedder.compute_document_embeddings(documents_te, tfidf_matrix_te)
    print("Test TF-IDF matrix shape:", tfidf_matrix_te.shape)
    save_npz(f'{data_dir}{args.data}/X.tst_My.npz', tfidf_matrix_te)
    np.save(f'{data_dir}{args.data}/X.bag_tst.npy', document_embeddings_te)

    print('done')

if __name__ == "__main__":
    data_dir = './data_dir/'
    parser = argparse.ArgumentParser(description='Text Embedding Processor')
    parser.add_argument('--data', type=str, help='Path of data')
    parser.add_argument('--train-path', type=str, help='Path of train data')
    parser.add_argument('--test-path', type=str, help='Path of test data')
    parser.add_argument('--w2v-model', type=str, help='Path of Word2Vec model')
    args = parser.parse_args()
    main(args)


"""
scripts for Eurlex-4K
python3 embedding.py --data Eurlex-4K --train-path train_raw_texts.txt --test-path test_raw_texts.txt --w2v-model /home/hayashi/categorize/AttentionXML/data/glove.840B.300d.gensim
"""

