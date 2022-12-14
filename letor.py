import numpy as np
import lightgbm

from gensim.models import TfidfModel
from gensim.models import LsiModel
from gensim.corpora import Dictionary
from scipy.spatial.distance import cosine
import random

class Letor:

    def __init__(self):
        self.documents = {}
        self.queries = {}
        self.q_docs_rel = {}
        self.X = []
        self.Y = []
        self.group_qid_count = []

    def load_dataset(self):
        NUM_NEGATIVES = 1
        with open("nfcorpus/train.vid-desc.queries") as file:
            for line in file:
                q_id, content = line.split("\t")
                self.queries[q_id] = content.split()
        
        with open("nfcorpus/train.docs") as file:
            for line in file:
                doc_id, content = line.split("\t")
                self.documents[doc_id] = content.split()
        with open("nfcorpus/train.3-2-1.qrel") as file:
            for line in file:
                q_id, _, doc_id, rel = line.split("\t")
                if (q_id in self.queries) and (doc_id in self.documents):
                    if q_id not in self.q_docs_rel:
                        self.q_docs_rel[q_id] = []
                    self.q_docs_rel[q_id].append((doc_id, int(rel)))

            # group_qid_count untuk model LGBMRanker
            dataset = []
            for q_id in self.q_docs_rel:
                docs_rels = self.q_docs_rel[q_id]
                self.group_qid_count.append(len(docs_rels) + NUM_NEGATIVES)
                for doc_id, rel in docs_rels:
                    dataset.append((self.queries[q_id], self.documents[doc_id], rel))
                # tambahkan satu negative (random sampling saja dari documents)
                dataset.append((self.queries[q_id], random.choice(list(self.documents.values())), 0))

    def build_model(self):
        self.load_dataset()
        NUM_LATENT_TOPICS = 200

        dictionary = Dictionary()
        bow_corpus = [dictionary.doc2bow(doc, allow_update = True) for doc in self.documents.values()]
        model = LsiModel(bow_corpus, num_topics = NUM_LATENT_TOPICS) # 200 latent topics

        # test melihat representasi vector dari sebuah dokumen & query
        def vector_rep(text):
            rep = [topic_value for (_, topic_value) in model[dictionary.doc2bow(text)]]
            return rep if len(rep) == NUM_LATENT_TOPICS else [0.] * NUM_LATENT_TOPICS
        X = []
        Y = []
        def features(query, doc):
            v_q = vector_rep(query)
            v_d = vector_rep(doc)
            q = set(query)
            d = set(doc)
            cosine_dist = cosine(v_q, v_d)
            jaccard = len(q & d) / len(q | d)
            return v_q + v_d + [jaccard] + [cosine_dist]

        for (query, doc, rel) in self.dataset:
            X.append(features(query, doc))
            Y.append(rel)

        # ubah X dan Y ke format numpy array
        self.X = np.array(X)
        self.Y = np.array(Y)
    
    def train_and_predict(self):
        ranker = lightgbm.LGBMRanker(
                    objective="lambdarank",
                    boosting_type = "gbdt",
                    n_estimators = 100,
                    importance_type = "gain",
                    metric = "ndcg",
                    num_leaves = 40,
                    learning_rate = 0.02,
                    max_depth = -1)

        # di contoh kali ini, kita tidak menggunakan validation set
        # jika ada yang ingin menggunakan validation set, silakan saja
        ranker.fit(self.X, self.Y,
                group = self.group_qid_count,
                verbose = 10)
        return ranker

    
    def load(self, query, docs):
        self.build_model()
        ranker = self.train_and_predict()
        X_unseen = []
        for doc_id, doc in docs:
            X_unseen.append(self.features(query.split(), doc.split()))

            X_unseen = np.array(X_unseen)
        scores = ranker.predict(X_unseen)
        did_scores = [x for x in zip([did for (did, _) in docs], scores)]
        sorted_did_scores = sorted(did_scores, key = lambda tup: tup[1], reverse = True)

        print("query        :", query)
        print("SERP/Ranking :")
        
        for (did, score) in sorted_did_scores:
            print(did, score)

        def features(self, query, doc):
            v_q = self.vector_rep(query)
            v_d = self.vector_rep(doc)
            q = set(query)
            d = set(doc)
            cosine_dist = cosine(v_q, v_d)
            jaccard = len(q & d) / len(q | d)
            return v_q + v_d + [jaccard] + [cosine_dist]
