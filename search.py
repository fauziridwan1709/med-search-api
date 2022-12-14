from bsbi import BSBIIndex
from letor import Letor
from compression import VBEPostings

# sebelumnya sudah dilakukan indexing
# BSBIIndex hanya sebagai abstraksi untuk index tersebut
BSBI_instance = BSBIIndex(data_dir = 'collection', \
                          postings_encoding = VBEPostings, \
                          output_dir = 'index')

LETOR_instance = Letor()

queries = ["alkylated with radioactive iodoacetate"]
for query in queries:
    print("Query  : ", query)
    tfidf = BSBI_instance.retrieve_tfidf(query, k = 100)
    bm25 = BSBI_instance.retrieve_bm25(query, k = 100)
    print("Results TF-IDF:")
    for (score, doc) in tfidf:
        print(f"{doc:30} {score:>.3f}")
    print()
    print("Results BM25:")
    for (score, doc) in bm25:
        print(f"{doc:30} {score:>.3f}")
    print()