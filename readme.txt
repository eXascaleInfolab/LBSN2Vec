This is the implementation for LBSN2Vec in MATLAB (see the following paper)

- Dingqi Yang, Bingqing Qu, Jie Yang, Philippe Cudre-Mauroux, "Revisiting User Mobility and Social Relationships in LBSNs: A Hypergraph Embedding Approach." In WWW'19

How to use (Tested on 2017b):
1. Compile learn_LBSN2Vec_embedding.c using mex: 
mex CFLAGS='$CFLAGS -pthread -Ofast -march=native -Wall -funroll-loops -Wno-unused-result' learn_LBSN2Vec_embedding.c

2. Run experiment_LBSN2Vec.m

Please cite our paper if you use this code.
