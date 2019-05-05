//mex CFLAGS='$CFLAGS -pthread -Ofast -march=native -Wall -funroll-loops -Wno-unused-result' learn_LBSN2Vec_embedding.c

#include "mex.h"
#include "stdio.h"
#include "stdlib.h"
#include "math.h"
#include "pthread.h"
#include "limits.h"
#include "string.h"

#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6
#define RAND_MULTIPLIER 25214903917
#define RAND_INCREMENT 11

double *expTable;

// input 1
long long *walk;
long long num_w;
long long num_wl;
// input 2
const mxArray *user_checkins; // hyperedges
long long num_u;
// input 3
long long *user_checkins_count;
// input 4
double *emb_n; //node embedding
long long num_n;
long long dim_emb;
// input 5
double starting_alpha;
double alpha;
// input 6
double num_neg;
// input 7
long long *neg_sam_table_social; // negative sampling table social network
long long table_size_social;
// input 8
long long win_size;
// input 9
const mxArray *neg_sam_table_mobility; // negative sampling table checkins
long long table_num_mobility;
long long *neg_sam_table_mobility1;
long long table_size_mobility1;
long long *neg_sam_table_mobility2;
long long table_size_mobility2;
long long *neg_sam_table_mobility3;
long long table_size_mobility3;
long long *neg_sam_table_mobility4;
long long table_size_mobility4;
// input 10
long long num_epoch;
// input 11
long long num_threads;
// input 12
double mobility_ratio;
// double *counter;
// double *alpha_Katz_Table;
// unsigned long next_random_max=0;

const mxArray *temp;

void getNextRand(unsigned long *next_random){
    *next_random = (*next_random) * (unsigned long) RAND_MULTIPLIER + RAND_INCREMENT;
}

long long get_a_neg_sample(unsigned long next_random, long long *neg_sam_table, long long table_size){
    long long target_n;
    unsigned long long ind;

    ind = (next_random >> 16) % table_size;
    target_n = neg_sam_table[ind];

    return target_n;
}

long long get_a_checkin_sample(unsigned long next_random, long long  table_size){
    return (next_random >> 16) % table_size;
}


double sigmoid(double f) {
    if (f >= MAX_EXP) return 1;
    else if (f <= -MAX_EXP) return 0;
    else return expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2 ))];
}

int get_a_neg_sample_Kless1(unsigned long next_random){
    double v_rand_uniform = (double) next_random/(double)(ULONG_MAX);
    if (v_rand_uniform<=num_neg){
        return 1;
    }else{
        return 0;
    }
}

int get_a_social_decision(unsigned long next_random){
    double v_rand_uniform = (double) next_random/(double)(ULONG_MAX);
    if (v_rand_uniform<=mobility_ratio){
        return 0;
    }else{
        return 1;
    }
}

int get_a_mobility_decision(unsigned long next_random){
    double v_rand_uniform = (double) next_random/(double)(ULONG_MAX);
    if (v_rand_uniform<=mobility_ratio){
        return 1;
    }else{
        return 0;
    }
}

double get_norm_l2_loc(long long loc_node){
    double norm = 0;
    for (int d=0; d<dim_emb; d++) norm = norm + emb_n[loc_node+d] * emb_n[loc_node+d];
    return sqrt(norm);
}

double get_norm_l2_pr(double *vec){
    double norm = 0;
    for (int d=0; d<dim_emb; d++) norm = norm + vec[d] * vec[d];
    return sqrt(norm);
}

void learn_a_pair_loc_loc_cosine(int flag, long long loc1, long long loc2, double *loss)
{
    double f=0,tmp1,tmp2,c1,c2,c3; //f2=0,
    double norm1 = get_norm_l2_loc(loc1);
    double norm2 = get_norm_l2_loc(loc2);

    for (int d=0;d<dim_emb;d++)
        f += emb_n[loc1+d] * emb_n[loc2+d];

    c1 = 1/(norm1*norm2)*alpha;
    c2 = f/(norm1*norm1*norm1*norm2)*alpha;
    c3 = f/(norm1*norm2*norm2*norm2)*alpha;


    if (flag==1){
//         *loss += f;
        for (int d=0; d<dim_emb; d++){
            tmp1 = emb_n[loc1 + d];
            tmp2 = emb_n[loc2 + d];
            emb_n[loc2 + d] += c1*tmp1 - c3*tmp2;
            emb_n[loc1 + d] += c1*tmp2 - c2*tmp1;
        }
    }else{
//         *loss -= f/num_neg;
        for (int d=0; d<dim_emb; d++){
            tmp1 = emb_n[loc1 + d];
            tmp2 = emb_n[loc2 + d];
            emb_n[loc2 + d] -= c1*tmp1 - c3*tmp2;
            emb_n[loc1 + d] -= c1*tmp2 - c2*tmp1;
        }
    }

}

void learn_a_pair_loc_pr_cosine(int flag, long long loc1, double *best_fit, double *loss)
{
    double f=0,g=0,a=0,c1,c2; //f2=0,
    double norm1 = get_norm_l2_loc(loc1);

    for (int d=0;d<dim_emb;d++)
        f += emb_n[loc1+d] * best_fit[d];

    g = f/norm1;

    a = alpha;
    c1 = 1/(norm1)*a;
    c2 = f/(norm1*norm1*norm1)*a;

    if (flag==1){
//         *loss += g;
        for (int d=0; d<dim_emb; d++)
            emb_n[loc1 + d] += c1*best_fit[d] - c2*emb_n[loc1 + d];
    }else{
//         *loss -= g/num_neg;
        for (int d=0; d<dim_emb; d++)
            emb_n[loc1 + d] -= c1*best_fit[d] - c2*emb_n[loc1 + d];
    }
}

void learn_an_edge(long long word, long long target_e, unsigned long *next_random, double* counter)
{
    long long target_n, loc_neg;
    long long loc_w = (word-1)*dim_emb;
    long long loc_e = (target_e-1)*dim_emb;
    learn_a_pair_loc_loc_cosine(1, loc_w, loc_e, counter);

    if (num_neg<1){
        getNextRand(next_random);
        if (get_a_neg_sample_Kless1(*next_random)==1){
            getNextRand(next_random);
            target_n = get_a_neg_sample(*next_random, neg_sam_table_social, table_size_social);
            if ((target_n != target_e) && (target_n != word)){
                loc_neg = (target_n-1)*dim_emb;
                learn_a_pair_loc_loc_cosine(0, loc_w, loc_neg, counter);
            }
        }
    }else{
        for (int n=0;n<num_neg;n++){
            getNextRand(next_random);
            target_n = get_a_neg_sample(*next_random, neg_sam_table_social, table_size_social);
            if ((target_n != target_e) && (target_n != word)){
                loc_neg = (target_n-1)*dim_emb;
                learn_a_pair_loc_loc_cosine(0, loc_w, loc_neg, counter);
            }
        }
    }
}


void learn_an_edge_with_BFT(long long word, long long target_e, unsigned long *next_random, double *best_fit, double* counter)
{
    long long target_n, loc_neg;
    double norm;
    long long loc_w = (word-1)*dim_emb;
    long long loc_e = (target_e-1)*dim_emb;

    for (int d=0; d<dim_emb; d++) best_fit[d] = emb_n[loc_w+d] + emb_n[loc_e+d];
    norm = get_norm_l2_pr(best_fit);
    for (int d=0; d<dim_emb; d++) best_fit[d] = best_fit[d]/norm;

    learn_a_pair_loc_pr_cosine(1, loc_w, best_fit, counter);
    learn_a_pair_loc_pr_cosine(1, loc_e, best_fit, counter);

    if (num_neg<1){
        getNextRand(next_random);
        if (get_a_neg_sample_Kless1(*next_random)==1){
            getNextRand(next_random);
            target_n = get_a_neg_sample(*next_random, neg_sam_table_social, table_size_social);
            if ((target_n != target_e) && (target_n != word)){
                loc_neg = (target_n-1)*dim_emb;
                learn_a_pair_loc_pr_cosine(0, loc_neg, best_fit, counter);
            }
        }
    }else{
        for (int n=0;n<num_neg;n++){
            getNextRand(next_random);
            target_n = get_a_neg_sample(*next_random, neg_sam_table_social, table_size_social);
            if ((target_n != target_e) && (target_n != word)){
                loc_neg = (target_n-1)*dim_emb;
                learn_a_pair_loc_pr_cosine(0, loc_neg, best_fit, counter);
            }
        }
    }
}



void learn_a_hyperedge(long long *edge, long long edge_len, unsigned long *next_random, double *best_fit, double* counter)
{
    long long node, target_neg;
    long long loc_n, loc_neg;
    double norm;

//#################### get best-fit-line
    for (int d=0; d<dim_emb; d++) best_fit[d] = 0;
    for (int i=0; i<edge_len; i++) {
        loc_n = (edge[i]-1)*dim_emb;
        norm = get_norm_l2_pr(&emb_n[loc_n]);
        for (int d=0; d<dim_emb; d++) best_fit[d] += emb_n[loc_n + d]/norm;
    }
//  normalize best fit line for fast computation
    norm = get_norm_l2_pr(best_fit);
    for (int d=0; d<dim_emb; d++) best_fit[d] = best_fit[d]/norm;


//#################### learn learn learn
    for (int i=0; i<edge_len; i++) {
        node = edge[i];
        loc_n = (node-1)*dim_emb;
        learn_a_pair_loc_pr_cosine(1, loc_n, best_fit, counter);

        if (num_neg<1){
            getNextRand(next_random);
            if (get_a_neg_sample_Kless1(*next_random)==1){
                getNextRand(next_random);
                if (i==0) target_neg = get_a_neg_sample(*next_random, neg_sam_table_mobility1, table_size_mobility1);
                else if (i==1) target_neg = get_a_neg_sample(*next_random, neg_sam_table_mobility2, table_size_mobility2);
                else if (i==2) target_neg = get_a_neg_sample(*next_random, neg_sam_table_mobility3, table_size_mobility3);
                else if (i==3) target_neg = get_a_neg_sample(*next_random, neg_sam_table_mobility4, table_size_mobility4);

                if (target_neg != node) {
                    loc_neg = (target_neg-1)*dim_emb;
                    learn_a_pair_loc_pr_cosine(0, loc_neg, best_fit, counter);
                }
            }
        }else{
            for (int n=0;n<num_neg;n++){
                getNextRand(next_random);
                if (i==0) target_neg = get_a_neg_sample(*next_random, neg_sam_table_mobility1, table_size_mobility1);
                else if (i==1) target_neg = get_a_neg_sample(*next_random, neg_sam_table_mobility2, table_size_mobility2);
                else if (i==2) target_neg = get_a_neg_sample(*next_random, neg_sam_table_mobility3, table_size_mobility3);
                else if (i==3) target_neg = get_a_neg_sample(*next_random, neg_sam_table_mobility4, table_size_mobility4);

                if (target_neg != node) {
                    loc_neg = (target_neg-1)*dim_emb;
                    learn_a_pair_loc_pr_cosine(0, loc_neg, best_fit, counter);
                }
            }
        }
    }
}


void merge_hyperedges(long long *edge_merged, long long* edge_merged_len, long long *a_edge, long long a_edge_len)
{
    memcpy(edge_merged+(*edge_merged_len), a_edge, a_edge_len * sizeof(long long));
    *edge_merged_len += a_edge_len;
}



void normalize_embeddings(){
    long long loc_node;
    double norm;
    int i,d;
    for (i=0;i<num_n;i++) {
        loc_node = i*dim_emb;
        norm=0;
        for (d=0; d<dim_emb; d++) norm = norm + emb_n[loc_node+d] * emb_n[loc_node+d];
        for (d=0; d<dim_emb; d++) emb_n[loc_node+d] = emb_n[loc_node+d]/sqrt(norm);
    }
}


void learn(void *id)
{
    long long word, target_e, a_checkin_ind, a_checkin_loc;
    double *best_fit = (double *)mxMalloc(dim_emb*sizeof(double)); //a node embedding

    double counter;
//     double norm;

    unsigned long next_random = (long) rand();
    const mxArray *user_pr;
    long long *a_user_checkins;
    long long *edge;
    long long edge_len = 4; // here 4 is a checkin node number user-time-POI-category



    long long ind_start = num_w/num_threads * (long long)id;
    long long ind_end = num_w/num_threads * ((long long)id+1);

    long long ind_len = ind_end-ind_start;
    double progress=0,progress_old=0;
    alpha = starting_alpha;

    long long loc_walk;
//     mexPrintf("Thread %lld starts from hyperedges %lld to %lld\n",(long long)id,ind_start,ind_end);

    for (int pp=0; pp<num_epoch; pp++){
        counter = 0;

        for (int w=ind_start; w<ind_end; w++) {
            progress = ((pp*ind_len)+(w-ind_start)) / (double) (ind_len*num_epoch);
            if (progress-progress_old > 0.001) {
                alpha = starting_alpha * (1 - progress);
                if (alpha < starting_alpha * 0.001) alpha = starting_alpha * 0.001;
                progress_old = progress;
//                 if( (long long) id == 0) {
//                     mexPrintf("current alpha is: %f; Progress %.0f%%\n", alpha, progress*100);
// //                     shownorm();
//                 }
            }

            loc_walk = w*num_wl;
            for (int i=0; i<num_wl; i++) {
                word = walk[loc_walk+i];

                for (int j=1;j<=win_size;j++){
                    getNextRand(&next_random);
                    if (get_a_social_decision(next_random)==1){
//                         printf("social \n");
                        if (i-j>=0) {
                            target_e = walk[loc_walk+i-j];
                            if (word!=target_e)
                                learn_an_edge_with_BFT(word, target_e, &next_random, best_fit, &counter);
//                                 learn_an_edge(word, target_e, &next_random, &counter);
                        }
                        if (i+j<num_wl) {
                            target_e = walk[loc_walk+i+j];
                            if (word!=target_e)
                                learn_an_edge_with_BFT(word, target_e, &next_random, best_fit, &counter);
//                                 learn_an_edge(word, target_e, &next_random, &counter);
                        }
                    }
//                     printf("user %d has %d checkins.\n",word,user_checkins_count[word-1]);



                }

                if ((user_checkins_count[word-1]>0) ){
                    for (int m=0; m < fmin(win_size*2,user_checkins_count[word-1]); m++){
                        getNextRand(&next_random);
                        if (get_a_mobility_decision(next_random)==1) {
//                             printf("mobility \n");
                            user_pr = mxGetCell(user_checkins, word-1);
                            a_user_checkins = (long long *)mxGetData(user_pr);

                            getNextRand(&next_random);
                            a_checkin_ind = get_a_checkin_sample(next_random, user_checkins_count[word-1]);
//                         printf("sampled checkin index is %d\n",a_checkin_ind);
                            a_checkin_loc = a_checkin_ind*edge_len;
                            edge = &a_user_checkins[a_checkin_loc];
//                         printf("sampled checkin is %d-%d-%d-%d\n",edge[0],edge[1],edge[2],edge[3]);
//                         if (a_checkin_ind > mxGetN(user_pr))
//                             printf("ERROR: sampled checkin index is %d with %d!=%d\n",a_checkin_ind,mxGetN(user_pr),user_checkins_count[word-1]);
//
//                         if (word != edge[0])
//                             printf("ERROR: user %d is not user %d!=%d\n",word,edge[0]);

                            learn_a_hyperedge(edge, edge_len, &next_random, best_fit, &counter);
                        }
                    }
                }
            }

        }
//         printf("Thread %lld iteration %d loss: %f \n",(long long)id, pp, counter);
    }
//     printf("counter (word=target_e) : %lld\n", counter);
    mxFree(best_fit);
    pthread_exit(NULL);
}



void mexFunction(int nlhs, mxArray *plhs[],
        int nrhs, const mxArray *prhs[])
{
    if(nrhs != 12) {
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:nrhs",
                "12 inputs required.");
    }
    if(nlhs != 1) {
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:nlhs",
                "1 output required.");
    }

    walk = (long long *)mxGetData(prhs[0]); // read from file
    num_w = mxGetN(prhs[0]);
    num_wl = mxGetM(prhs[0]);

    user_checkins = prhs[1]; // user checkins cell
    num_u = mxGetNumberOfElements(prhs[1]);

    user_checkins_count = (long long *)mxGetData(prhs[2]);


    emb_n = mxGetPr(prhs[3]);
    num_n = mxGetN(prhs[3]);
    dim_emb = mxGetM(prhs[3]);

    starting_alpha = mxGetScalar(prhs[4]);
    num_neg = mxGetScalar(prhs[5]);

    neg_sam_table_social = (long long *)mxGetData(prhs[6]);
    table_size_social = mxGetM(prhs[6]);
    win_size = mxGetScalar(prhs[7]);

    neg_sam_table_mobility = prhs[8];
    table_num_mobility = mxGetNumberOfElements(prhs[8]);
    if(table_num_mobility != 4) {
        mexErrMsgTxt("four negative sample tables are required in neg_sam_table_mobility");
    }
    temp = mxGetCell(neg_sam_table_mobility, 0);
    neg_sam_table_mobility1 = (long long *)mxGetData(temp);
    table_size_mobility1 = mxGetM(temp);
    temp = mxGetCell(neg_sam_table_mobility, 1);
    neg_sam_table_mobility2 = (long long *)mxGetData(temp);
    table_size_mobility2 = mxGetM(temp);
    temp = mxGetCell(neg_sam_table_mobility, 2);
    neg_sam_table_mobility3 = (long long *)mxGetData(temp);
    table_size_mobility3 = mxGetM(temp);
    temp = mxGetCell(neg_sam_table_mobility, 3);
    neg_sam_table_mobility4 = (long long *)mxGetData(temp);
    table_size_mobility4 = mxGetM(temp);





    num_epoch = mxGetScalar(prhs[9]);
    num_threads = mxGetScalar(prhs[10]);

    mobility_ratio = mxGetScalar(prhs[11]);


    mexPrintf("walk size = %d %d\n", num_w,num_wl);
    mexPrintf("user checkins, user count = %d\n", num_u);
    mexPrintf("num of nodes: %lld; embedding dimension: %lld\n",num_n,dim_emb);
    mexPrintf("learning rate: %f\n",starting_alpha);
    mexPrintf("negative sample number: %f\n",num_neg);
    mexPrintf("social neg table size: %lld\n",table_size_social);
    mexPrintf("mobility neg table num: %lld\n",table_num_mobility);
    mexPrintf("mobility neg table sizes: %lld,%lld,%lld,%lld\n",table_size_mobility1,table_size_mobility2,table_size_mobility3,table_size_mobility4);
    mexPrintf("num_epoch: %lld\n",num_epoch);
    mexPrintf("num_threads: %lld\n",num_threads);

    fflush(stdout);

    long long a;
    pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
    for (a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, learn, (long long *)a);
    for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);

//      learn(0);
//
//     /* create the output matrix */
    plhs[0] = mxDuplicateArray(prhs[3]);
//
}
