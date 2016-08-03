
// Matrix factorization: X=UV

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <ostream>
#include <vector>
#include "utils.h"
using namespace std;


#define _FILE_OFFSET_BITS 64
#define MAX_STRING_LENGTH 1000
pthread_mutex_t mutex1 = PTHREAD_MUTEX_INITIALIZER;
typedef double real;

typedef struct cooccur_rec {
    int word1;
    int word2;
    real val;
} CREC;

int verbose = 2; // 0, 1, or 2
int num_threads = 8; // pthreads
int num_iter = 25; // Number of full passes through cooccurrence matrix
int vector_size = 50; // Word vector size
int u_size=0;
int v_size=0;
real eta = 0.05; // Initial learning rate
real *U, *V, *gradu, *gradv, *cost;
long long num_lines, *lines_per_thread, vocab_size, file_size ;
char  *input_file, *output_file;


/* Efficient string comparison */
int scmp( char *s1, char *s2 ) {
    while(*s1 != '\0' && *s1 == *s2) {s1++; s2++;}
    return(*s1 - *s2);
}

void initialize_parameters() {
	long long a, b;
	//vector_size++;
	a = posix_memalign((void **)&U, 128, u_size * vector_size * sizeof(real));
	a = posix_memalign((void **)&gradu, 128, u_size * vector_size * sizeof(real));
	a = posix_memalign((void **)&V, 128, v_size * vector_size * sizeof(real));
	a = posix_memalign((void **)&gradv, 128, v_size * vector_size * sizeof(real));
    if (V == NULL || U == NULL || gradu == NULL || gradv == NULL) {
        fprintf(stderr, "Error allocating memory \n");
        exit(1);
    }

    for (b = 0; b < vector_size; b++){
        for (a = 0; a < u_size; a++){
            U[a * vector_size + b] = (rand() / (real)RAND_MAX - 0.5) / vector_size;
            gradu[a * vector_size + b] = 1.0;
        }
    }
    for (b = 0; b < vector_size; b++){
        for (a = 0; a < v_size; a++){
            V[a * vector_size + b] = (rand() / (real)RAND_MAX - 0.5) / vector_size;
            gradv[a * vector_size + b] = 1.0;
        }
    }
    //vector_size--;
}

/* Train the mf model */
void *glove_thread(void *vid) {

    ifstream fin(input_file);
    long start_index = file_size / (long long) num_threads * (long long) vid;
    fin.seekg(start_index);
    long long local_num = 0;
    long long read_size = file_size / num_threads;
    pthread_mutex_lock(&mutex1);
    cerr << "thread" << long(vid) << " start_index:" << start_index
        << " read_size:" << read_size << endl;
    pthread_mutex_unlock(&mutex1);

    string line;
    vector<string> vec;
    long long cur_size = 0;

    long long a, b, l1, l2;
    long long id = (long long) vid;
    CREC cr;
    real diff, temp1, temp2;
    cost[id] = 0;

    if((long long) vid>0) getline(fin, line);
    while (getline(fin, line)) {
        if (cur_size > read_size+1)
            break;
        cur_size += line.size() + 1;
        local_num++;
        int sub_id = 0;
        split_str(vec, line, " ");
        cr.word1=atoi(vec[0].c_str());
        cr.word2=atoi(vec[1].c_str());
        cr.val =atof(vec[2].c_str());

        l1 = (cr.word1) * (vector_size); // U index
        l2 = (cr.word2) * (vector_size); // V index

        /* Calculate cost, save diff for gradients */
        diff = 0;
        for (b = 0; b < vector_size; b++)
            diff += U[b + l1] * V[b + l2]; // dot product of word and context word vector
        real before = diff;
        if(local_num%100000==0){
            cerr << cr.word1 << " " << cr.word2 << " " << before << " " << cr.val << endl;
        }
        diff = diff - cr.val;
        cost[id] += 0.5 * diff * diff; // weighted squared error

        /* Adaptive gradient updates */
        diff *= eta; // for ease in calculating gradient
        for (b = 0; b < vector_size; b++) {
            // learning rate times gradient for word vectors
            temp1 = diff * V[b + l2];
            temp2 = diff * U[b + l1];
            // adaptive updates
            U[b + l1] -= temp1 / sqrt(gradu[b + l1]);
            V[b + l2] -= temp2 / sqrt(gradv[b + l2]);
            gradu[b + l1] += temp1 * temp1;
            gradv[b + l2] += temp2 * temp2;
        }

    }
    lines_per_thread[id]=local_num;
    fin.close();
}

/* Save params to file */
int save_params() {
    long long a, b;
    FILE *fout;
    fout = fopen(output_file,"wb");
    fprintf(fout, "%d %d\n", u_size, v_size);
    for (a = 0; a < u_size; a++) {
        //fprintf(fout, "%s", word);
        fprintf(fout, "%lf", U[a * vector_size]);
        for (b = 1; b < vector_size; b++) {
            fprintf(fout, " %lf", U[a * vector_size + b]);
        }
        fprintf(fout,"\n");
    }
    
    for (a = 0; a < v_size; a++) {
        fprintf(fout, "%lf", V[a * vector_size ]);
        for (b = 1; b < vector_size ; b++) {
            fprintf(fout, " %lf", V[a * vector_size + b]);
        }
        fprintf(fout,"\n");
    }
    fclose(fout);
    return 0;
}

/* Train model */
int train_glove() {
    long long a;
    int b;
    FILE *fin;
    real total_cost = 0;
    fprintf(stderr, "TRAINING MODEL\n");
    
    fin = fopen(input_file, "rb");
    if(fin == NULL) {fprintf(stderr,"Unable to open cooccurrence file %s.\n",input_file); return 1;}
    fseeko(fin, 0, SEEK_END);
    file_size = ftello(fin);
    fclose(fin);

    if(verbose > 1) fprintf(stderr,"Initializing parameters...");
    initialize_parameters();
    if(verbose > 1) fprintf(stderr,"done.\n");
    if(verbose > 0) fprintf(stderr,"vector size: %d\n", vector_size);
    if(verbose > 0) fprintf(stderr," u_size: %d v_size: %d \n", u_size, v_size);

    pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
    lines_per_thread = (long long *) malloc(num_threads * sizeof(long long));
    
    // Lock-free asynchronous SGD
    for(b = 0; b < num_iter; b++) {
        total_cost = 0;
        for (a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, glove_thread, (void *)a);
        for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);
        num_lines=0;
        for (a = 0; a < num_threads; a++) {
            total_cost += cost[a];
            num_lines += lines_per_thread[a];
        }
        fprintf(stderr,"iter: %03d, cost: %lf , num_lines: %d\n", b+1, total_cost/num_lines, num_lines);
    }
    return save_params();
}

int find_arg(char *str, int argc, char **argv) {
    int i;
    for (i = 1; i < argc; i++) {
        if(!scmp(str, argv[i])) {
            if (i == argc - 1) {
                printf("No argument given for %s\n", str);
                exit(1);
            }
            return i;
        }
    }
    return -1;
}

int main(int argc, char **argv) {
    int i;
    input_file = (char *)malloc(sizeof(char) * MAX_STRING_LENGTH);
    output_file = (char *)malloc(sizeof(char) * MAX_STRING_LENGTH);

    
    if (argc == 1) {
        printf("Usage options:\n");
        printf("\t-verbose <int>\n");
        printf("\t\tSet verbosity: 0, 1, or 2 (default)\n");
        printf("\t-vector-size <int>\n");
        printf("\t\tDimension of word vector representations (excluding bias term); default 50\n");
        printf("\t-threads <int>\n");
        printf("\t\tNumber of threads; default 8\n");
        printf("\t-iter <int>\n");
        printf("\t\tNumber of training iterations; default 25\n");
        printf("\t-eta <float>\n");
        printf("\t\tInitial learning rate; default 0.05\n");
        printf("\t-alpha <float>\n");
        printf("\t-usize <int>\n");
        printf("\t-vsize <int>\n");
        printf("\t-input-file <file>\n");
        printf("\t-save-file <file>\n");
        return 0;
    }
    
    
    if ((i = find_arg((char *)"-verbose", argc, argv)) > 0) verbose = atoi(argv[i + 1]);
    if ((i = find_arg((char *)"-vector-size", argc, argv)) > 0) vector_size = atoi(argv[i + 1]);
    if ((i = find_arg((char *)"-iter", argc, argv)) > 0) num_iter = atoi(argv[i + 1]);
    if ((i = find_arg((char *)"-threads", argc, argv)) > 0) num_threads = atoi(argv[i + 1]);
    cost = (real*) malloc(sizeof(real) * num_threads);
    if ((i = find_arg((char *)"-eta", argc, argv)) > 0) eta = atof(argv[i + 1]);
    if ((i = find_arg((char *)"-save-file", argc, argv)) > 0) strcpy(output_file, argv[i + 1]);
    if ((i = find_arg((char *)"-input-file", argc, argv)) > 0) strcpy(input_file, argv[i + 1]);
    if ((i = find_arg((char *)"-usize", argc, argv)) > 0) u_size=atoi(argv[i + 1]);
    if ((i = find_arg((char *)"-vsize", argc, argv)) > 0) v_size=atoi(argv[i + 1]);
    return train_glove();
}
