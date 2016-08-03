//============================================================================
// Name        : toolkit.cpp
// Author      : zengzengfeng
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>
#include "Kmeans.h"
#include <stdlib.h>
#include "utils.h"
using namespace std;

void testKmeans(){

}

int main(int argc, char **argv) {
    if(argc<2){
	    cout << "usage:./" << argv[0]<<" -data datafile -save outfile -branch-num 5 -iter-num 10 -max-cluster-size 10 -topk 1"<<endl;
	    cout << "usage:./" << argv[0]<<" -data datafile -save outfile -k 5 -iter-num 10 -topk 1"<<endl;
        return -1;
    }
    Kmeans k1;
    int i=0;
    char* data;
    char* out_file;
    int branch_num=5;
    int iter_num=10;
    int max_cluster_size=10;
	int k=0;
	int topk=1;
    if ((i = ArgPos((char *)"-data", argc, argv)) > 0) data = argv[i + 1];
    if ((i = ArgPos((char *)"-save", argc, argv)) > 0) out_file = argv[i + 1];
    if ((i = ArgPos((char *)"-branch-num", argc, argv)) > 0) branch_num=atoi(argv[i+1]);
    if ((i = ArgPos((char *)"-iter-num", argc, argv)) > 0) iter_num=atoi(argv[i+1]);
    if ((i = ArgPos((char *)"-max-cluster-size", argc, argv)) > 0) max_cluster_size=atoi(argv[i+1]);
    if ((i = ArgPos((char *)"-k", argc, argv)) > 0) k=atoi(argv[i+1]);
    if ((i = ArgPos((char *)"-topk", argc, argv)) > 0) topk=atoi(argv[i+1]);
    k1.load_data(data);
    //k.level_kmeans(max_cluster_size,branch_num,iter_num);
	if(k>0){
		k1.cluster(k,iter_num);
	}else{
		k1.level_cluster(max_cluster_size,branch_num,iter_num);
	}
	if(topk>1){
		k1.cal_topk(topk);
	}
    k1.save(out_file);

    
	return 0;
}
