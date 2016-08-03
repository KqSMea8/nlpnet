//============================================================================
// Name        : testIBM.cpp
// Author      : zengzengfeng
// Version     :
// Copyright   : Your copyright notice
// Description :
//============================================================================

#include <iostream>
#include <vector>
#include <map>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include "IBM1.h"
using namespace std;

int main(int argc, char **argv) {
	if(argc<5){
		cerr<<"usage:./"<<argv[1]<<" infile outfile mode[disk_run:0|pipe_run:1] iter_num "<<endl;
	}
	IBM1 ibm;
	int mode=atoi(argv[3]);
	if(mode==0){
		ibm.disk_run(argv[1],argv[2],atoi(argv[4]));
	}
	if(mode==1){
		ibm.pipe_run(cin);
		ibm.save_align_prob(argv[2]);
	}
	return 0;
}
