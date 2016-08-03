//============================================================================
// Name        : HMM.cpp
// Author      : zengzengfeng
// Version     :
// Copyright   : Your copyright notice
// Description : viterbi approximate for HMM
//============================================================================

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <iostream>
#include <stdlib.h>
#include <malloc.h>
#include <map>
#include <vector>
#include <stdio.h>
#include <math.h>
#include <algorithm>
#include "HmmCluster.h"
#include "time.h"
using namespace std;

HmmCluster::HmmCluster() {
	// TODO Auto-generated constructor stub
	iteration=3;
}

HmmCluster::~HmmCluster() {
	// TODO Auto-generated destructor stub
	if(A){
		for(int i=0;i<N;i++){
			delete []A[i];
		}
		delete [] A;
	}
	if(cntA){
		for(int i=0;i<N;i++){
			delete []cntA[i];
		}
		delete [] cntA;
	}
	if(B) delete [] B;
	if(cntB)delete [] cntB;
	if(totalB) delete [] totalB;
	if(pi) delete [] pi;

}

bool compare(const pair<int, double> &v1, const pair<int, double> &v2) {
	return v1.second > v2.second;
}
void HmmCluster::init() {
    srand(time(NULL));

	A = new double*[N];
	cntA = new int*[N];
	B = new map<int, double> [N];
	cntB = new map<int, int> [N];
	totalB = new int[N];
	for (int i = 0; i < N; i++) {
		A[i] = new double[N];
		cntA[i] = new int[N];
		totalB[i]=1;
	}
	pi = new double[N];

	double sum;
	for (int i = 0; i < N; i++) {
		sum = 0.0;
		for (int j = 0; j < N; j++) {
			A[i][j] = (double) rand() / RAND_MAX;
            cntA[i][j]=0;
			sum += A[i][j];
		}
		for (int j = 0; j < N; j++)
			A[i][j] /= sum;
	}

	sum = 0.0;
	for (int i = 0; i < N; i++) {
		pi[i] = (double) rand() / RAND_MAX;
		sum += pi[i];
	}
	for (int i = 0; i < 0; i++)
		pi[i] /= sum;
}

bool HmmCluster::pushLattice(vector<pair<int, double> >& lattice, int i, double score,
		int beamWidth) {
	if (lattice.size() < beamWidth) {
		lattice.push_back(pair<int, double>(i, score));
		make_heap(lattice.begin(), lattice.end(), compare);
		return true;
	} else {
		if (score > lattice.front().second) {
			pop_heap(lattice.begin(), lattice.end());
			lattice.pop_back();
			lattice.push_back(pair<int, double>(i, score));
			push_heap(lattice.begin(), lattice.end(), compare);
			make_heap(lattice.begin(), lattice.end(), compare);
			return true;
		} else {
			return false;
		}
	}
}
void printLattice(vector<pair<int, double> >& lattice){
    for(int i=0;i<lattice.size();i++){
        cout<<lattice[i].first<<" "<<lattice[i].second<<endl;
    }
}
double HmmCluster::getOmitProb(int s, int o) {
	double smooth = 1.0 / M;
	double omitProb = 0.0;
	map<int, double>::iterator it = B[s].find(o);
	if (it != B[s].end()) {
		omitProb = it->second;
	}
	omitProb = alpha * omitProb + (1 - alpha) * smooth;
	return omitProb;
}
void HmmCluster::viterbi(int T, int *O, int *q) {
    //cout<<"viterbi....."<<endl;
	/* 1. Initialization  */
	vector<map<int, int> > psi(T);
	vector<pair<int, double> > initLattice;
	for (int i = 0; i < N; i++) {
		double omitProb = getOmitProb(i,O[0]);
		//double score = omitProb * pi[i];
		double score = log(omitProb)+log(pi[i]);
		pushLattice(initLattice, i, score, 100);
		psi[0][i] = 0;
	}
	/* 2. Recursion */
	vector<pair<int, double> > * preLattice = &initLattice;
	vector<pair<int, double> > * nextLattice = new vector<pair<int, double> >();
	int maxvalind;
	double maxval, val;
	for (int t = 1; t < T; t++) {
        //cout<<"prelattice:"<<endl;
        //printLattice(*preLattice);
		(*nextLattice).clear();
		//sparsity of transition matrix for performance
		for (int i = 0; i < N; i++) {
			maxval = -100000000000.0;
			maxvalind = 1;
			for (int j = 0; j < (*preLattice).size(); j++) {
				//val = (*preLattice)[j].second * A[(*preLattice)[j].first][i];
				val = (*preLattice)[j].second +log(A[(*preLattice)[j].first][i]);
				if (val > maxval) {
					maxval = val;
					maxvalind = (*preLattice)[j].first;
				}
			}
			//double score = maxval * getOmitProb(i,O[t]);
			double score = maxval +log(getOmitProb(i,O[t]));
			pushLattice(*nextLattice, i, score, 100);
			psi[t][i] = maxvalind;
		}
        //cout<<"nextlattice:"<<endl;
        //printLattice(*nextLattice);
		vector<pair<int, double> > *p = nextLattice;
		nextLattice = preLattice;
		preLattice = p;
	}
	/* 3. Termination */

	double pprob = -100000000000.0;
	q[T - 1] = 0;
	for (int i = 0; i < preLattice->size(); i++) {
		if ((*preLattice)[i].second > pprob) {
			pprob = (*preLattice)[i].second;
			q[T - 1] = (*preLattice)[i].first;
		}
	}
	/* 4. Path (state sequence) backtracking */
	for (int t = T - 2; t >= 0; t--){
		q[t] = psi[t + 1][q[t + 1]];
	}

}

void HmmCluster::esitmate() {
	//re-estimate the parameter
    int smooth=1;
	for (int a = 0; a < N; a++) {
		int sum = 0;
		for (int b = 0; b < N; b++) {
            cntA[a][b]+=smooth;
			sum += cntA[a][b];
		}
		for (int b = 0; b < N; b++) {
			A[a][b] = cntA[a][b] * 1.0 / sum;
			cntA[a][b] = 0;
		}
	}

	int total = 0;
    //cout<<"B:cntB"<<endl;
	for (int a = 0; a < N; a++) {
		B[a].clear();
		total += totalB[a];
		for (map<int, int>::iterator it = cntB[a].begin(); it != cntB[a].end();
				++it) {
			B[a][it->first] = cntB[a][it->first]*1.0 / totalB[a];
            //cout<<a<<"->"<<it->first<<" "<<id2Word[it->first]<<" "<<B[a][it->first]<<" "<<cntB[a][it->first]<<endl;
		}
        cntB[a].clear();
	}
   // cout<<"pi:"<<endl;
	for (int a = 0; a < N; a++) {
		pi[a] = totalB[a]*1.0 / total;
        //cout<<pi[a]<<" ";
		totalB[a] = 0;
	}
    //cout<<endl;
}

void HmmCluster::diskTrain(const char* inFile) {

	init();
	ifstream idstream(inFile);
	string line;
	int O[MAX_T];
	int q[MAX_T];
	for (int i = 0; i < iteration; i++) {
        cout<<"################ iteration "<<i<<endl;
		while (getline(idstream, line)) {
			vector<string> tokens;
            //cout<<"iter"<<i<<"\t"<<line<<endl;
			splitStr(tokens, line, " ");
			int t = tokens.size();
			for (int k = 0; k < t; k++) {
				O[k] = atoi(tokens[k].c_str());
			}
			if(i==0){
				for(int k=0;k<t;k++) q[k]=rand()%N;
			}else{
				viterbi(tokens.size(), O, q);
			}

			for (int k = 0; k < t; k++) {
				map<int, int>::iterator iter = cntB[q[k]].find(O[k]);
				if(iter==cntB[q[k]].end()){
					cntB[q[k]][O[k]]=1;
				}else{
					iter->second+=1;
				}
				totalB[q[k]]+=1;
				if(k>0){
					cntA[q[k-1]][q[k]]+=1;
				}
			}
		}
		idstream.clear();		// 清掉所有错误标志
		idstream.seekg(0, ios_base::beg);	// ok
		//re-estimate the parameter
		esitmate();
	}
	idstream.close();
}
void HmmCluster::splitStr(vector<string>& vecToken, const string& str, string separator) {
	size_t pos1, pos2;
	string token;
	size_t len = separator.length();
	pos1 = pos2 = 0;
	vecToken.clear();
	while ((pos2 = str.find(separator.c_str(), pos1)) != string::npos) {
		token = str.substr(pos1, pos2 - pos1);
		pos1 = pos2 + len;
		if (!token.empty()) {
			vecToken.push_back(token);
		}
	}
	token = str.substr(pos1);
	if (!token.empty()) {
		vecToken.push_back(token);
	}
}
int HmmCluster::word2id(const char* inFile, const char* idFile) {
	ifstream in(inFile);
	map<string, int> word2Id;
	if (!in) {
		fprintf(stderr, "error: the file %s doesn't exist", inFile);
		return -1;
	}
	string line;
	int id = 0;
    id2Word.clear();
	ofstream idstream(idFile);
	while (getline(in, line)) {
		vector<string> tokens;
		splitStr(tokens, line, " ");
		ostringstream oss;
		for (unsigned i = 0; i < tokens.size(); i++) {
			map<string, int>::iterator iter = word2Id.find(tokens[i]);
			int tid = 0;
			if (iter == word2Id.end()) {
				word2Id[tokens[i]] = id;
				tid = id;
				this->id2Word.push_back(tokens[i]);
				id++;
			} else {
				tid = iter->second;
			}
			idstream << tid << " ";
		}
		idstream << endl;
	}
	in.close();
	idstream.flush();
	idstream.close();
	M=id2Word.size();
	return 0;
}
void HmmCluster::save(const char* dir){

	string matFile=string(dir)+"/A.txt";
	ofstream matOut(matFile.c_str());
	for(int i=0;i<N;i++){
		for(int j=0;j<N;j++){
			matOut<<A[i][j]<<"\t";
		}
		matOut<<endl;
	}
	matOut.flush();
	matOut.close();
	string clusterFile=string(dir)+"/B.txt";
	ofstream clusterOut(clusterFile.c_str());
	for(int i=0;i<N;i++){
		for(map<int,double>::iterator iter=B[i].begin();iter !=B[i].end();++iter){

			clusterOut<<i<<"\t"<<id2Word[iter->first]<<"\t"<<iter->second<<endl;
		}
	}
	clusterOut.flush();
	clusterOut.close();


}
void HmmCluster::setIteration(int iteration) {
	this->iteration = iteration;
}

void HmmCluster::setSymbolsNum(int m) {
	M = m;
}

void HmmCluster::setStatesNum(int n) {
	N = n;
}




