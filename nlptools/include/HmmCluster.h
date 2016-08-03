/*
 * Hmm.h
 *
 *  Created on: 2013-11-10
 *      Author: zengzengfeng
 */

#ifndef HMM_H_
#define HMM_H_
#include<map>
#include<vector>
using namespace std;
const double alpha = 0.8;
const int beamWidth = 20;
const int MAX_T = 100;
class HmmCluster {
public:
	HmmCluster();
	~HmmCluster();
	void init();
	void diskTrain(const char* inFile);
	int word2id(const char* inFile, const char* idFile);
	void setIteration(int iteration);
	void setSymbolsNum(int m);
	void setStatesNum(int n);
	void save(const char* dir);
private:
	void splitStr(vector<string>& vecToken, const string& str,
			string separator);
	bool pushLattice(vector<pair<int, double> > & lattice, int i, double score,
			int beamWidth);
	double getOmitProb(int s, int o);
	void viterbi(int T, int *O, int *q);
	void esitmate();
private:
	int iteration;
	int M; //the number of symbols
	int N; // the number of states
	double ** A; //A[i][j] prob that state i transfer to j
	int ** cntA;
	map<int, double>* B; //B[i][j] prob that state i omit j
	map<int, int>* cntB; //cntB[i][j] times that state j omit i
	int* totalB; //totalB[j] total time that state j happens
	vector<string> id2Word;
	double * pi;

};

#endif /* HMM_H_ */
