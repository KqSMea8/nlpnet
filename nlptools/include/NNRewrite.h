/*
 * RWVector.h
 *
 *  Created on: 2014-7-11
 *      Author: zengzengfeng
 */

#ifndef NNRewrite_H_
#define NNRewrite_H_

#include <iostream>
#include <vector>
#include <map>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "utils.h"
#include "typedef.h"

using namespace std;
#define MAX_STRING 100
#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6

const int table_size = 1e8;
const real EPSILON = 0.00000001;
enum {
    TEXT, ID
};


struct nn_pack
{
    vector<real *> x;
    vector<real *> u;
    vector<real> hidden;
    vector<real> z2;
    vector<real> a2;
    vector<real> a3;
    vector<real> delta2;
    vector<real> grad_x;
    vector<real> grad_u;
	
	vector<vector<real> > grad_z1;
	vector<real> grad_z1_OfHid;
	vector<real> delta1;
	

    vector<vector<real> > conv;
    vector<int> max_pos;
    int conv_num;

    int label;
    int fid;
    real freq;

    vector<string> words;
    vector<int> wids;

    vector<real> xl;
    vector<real> xr;
    
};


class NNRewrite {
public:
    NNRewrite();
    virtual ~NNRewrite();
    virtual void InitNet();
    virtual void InitPack(nn_pack& pack);

    virtual void SaveModel(const string& outfile);
    virtual void LoadModel(const string& infile);
    //virtual void LoadModel2(const string& infile);
    virtual void PrintModel();

    void Train(int id, string& train_file,long long file_size);
    void SampleLoss();

    void SetThreadNum(int num);
    void SetIterNum(int iter, int max_iter);


    void LoadDicts(const string& subfile,const string& wordfile);
    void SetInputType(int type);
    void GradientCheck();
    virtual void Predict(vector<string>& query, size_t index,
        map<string, real>& score, nn_pack& pack, int type);
    virtual void Predict2(vector<string>& query, size_t begin, size_t end,
        map<string, pair<real,real> >& score, nn_pack& pack);

    virtual void Expand(vector<string>& query1, string& query2,
           map<string, real>& score, nn_pack& pack);
protected:
    void loadSubDict(const string& subfile);
    void loadWordDict(const string& wordfile);
    int parse_pack(vector<string>& vec,nn_pack& pack, int type);
    void word2index(vector<string>& words, vector<int>& wids);
    void getWordVec(vector<int>& wids,vector<real* >& x);
    void getWordVec(vector<int>& wids,vector<real* >& x, size_t begin, size_t end);
    void getSubVec(pair<int, int>& range,vector<real* >& u);
    void sumFunc( vector<real*>& x, vector<real>& hidden, nn_pack& pack);
    void softmax(vector<real>& z, vector<real>& a);
    virtual real forward(nn_pack& pack);
    virtual void backward(nn_pack& pack);
    virtual void sgdTrain(nn_pack& pack, const string& line, int local_num);
    real fast_exp(real f);
    void print_vec(vector<real>& z);

	inline real sigmod(real z){
		real a = exp(z);
		return a / (1 + a);
	}

protected:
    map<string, int> word_index;
    map<string, int> sub_index;
    map<string,pair<int,int> > sub_list;

    vector<int> sub_buffer;
    vector<string> sub_string;
    vector<string> vocab;

    vector<string> samples;

    real* word_vec;
    real* sub_vec;
    //real* w1;
    real * w;
    real * b;
    real* expTable;

    int word_vec_size;
    int sub_vec_size;
    int sub_num;
    int hidden_size;
    int window_size;
    int min_word_freq;
    real alpha;

    int thread_num;
    //long long file_size;
    int vocab_size;
    int line_num;
    int line_cnt;
    int iter_num;
    int max_iter_num;

    int input_type;
    bool update_word;

    int label;

    map<string,int> sub_label;
    vector<real *> w_vector;
    vector<real *> b_vector;
    int model_num;
    unsigned int w_size;
	unsigned long long line_trained;
};

#endif /* NNRewrite_H_ */
