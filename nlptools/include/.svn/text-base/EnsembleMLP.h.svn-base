/*
 * EnsembleMLP.h
 *
 *  Created on: 2015-7-1
 *      Author: zengzengfeng
 */

#ifndef ENSEMBLEMLP_H_
#define ENSEMBLEMLP_H_
#include "MLPRewrite.h"
#include "NNRewrite.h"
using namespace std;

class EnsembleMLP :public MLPRewrite{
public:
    EnsembleMLP();
    ~EnsembleMLP();
    void InitNet();
    virtual void LoadModel(const string& infile);
    //void EmbPredict(vector<string>& query, size_t index, map<string, real>& score, nn_pack& pack);
    virtual void Predict(vector<string>& query, size_t index, map<string, real>& score, nn_pack& pack,int type);

protected:
    map<string,int> sub_label;
    vector<real *> w_vector;
    vector<real *> b_vector;
    int model_num;
};

#endif /* ENSEMBLEMLP_H_ */
