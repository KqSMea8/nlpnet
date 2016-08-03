/*
 * ConvRewrite.h
 *
 *  Created on: 2015-6-15
 *      Author: zengzengfeng
 */

#ifndef CONVREWRITE_H_
#define CONVREWRITE_H_
#include "NNRewrite.h"
using namespace std;

class ConvRewrite: public NNRewrite{
public:
    ConvRewrite();
    ~ConvRewrite();
    virtual void InitPack(nn_pack& pack);
private:
    virtual real forward(nn_pack& pack);
    virtual void backward(nn_pack& pack);
    void convFunc(vector<real*>& x,size_t begin,size_t end, vector<real>& out, nn_pack& pack);
private:
    int max_conv_num;
};

#endif /* CONVREWRITE_H_ */
