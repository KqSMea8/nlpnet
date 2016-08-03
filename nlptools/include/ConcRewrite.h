/*
 * ConcRewrite.h
 *
 *  Created on: 2015-6-15
 *      Author: zengzengfeng
 */

#ifndef CONCREWRITE_H_
#define CONCREWRITE_H_
#include "NNRewrite.h"
using namespace std;

class ConcRewrite: public NNRewrite{
public:
    ConcRewrite();
    ~ConcRewrite();
    virtual void InitNet();
private:
    virtual real forward(nn_pack& pack);
    virtual void backward(nn_pack& pack);
};

#endif /* CONCREWRITE_H_ */
