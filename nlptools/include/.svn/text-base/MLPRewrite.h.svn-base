/*
 * MLPRewrite.h
 *
 *  Created on: 2015-6-15
 *      Author: zengzengfeng
 */

#ifndef MLPREWRITE_H_
#define MLPREWRITE_H_
#include "NNRewrite.h"
using namespace std;

class MLPRewrite: public NNRewrite{
public:
    MLPRewrite();
    ~MLPRewrite();
	void print_pack(const nn_pack&);
protected:
    virtual real forward(nn_pack& pack);
    virtual void backward(nn_pack& pack);
    virtual void sgdTrain(nn_pack& pack, const string& line, int local_num);
    void softmax2(vector<real>& z, vector<real>& a);


	inline void stop() {
		cout << "hit enter to continue¡­¡­" << endl;
		getchar();
	}
	inline real acti(real arg){
		if (arg < -6) return -1.0;
		else if (arg > 6) return 1.0;
		return tanh(arg);
	}

	inline real acti_der(real sigma) {
		return 1 - (sigma*sigma);
	}

	inline real inner(const real * arg1, const real * arg2, unsigned long long len)
	{
		real re = 0;
		for (unsigned long long i = 0; i < len; i++)
		{
			re += arg1[i] * arg2[i];
			//cout << arg1[i] << " " << arg2[i] << endl;
		}
		//cout << re << endl;
		return re;
	}
	inline real inner(const real * arg1, const vector<real>& arg2) {
		real re = 0;
		for (unsigned long long i = 0; i < arg2.size(); i++)
		{
			re += arg1[i] * arg2[i];
			//cout << arg1[i] << " " << arg2[i] << endl;
		}
		//cout << re << endl;
		return re;
	}


};

#endif /* MLPREWRITE_H_ */
