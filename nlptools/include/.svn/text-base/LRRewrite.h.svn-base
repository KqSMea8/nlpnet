#ifndef NNREWRITE_H
#define NNREWRITE_H
#include "NNRewrite.h"
class LRRewrite :
	public NNRewrite
{
public:
	LRRewrite();
	~LRRewrite();
	virtual void PrintModel();
	virtual void InitPack(nn_pack& pack);
	void print_pack(const nn_pack&);
private:
	virtual real forward(nn_pack& pack);
	virtual void backward(nn_pack& pack);
	virtual void sgdTrain(nn_pack& pack, const string& line, int local_num);

	unsigned int w_left_0;
	unsigned int w_right_0;


	inline void stop() {
		cout << "hit enter to continue¡­¡­" << endl;
		getchar();
	}
	inline real sigmod(real z){
		real a = exp(z);
		return a / (1 + a);
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

	inline void add(const vector<real>& arg1, const real* arg2, vector<real>& result)
	{
		for (unsigned long long i = 0; i < result.size(); i++)
		{
			//cout << i << ":" << arg1[i] << "+" << arg2[i] << "=";
			result[i] = arg1[i] + arg2[i];
			 //cout << result[i] << endl;
		}
	}
};

#endif
