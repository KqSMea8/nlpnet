
#include "NNRewrite.h"
class LstmRewrite :
	public NNRewrite
{
public:
	LstmRewrite();
	~LstmRewrite();
protected:
	virtual real forward(nn_pack& pack);
	virtual void backward(nn_pack& pack);
	virtual void sgdTrain(nn_pack& pack, const string& line, int local_num);
	void lstmForward(nn_pack pack);
	void lstmBackward(nn_pack pack);
};

