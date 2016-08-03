#include "LstmRewrite.h"


LstmRewrite::LstmRewrite()
{
	cerr << "LstmRewrite constructor:" << endl;
	word_vec_size = 50;
	sub_vec_size = 51;
	hidden_size = 50;
	thread_num = 1;

	vocab_size = 0;
	line_num = 0;
	alpha = 0.025;
	line_trained = 0;

	expTable = NULL;
	word_vec = NULL;
	sub_vec = NULL;
	w_size = (word_vec_size + hidden_size + 1) *  (4 * hidden_size);
	srand((unsigned)time(NULL));
}


LstmRewrite::~LstmRewrite()
{
	// TODO Auto-generated destructor stub
	if (expTable != NULL)
		delete expTable;
	expTable = NULL;
	if (word_vec != NULL)
		delete word_vec;
	word_vec = NULL;
	if (sub_vec != NULL)
		delete sub_vec;
	sub_vec = NULL;

	if (w != NULL)
		delete w;
	w = NULL;
	if (b != NULL)
		delete b;
	b = NULL;
}


void LstmRewrite::lstmForward(nn_pack pack)
{
}


void LstmRewrite::lstmBackward(nn_pack pack)
{
}


real LstmRewrite::forward(nn_pack& pack) {

}
void LstmRewrite::backward(nn_pack& pack) {

}
void LstmRewrite::sgdTrain(nn_pack& pack, const string& line, int local_num) {

}