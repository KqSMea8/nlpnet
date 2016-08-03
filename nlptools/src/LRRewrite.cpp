#include "LRRewrite.h"


LRRewrite::LRRewrite() 
{
	word_vec_size = 50;
	sub_vec_size = 50;
	window_size = 5;
	thread_num = 1;

	vocab_size = 0;
	line_num = 0;
	alpha = 0.025;

	w_left_0 = 0;
	w_right_0 = word_vec_size*sub_vec_size;
	w_size = 2 * word_vec_size * sub_vec_size;

	expTable = NULL;
	word_vec = NULL;
	sub_vec = NULL;

	line_trained = 0;
}


LRRewrite::~LRRewrite()
{
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




void LRRewrite::InitPack(nn_pack& pack)
{
    const unsigned int MAX_SUB_NUM= 250;
	pack.hidden = vector<real>(sub_vec_size, 0);
	pack.grad_z1.clear();
	for (int k = 0; k < MAX_SUB_NUM; k++)
	{
		vector<real> tmp;
		for (int m = 0; m < sub_vec_size; m++)
		{
			tmp.push_back(0.0);
		}
		pack.grad_z1.push_back(tmp);
	}
	pack.grad_z1_OfHid = vector<real>(sub_vec_size, 0);
	pack.delta1 = vector<real>(sub_vec_size, 0);
	pack.xr = vector<real>(word_vec_size, 0);
	pack.xl = vector<real>(word_vec_size, 0);

}
void LRRewrite::print_pack(const nn_pack & pack) {
	for (size_t i = 0; i < pack.x.size(); i++)
	{
		if (i == pack.fid)
		{
			continue;
		}
		cout << "x_" << i << ": ";
		for (int j = 0; j < word_vec_size; j++)
		{
			cout << pack.x[i][j] << " ";
		}
		cout << endl;
	}
	cout << "xl: ";
	for (int j = 0; j < word_vec_size; j++)
	{
		cout << pack.xl[j] << " ";
	}
	cout << endl;
	cout << "xr: ";
	for (int j = 0; j < word_vec_size; j++)
	{
		cout << pack.xr[j] << " ";
	}
	cout << endl;
	for (size_t i = 0; i < pack.hidden.size(); i++)
	{
		cout << "h_" << i << ": " << pack.hidden[i] << endl;
	}
	for (size_t i = 0; i < pack.z2.size(); i++)
	{
		cout << "z2_" << i << ": " << pack.z2[i] << endl;
	}
	for (size_t i = 0; i < pack.a2.size(); i++)
	{
		cout << "a2_" << i << ": " << pack.a2[i] << endl;
	}

}
real LRRewrite::forward(nn_pack& pack) {

	pack.z2.clear();
	pack.a2.clear();
	pack.hidden.clear();


	for (size_t i = 0; i < word_vec_size; i++)
	{
		pack.xr[i] = 0.0;
		pack.xl[i] = 0.0;
	}


	
	for (size_t i = 0; i < pack.fid; i++)
	{
		add(pack.xl, pack.x[i], pack.xl);
	}
	
	for (size_t i = pack.fid+1; i < pack.x.size(); i++)
	{
		add(pack.xr, pack.x[i], pack.xr);
	}

	//for (size_t i = 0; i < word_vec_size; i++)
	//{
	//	cout << i << ": " << pack.xl[i] << "   " << pack.xr[i] << endl;
	//}

	unsigned int l_idx = w_left_0;
	unsigned int r_idx = w_right_0;
	for (size_t m= 0; m < sub_vec_size; m++)
	{
		real z1_m = 
			inner(&w[l_idx], pack.xl) + 
			inner(&w[r_idx], pack.xr) +
			b[m];
		//cout << "z1_" << m <<": " << z1_m << endl;
		pack.hidden.push_back(acti(z1_m));
		//cout << m << ": " << pack.hidden[m] << endl;
		l_idx += word_vec_size;
		r_idx += word_vec_size;
	}

	//stop();

	
	for (size_t k = 0; k < pack.u.size(); k++) {

		real z2_k = inner(pack.u[k], pack.hidden);
		//cout << k << ": " << z2_k << endl;
		//for (int i = 0; i < sub_vec_size; i++)
		//{
		////cout << pack.u[k][i] << " " << pack.hidden[i] << endl;
		//	cout << "<"<<k << ">"<< pack.hidden[i] << endl;
		//}
		

		pack.z2.push_back(z2_k);
	}
	
	softmax(pack.z2, pack.a2);

	/*for (size_t k = 0; k < pack.a2.size(); k++)
	{
		cout << k << ": " << pack.a2[k] << endl;
	}*/

	return -log(pack.a2[pack.label]);
}

void LRRewrite::backward(nn_pack& pack)
{
	real eps = 1e-5;
	pack.delta2 = pack.a2;
	pack.delta2[pack.label] = pack.delta2[pack.label] - 1.0;

	/*float ratio=(pack.delta2.size()- pack.freq-2)/pack.delta2.size();
	if(ratio<0) ratio=0.0;

	for(size_t i=0; i< pack.delta2.size(); i++){
	if(i==pack.label) continue;
	real randnum=rand() / (real) RAND_MAX;
	if(randnum<ratio)  pack.delta2[i]=0;

	}*/


	//float beta = 0.0;
	int max = line_cnt*max_iter_num / 2;
	alpha = 0.01*(1 - real(line_trained) / real(max + 1));
	if (alpha < 0.0001)
	{
		alpha = 0.0001;
	}
	if (iter_num == 0) alpha = 0.01;
	alpha = alpha*pack.freq;

	for (int m = 0; m< sub_vec_size; m++)
	{
		pack.grad_z1_OfHid[m] = acti_der(pack.hidden[m]);
	}


	for (int m = 0; m < sub_vec_size; m++)
	{
		pack.delta1[m] = 0.0;
	}

	for (size_t k = 0; k < pack.u.size(); k++)
	{
		for (int m = 0; m < sub_vec_size; m++)
		{
			pack.grad_z1[k][m] =
				pack.delta2[k] *
				pack.u[k][m] *
				pack.grad_z1_OfHid[m];

			pack.delta1[m] += pack.grad_z1[k][m];
			//cout << "sub: " << k << " hid: " << m << " " << pack.grad_z1[k][m] << endl;
			/*cout << "grad_z1_sumAllSub " << m
			<< ": " << pack.grad_z1_sumAllSub[m] << endl;*/
		}
	}

	//update sub_vec
	for (size_t k = 0; k < pack.u.size(); k++) {
		for (size_t m = 0; m < sub_vec_size; m++) {
			real g_u_k_m = pack.delta2[k] * pack.hidden[m];
			//**********verify grad***********
			/*pack.u[k][m] += eps;
			real f1 = forward(pack);
			cout << f1 << " - ";
			pack.u[k][m] -= 2 * eps;
			real f2 = forward(pack);
			cout << f2 << endl;
			pack.u[k][m] += eps;
			cout << "verify grad_u_" << k << "_" << m << ": "
				<< (f1 - f2) / (2 * eps) << " " << g_u_k_m << endl;
			stop();*/
			//******************************
			pack.u[k][m] -= alpha*g_u_k_m;
		}
	}

	//update b
	for (int m = 0; m < sub_vec_size; m++)
	{
		real g_b_m =  pack.delta1[m];

		//**********verify grad***********
		/*b[m] += eps;
		real f1 = forward(pack);
		cout << f1 << " ";
		b[m] -= 2 * eps;
		real f2 = forward(pack);
		cout << f2 << endl;
		b[m] += eps;
		cout << b[m] << endl;
		cout << "verify grad_b_" << m << ": "
			<< (f1 - f2) / (2 * eps) << " " << g_b_m << endl;
		stop();*/
		//******************************
		b[m] -= alpha*g_b_m;
	}

	//update w
	unsigned int l_idx = w_left_0;
	unsigned int r_idx = w_right_0;
	for (int m = 0; m < sub_vec_size; m++)
	{
		
		for (int j = 0; j < word_vec_size; j++)
		{
			real b_lw_m_j = pack.delta1[m] * pack.xl[j];
			//**********verify grad***********
			/*w[l_idx] += eps;
			real f1 = forward(pack);
			cout << f1 << " ";
			w[l_idx] -= 2 * eps;
			real f2 = forward(pack);
			cout << f2 << endl;
			w[l_idx] += eps;
			cout << "verify grad_lw_" << m << "_" << j << ": "
				<< (f1 - f2) / (2 * eps) << " " << b_lw_m_j << endl;
			stop();*/
			//******************************
			w[l_idx] -= alpha * b_lw_m_j;

			real b_rw_m_j = pack.delta1[m] * pack.xr[j];
			//**********verify grad***********
			/*wr[w_idx] += eps;
			real f1 = forward(pack);
			cout << f1 << " ";
			wr[w_idx] -= 2 * eps;
			real f2 = forward(pack);
			cout << f2 << endl;
			wr[w_idx] += eps;
			cout << "verify grad_rw_" << m << "_" << j << ": "
				<< (f1 - f2) / (2 * eps) << " " << b_rw_m_j << endl;
			stop();*/
			//******************************
			w[r_idx] -= alpha * b_rw_m_j;

			l_idx++;
			r_idx++;
		}
	}

	
	//update word_vec

		for (int j = 0; j < word_vec_size; j++)
		{
			if (update_word == false) break; 
			real g_xl_i_j = 0;
			real g_xr_i_j = 0;
			unsigned int l_idx = w_left_0;
			unsigned int r_idx = w_right_0;
			for (int m = 0; m < sub_vec_size; m++)
			{
				g_xl_i_j += pack.delta1[m] * w[l_idx + j];
				g_xr_i_j += pack.delta1[m] * w[r_idx + j];
				l_idx += word_vec_size;
				r_idx += word_vec_size;
			}
			for (size_t i = 0; i < pack.fid; i++)
			{
				//**********verify grad***********
				/*pack.x[i][j] += eps;
				real f1 = forward(pack);
				cout << f1 << "-";
				pack.x[i][j] -= 2 * eps;
				real f2 = forward(pack);
				cout << f2 << endl;
				pack.x[i][j] += eps;
				cout << "verify grad_x_" << i << "_" << j << ": "
					<< (f1 - f2) / (2 * eps) << " " << g_xl_i_j << endl;
				stop();*/
				//*****************************
				pack.x[i][j] -= alpha*g_xl_i_j;
			}

			for (size_t i = pack.fid + 1; i < pack.x.size(); i++)
			{
				//**********verify grad***********
				/*pack.x[i][j] += eps;
				real f1 = forward(pack);
				cout << f1 << "-";
				pack.x[i][j] -= 2 * eps;
				real f2 = forward(pack);
				cout << f2 << endl;
				pack.x[i][j] += eps;
				cout << "verify grad_x_" << i << "_" << j << ": "
					<< (f1 - f2) / (2 * eps) << " " << g_xr_i_j << endl;
				stop();*/
				//*****************************
				pack.x[i][j] -= alpha*g_xr_i_j;
			}

		}


	line_trained++;
	
}

void LRRewrite::sgdTrain(nn_pack& pack, const string& line, int local_num)
{
	if (pack.freq > 100) pack.freq = 100;
	pack.freq = sqrt(pack.freq);
	//pack.freq=pack.freq*0.1;

	int print_step = 10000;

	//forward:
	forward(pack);
	float t1 = pack.a2[pack.label];
	if (t1 > 1.0){
		if (local_num%print_step == 0){
			cerr << line << "\t" << t1 << "\tdone" << endl;
		}
		return;
	}
	//backward:
	backward(pack);
	if (local_num%print_step == 0){
		forward(pack);
		cerr << line << "\t" << t1 << "\t" << pack.a2[pack.label] << endl;
	}

	
}


void LRRewrite::PrintModel(){
	cerr << "################### line_cnt:" << line_cnt << endl;
	cerr << "line_trained:" << line_trained << endl;
	cerr << "max_iter_num:" << max_iter_num << endl;

	sub_num = sub_index.size();
	cout << "#subvec\t" << sub_num << "\t" << sub_vec_size << endl;
	cout << "#w_num\t1" << endl;
	cout << "#b_num\t1" << endl;
	for (map<string, int>::iterator it = sub_index.begin(); it != sub_index.end(); it++)
	{
		real * p = sub_vec + it->second*sub_vec_size;
		cout << label << "\t1\t" << it->first;
		for (int i = 0; i < sub_vec_size; i++){
			cout << "\t" << p[i];
		}
		cout << endl;

	}
	cerr << "print subvec done" << endl;


	cout << label << "\t2\t";
	for (int wi = 0; wi < w_size; wi++)
	{
		if (wi>0) cout << "\t";
		cout << w[wi];
	}
	cout << endl;


	cerr << "print w done" << endl;

	cout << label << "\t3\t";
	for (int m = 0; m < sub_vec_size; m++)
	{
		if (m>0) cout << "\t";
		cout << b[m];
	}
	cout << endl;

	cerr << "print b done" << endl;

}
