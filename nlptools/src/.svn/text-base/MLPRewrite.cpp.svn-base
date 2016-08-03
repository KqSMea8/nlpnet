/*
 * MLPRewrite.cpp
 *
 *  Created on: 2015-6-15
 *      Author: kegeyang
 */

#include "MLPRewrite.h"
#include <cmath>

MLPRewrite::MLPRewrite()
{
	// TODO Auto-generated constructor stub
	cerr << "MLPRewrite constructor:" << endl;
	word_vec_size = 50;
	sub_vec_size = 50;
	window_size = 5;
	thread_num = 1;

	vocab_size = 0;
	line_num = 0;
	alpha = 0.025;
	line_trained = 0;

	expTable = NULL;
	word_vec = NULL;
	sub_vec = NULL;
	w_size=sub_vec_size*word_vec_size*(window_size-1);
    srand( (unsigned)time( NULL ) );
	//w1 = NULL;

}

MLPRewrite::~MLPRewrite() {
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



void MLPRewrite::softmax2(vector<real>& z, vector<real>& a )
{
    real max_val=-1e12;
    for(size_t i=0; i< z.size(); i++){
        if(z[i]>max_val) max_val=z[i];
    }
    real sum=0.0;
    for(size_t i=0; i< z.size(); i++){
        //real v1=exp(z[i]-max_val);
        real v1=sigmod(z[i]);
        //real v1=fast_exp(z[i]-max_val);
        //cout<< "v1:" << v1 <<" v2: " << v2 << endl;
        a.push_back(v1);
        sum+=v1;
     }
    /*for(size_t i=0; i< a.size(); i++){
        a[i]=a[i]/sum;
    }*/
}

real MLPRewrite::forward(nn_pack& pack) {


#ifdef _DEBUG
	//cerr<< "MLPRewrite::forward" << endl;
#endif

	pack.z2.clear();
	pack.a2.clear();
	pack.a3.clear();
	pack.hidden.clear();

	size_t wid = 0;
	for (size_t i = 0; i < sub_vec_size; i++)
	{
		size_t xid = pack.fid - 2;
		pack.hidden.push_back(b[i]);
		for (size_t j = 0; j < window_size - 1; j++)
		{
			pack.hidden[i] += inner(&w[wid], pack.x[xid], word_vec_size);
			wid += word_vec_size;
			xid++;
			if (xid == pack.fid)
			{
				xid++;
			}
			//print_pack(pack);
		}

		pack.hidden[i] = acti(pack.hidden[i]);

			//print_pack(pack);

	}
	//stop();


	for (size_t i = 0; i < pack.u.size(); i++) {
		real z = inner(pack.u[i], pack.hidden);
		pack.z2.push_back(z);
	}

	softmax(pack.z2, pack.a2);

    for(size_t i=0; i< pack.z2.size(); i++)
    {
        pack.a3.push_back( sigmod(pack.z2[i]) );
    }

    if( pack.label >= pack.a2.size()){
        cerr << "############# forward err" << endl;
        return -1.0;
    }

	return -log(pack.a2[pack.label]);
}


void cal_delta2(nn_pack& pack)
{
	pack.delta2 = pack.a3;
    float ratio=1.0/pack.delta2.size();
    ratio=ratio*3;
    if(ratio>1.0) ratio=1.0;
    
    for(size_t i=0; i< pack.delta2.size(); i++)
    {
        if(i==pack.label) continue;
        pack.delta2[i] = pack.a3[i]*ratio;
    }
	pack.delta2[pack.label] = pack.delta2[pack.label] - 1;

    /*for(size_t i=0; i< pack.delta2.size(); i++){
        if(i==pack.label) continue;
        real randnum=rand() / (real) RAND_MAX;
        if(randnum > pack.freq)  pack.delta2[i]=0;
    }*/

}

void cal_delta(nn_pack& pack)
{
	pack.delta2 = pack.a2;
	pack.delta2[pack.label] = pack.delta2[pack.label] - 1;
    //pack.delta2[pack.label] += (pack.a3[pack.label]-pack.a2[pack.label])*0.1;
    /*for(size_t i=0; i< pack.delta2.size(); i++){
        pack.delta2[i] += (pack.a3[i]-pack.a2[i])*0.1;
    }*/

}
void MLPRewrite::backward(nn_pack& pack)
{

	real eps = 1.0e-4;
	//pack.delta2 = pack.a2;
	//pack.delta2[pack.label] = pack.delta2[pack.label] - 1.0;

    /*float ratio1=0.01;
    float ratio2=1.0/pack.delta2.size();
    ratio2=ratio2*3;
    if(ratio2>1.0) ratio2=1.0;
    ratio2=1.0;
    for(size_t i=0; i< pack.delta2.size(); i++){
        if(i == pack.label){
            pack.delta2[i]+=ratio1*pack.a3[i];
        }else{
            pack.delta2[i]+=ratio1*ratio2*pack.a3[i];
        }
    }
	pack.delta2[pack.label] = pack.delta2[pack.label] - 1.0 -ratio1;*/
    cal_delta(pack);

	pack.grad_x.clear();
	pack.grad_u.clear();
	float beta = 0.0;
    int max=line_cnt*max_iter_num/2;
	alpha = 0.01*(1 - real(line_trained) / real(max + 1));
	if (alpha < 0.0001)
	{
		alpha = 0.0001;
	}
    if(iter_num==0) alpha =0.01;
	alpha = alpha*pack.freq;

	
	pack.delta1.clear();
	pack.grad_z1_OfHid.clear();

	for (int m = 0; m< sub_vec_size; m++)
	{
		pack.grad_z1_OfHid.push_back(acti_der(pack.hidden[m]));
	}


	for (int m = 0; m < sub_vec_size; m++)
	{
		pack.delta1.push_back(0.0);
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
	
	int wi;

	//update word_vec
	wi = 0;
	for (int i = pack.fid - 2; i <= pack.fid + 2; i++) {
        if(update_word==false) break;
		if (i == pack.fid){
			for (int j = 0; j < word_vec_size; j++) pack.grad_x.push_back(0);
			continue;
		}
		for (int j = 0; j < word_vec_size; j++)
		{
			real g_x_i_j = beta*pack.x[i][j];


			for (int m = 0; m < sub_vec_size; m++)
			{
				g_x_i_j +=
					pack.delta1[m] *
					w[m*word_vec_size*(window_size - 1) + wi + j];
			}

			//**********verify grad***********
			/*pack.x[i][j] += eps;
			real f1 = forward(pack);
			cout << f1 << "-";
			pack.x[i][j] -= 2 * eps;
			real f2 = forward(pack);
			cout << f2 << endl;
			pack.x[i][j] += eps;
			cout << "verify grad_x_" << i << "_" << j << ": "
			<< (f1 - f2) / (2 * eps) << " " << g_x_i_j - beta*pack.x[i][j] << endl;
			stop();*/
			//*****************************
			pack.x[i][j] -= alpha*g_x_i_j;
		}
		wi += word_vec_size;
	}

	//update w 
	wi = 0;
	for (int m = 0; m < sub_vec_size; m++)
	{
		for (int i = pack.fid - 2; i <= pack.fid + 2; i++) {
			if (i == pack.fid){
				continue;
			}
			for (int j = 0; j < word_vec_size; j++)
			{
				real g_w_m_i_j = pack.delta1[m] * pack.x[i][j] + beta*w[wi];
				
				//**********verify grad***********
				/*w[wi] += eps;
				real f1 = forward(pack);
				cout << f1 << " ";
				w[wi] -= 2 * eps;
				real f2 = forward(pack);
				cout << f2 << endl;
				w[wi] += eps;
				cout << w[wi] << endl;
				cout << "verify grad_w_" << wi << ": "
					<< (f1 - f2) / (2 * eps) << " " << g_w_m_i_j - beta*w[wi] << endl;
				stop();*/
				//******************************

				w[wi] -= alpha*g_w_m_i_j;
				wi++;
			}			
		}
	}

	//update b
	for (int m = 0; m < sub_vec_size; m++)
	{
		real g_b_m = beta * b[m] + pack.delta1[m];

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
		<< (f1 - f2) / (2 * eps) << " " << g_b_m - beta * b[m] << endl;
		stop();*/

		//******************************
		b[m] -= alpha*g_b_m;
	}





	//update sub_vec
	for (size_t k = 0; k < pack.u.size(); k++) {
		for (size_t i = 0; i < sub_vec_size; i++) {
			real g_u_k_i = pack.delta2[k] * pack.hidden[i] + beta*pack.u[k][i];
			//**********verify grad***********
			/*pack.u[k][i] += eps;
			real f1 = forward(pack);
			cout << f1 << "-";
			pack.u[k][i] -= 2 * eps;
			real f2 = forward(pack);
			cout << f2 << endl;
			pack.u[k][i] += eps;
			cout << "verify grad_u_" << k << "_" << i << ": "
			<< (f1 - f2) / (2 * eps) << " " << g_u_k_i - beta*pack.u[k][i] << endl;*/
			//stop();
			//******************************
			pack.u[k][i] -= alpha*g_u_k_i;
		}
	}

	line_trained++;
}


void MLPRewrite::sgdTrain(nn_pack& pack, const string& line, int local_num)
{
    //real randnum=rand() / (real) RAND_MAX;
    //if(randnum > pack.freq) return;

    if(pack.freq>100) pack.freq=100;
    pack.freq=sqrt(pack.freq);
    //pack.freq=pack.freq*0.1;

    int print_step=10000;

    //forward:
    forward(pack);
    float t1=pack.a2[pack.label];
    float t2=pack.a3[pack.label];
    if(t1>1.0){
        if(local_num%print_step==0){
            cerr << line << "\t" << t1 <<"\tdone" << endl;
        }
        return;
    }
    //backward:
    backward(pack);
    if(local_num%print_step==0){
        forward(pack);
        cerr << local_num << "\t" << line << "\t" << t1 << " --> " << pack.a2[pack.label] 
             << "\t" << t2 << " --> " << pack.a3[pack.label]<< endl;
    }

}

void MLPRewrite::print_pack(const nn_pack & pack) {
	for (size_t i = 0; i < pack.x.size(); i++)
	{
		if (i < pack.fid - window_size / 2 ||
			i > pack.fid + window_size / 2 ||
			i == pack.fid)
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






