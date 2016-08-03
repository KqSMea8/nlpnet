/*
 * ConvRewrite.cpp
 *
 *  Created on: 2015-6-15
 *      Author: zengzengfeng
 */

#include "ConvRewrite.h"

ConvRewrite::ConvRewrite() {
    // TODO Auto-generated constructor stub
    max_conv_num=20;
    window_size=3;
    cerr << "ConvRewrite constructor:" << endl;
}

ConvRewrite::~ConvRewrite() {
    // TODO Auto-generated destructor stub
}

void ConvRewrite::InitPack(nn_pack& pack)
{
    //cerr << "ConvRewrite::InitPack" << endl;
    pack.hidden = vector<real>(hidden_size,0);
    pack.max_pos = vector<int>(hidden_size,0);
    for(size_t i=0; i<max_conv_num; i++){
        pack.conv.push_back(vector<real>(hidden_size,0));
    }
}
void ConvRewrite::convFunc(vector<real*>& x,size_t begin,size_t end, 
        vector<real>& out, nn_pack& pack){
    for(int i=0; i<hidden_size; i++){
        out[i]=0;
        for(size_t j=begin; j<= end; j++){
            //if(j==pack.fid) continue;
            out[i]+=x[j][i];
        }
    }
}
real ConvRewrite::forward(nn_pack& pack) {

    //convolution
    //cerr << "ConvRewrite::forward" << endl;
    int k=0;
    for(size_t i=0; i<pack.x.size(); i++){
        size_t end=window_size+i-1;
        if(end >= pack.x.size()) break;
        convFunc(pack.x, i, end, pack.conv[k],pack);
        k+=1;
    }
    pack.conv_num=k;

    //max pooling
    for(int i=0; i< hidden_size; i++){
        float max=pack.conv[0][i];
        int maxj=0;
        for(int j=1; j<pack.conv_num; j++){
            if(pack.conv[j][i]>max){
                maxj=j;
                max=pack.conv[j][i];
            }
        }
        pack.hidden[i]=max;
        pack.max_pos[i]=maxj;
    }

    pack.z2.clear();
    pack.a2.clear();

    for (size_t i = 0; i < pack.u.size(); i++) {
        real z = 0.0;
        for (size_t j = 0; j < hidden_size; j++) {
            z += pack.u[i][j] * pack.hidden[j];
        }
        pack.z2.push_back(z);
    }
    softmax(pack.z2, pack.a2);

    return -log(pack.a2[pack.label]);

}


void ConvRewrite::backward(nn_pack& pack) {

    pack.delta2 = pack.a2;


    pack.delta2[pack.label] = pack.delta2[pack.label] - 1.0;
    pack.grad_x.clear();
    pack.grad_u.clear();

    float beta=0.001;

    beta=alpha*pack.freq/10;
    pack.freq=1.0;
    beta=0.0;
    alpha=0.001;
    //update sub_vec
    for (size_t i = 0; i < pack.u.size(); i++) {
        for (size_t j = 0; j < hidden_size; j++) {
            float g_u= (alpha*pack.freq*pack.delta2[i] * pack.hidden[j] + beta*pack.u[i][j]);
            pack.u[i][j] = pack.u[i][j] - g_u;
            //pack.u[i][j] = pack.u[i][j] - alpha*pack.delta2[i] * pack.hidden[j];
            pack.grad_u.push_back(pack.delta2[i] * pack.hidden[j]);
        }
    }

    //update word_vec
    for (size_t i = 0; i < hidden_size; i++) {
        pack.hidden[i] = 0;
        for (size_t k = 0; k < pack.delta2.size(); k++) {
            pack.hidden[i] += pack.delta2[k] * pack.u[k][i]; //save the gradient of hidden
        }
    }

    for(size_t i =0; i< pack.x.size(); i++){
        for(int j=0; j< word_vec_size; j++){
            pack.grad_x.push_back(0);
        }
    }

    for (size_t j = 0; j < hidden_size; j++) {
        int pos=pack.max_pos[j];
        for(int i=pos; i<pos+window_size; i++){
            //if(i==pack.fid) continue;
            float g_x = pack.freq*alpha*pack.hidden[j] + beta*pack.x[i][j];
            pack.x[i][j] = pack.x[i][j] - g_x;
            pack.grad_x[i*word_vec_size+j]+=pack.hidden[j];
        }
    }




}
