/*
 * ConcRewrite.cpp
 *
 *  Created on: 2015-6-15
 *      Author: zengzengfeng
 */

#include "ConcRewrite.h"

ConcRewrite::ConcRewrite() {
    // TODO Auto-generated constructor stub
    cerr << "ConcRewrite constructor:" << endl;
    word_vec_size = 50;
    sub_vec_size = 200;
    window_size = 5;
    thread_num = 1;

    vocab_size=0;
    line_num=0;
    alpha=0.01;

}

ConcRewrite::~ConcRewrite() {
    // TODO Auto-generated destructor stub
}
void ConcRewrite::InitNet() {

    NNRewrite::InitNet();

}


real ConcRewrite::forward(nn_pack& pack) {


#ifdef _DEBUG
    //cerr<< "ConcRewrite::forward" << endl;
#endif


    pack.z2.clear();
    pack.a2.clear();
    for (size_t i = 0; i < pack.u.size(); i++) {
        real z = 0.0;
        int uid=0;
        for (size_t j = pack.fid-2; j <= pack.fid+2; j++) {
            if(j==pack.fid) continue;
            for(size_t k=0; k<word_vec_size; k++){
                z += pack.u[i][uid] *pack.x[j][k];
                uid+=1;
            }
        }
        pack.z2.push_back(z);
    }

    //cout << "pack.z2:";
    //print_vec(pack.z2);
    softmax(pack.z2, pack.a2);

    return -log(pack.a2[pack.label]);
}


void ConcRewrite::backward(nn_pack& pack) {

    pack.delta2 = pack.a2;
    pack.delta2[pack.label] = pack.delta2[pack.label] - 1.0;
    pack.grad_x.clear();
    pack.grad_u.clear();
    float beta=0.0;
    //pack.freq=1.0;
    if(pack.freq>100) pack.freq=100;
    pack.freq=sqrt(pack.freq);


    //update sub_vec
    for (size_t i = 0; i < pack.u.size(); i++) {
        for (size_t j = 0; j < sub_vec_size; j++) {
            int wid=j/word_vec_size;
            int k=j%word_vec_size;
            if(wid>=pack.fid) wid+=1;
            float g_u= alpha*pack.freq * pack.delta2[i] * pack.x[wid][k] + beta*pack.u[i][j];
            pack.u[i][j] = pack.u[i][j] - g_u;
            pack.grad_u.push_back(pack.delta2[i] * pack.x[wid][k]);
        }
    }

    return;
    //update word_vec
    for (int i = pack.fid - 2; i <= pack.fid + 2; i++) {
        if (i == pack.fid){
            for (int j = 0; j < word_vec_size; j++) pack.grad_x.push_back(0);
            continue;
        }
        for (int j = 0; j < word_vec_size; j++) {
            int a=i;
            if(a>pack.fid) a=a-1;
            int uid = word_vec_size*a+j;
            float tmp_g=0.0;
            for (size_t k = 0; k < pack.u.size(); k++) {
                float g_x = alpha*pack.freq * pack.delta2[k] * pack.u[k][uid]
                    + beta * pack.x[i][j];
                pack.x[i][j] = pack.x[i][j] - g_x;
                tmp_g+=pack.delta2[k] * pack.u[k][uid];
            }
            pack.grad_x.push_back(tmp_g);
        }
    }

}
