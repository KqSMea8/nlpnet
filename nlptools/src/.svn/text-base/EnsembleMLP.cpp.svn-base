/*
 * EnsembleMLP.cpp
 *
 *  Created on: 2015-7-1
 *      Author: zengzengfeng
 */

#include "../include/EnsembleMLP.h"
#include "../include/NNRewrite.h"
EnsembleMLP::EnsembleMLP() {
    // TODO Auto-generated constructor stub
    model_num=1000;
}

EnsembleMLP::~EnsembleMLP() {
    // TODO Auto-generated destructor stub
    for(int i=0; i<model_num; i++){
        if(w_vector[i] != NULL){
            delete w_vector[i];
        }
        if(b_vector[i] != NULL){
            delete b_vector[i];
        }
    }
}

void EnsembleMLP::InitNet() {

    expTable = new real[EXP_TABLE_SIZE + 1];
    word_vec = new real[vocab_size * word_vec_size];
    cerr<< " InitNet sub_num:" << sub_num << endl;
    sub_vec = new real[sub_num * sub_vec_size];

    for(int i=0; i<model_num; i++){
       real* pw = new real[word_vec_size * sub_vec_size * (window_size - 1)];
       real* pb = new real[sub_vec_size];
       w_vector.push_back(pw);
       b_vector.push_back(pb);
    }

    expTable = (real *)malloc((EXP_TABLE_SIZE + 1) * sizeof(real));
    for (int i = 0; i < EXP_TABLE_SIZE; i++) {
      expTable[i] = exp((i / (real)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP); // Precompute the exp() table
    }
}

void EnsembleMLP::LoadModel(const string& infile) {

    ifstream fin1(infile.c_str());


    string line;
    vector<string> vec;
    vector<string> str_tmp;

    getline(fin1, line);
    split_str(vec, line, "\t");
    vocab_size= atoi(vec[1].c_str());
    word_vec_size= atoi(vec[2].c_str());


    getline(fin1, line);
    split_str(vec, line, "\t");
    sub_num = atoi(vec[1].c_str());
    sub_vec_size = atoi(vec[2].c_str());

    getline(fin1, line);
    split_str(vec, line, "\t");
    int w_num = atoi(vec[1].c_str());

    getline(fin1, line);
    split_str(vec, line, "\t");
    int b_num = atoi(vec[1].c_str());

    cerr << "vocab_size:" << vocab_size << "sub_num:" << sub_num << endl;
    cerr << "word_vec_size:" << word_vec_size << endl;
    cerr << "sub_vec_size:" << sub_vec_size << endl;

    InitNet();

    int wid = 0;    //load subvec
    int sid = 0;
    int begin = 0;
    string psk = "";
    string key = "";
    int sub_end=sub_num+vocab_size;
    int w_end=sub_num+vocab_size+w_num;
    int b_end=sub_num+vocab_size+w_num+b_num;
    while (getline(fin1, line)) {
        split_str(vec, line, "\t");
        if (wid < vocab_size) {
            vocab.push_back(vec[0]);
            word_index[vec[0]] = wid;
            for (size_t i = 0; i < word_vec_size; i++) {
                word_vec[wid * word_vec_size + i] = atof(vec[i + 2].c_str());
            }
        }

        if (wid>= vocab_size && wid < sub_end) {
             for (size_t i = 0; i < vec.size() - 3; i++) {
                 sub_vec[sid * sub_vec_size + i] = atof(vec[i + 3].c_str());
             }
             if (vec[1] != psk) {
                 if (sid > 0) {
                     pair<int, int> p(begin, sid - 1);
                     sub_list[psk] = p;
                 }
                 psk = vec[1];
                 begin = sid;
             }
             sub_label[vec[1]]=atoi(vec[0].c_str());
             key = vec[1] + "\t" + vec[2];
             sub_index[key] = sid;
             sub_string.push_back(key);
             sid+=1;
         }
         if(wid >= sub_end && wid < w_end){
             int label=atoi(vec[0].c_str());
             size_t wlen=word_vec_size * sub_vec_size * (window_size - 1);
             if(vec.size()<wlen+1){
                 cerr << "######## sub the w fail" << endl;
                 cerr << "vec.size:"<< vec.size() << endl;
                 cerr << word_vec_size << "\t" << hidden_size << endl;
                 return;
             }
             for(size_t i=0; i< wlen; i++){
                 w_vector[label][i]=atof(vec[i+1].c_str());
                 //if(i<10) cout << w_vector[label][i] << endl;
             }
         }
         if(wid >= w_end){
             int label=atoi(vec[0].c_str());
             if(vec.size()< hidden_size+1){
                 cerr << "######## sub the b fail" << endl;
                 return;
             }
             for(size_t i=0; i<  hidden_size; i++){
                 b_vector[label][i]=atof(vec[i+1].c_str());
             }
         }

        wid += 1;

    }

    //the last one
    pair<int, int> p(begin, sid - 1);
    sub_list[psk] = p;


    cerr << "load the model done" << endl;

    return ;

}

void EnsembleMLP::Predict(vector<string>& query, size_t index,
    map<string, real>& score, nn_pack& pack, int type) {


    if(sub_label.find(query[index])==sub_label.end()){
        return;
    }
    int label=sub_label[query[index]];
    w=w_vector[label];
    b=b_vector[label];
    MLPRewrite::Predict(query,index,score,pack,type);
}
