/*
 * testRWVector.cpp
 *
 *  Created on: 2014-7-6
 *      Author: zengzengfeng
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#ifdef __GNUC__ 
#include <ext/hash_map>
namespace std{
	using namespace __gnu_cxx;
}
#else
#include <hash_map>
#pragma comment(lib,"x86/pthreadVC2.lib")
#endif

#include <iostream>
#include <sstream>
#include <fstream>
#include <ostream>
#include <algorithm>

#include "utils.h"
#include "NNRewrite.h"
#include "ConcRewrite.h"
#include "MLPRewrite.h"
#include "ConvRewrite.h"
#include "EnsembleMLP.h"
#include "LRRewrite.h"
#include "LRAdvRewrite.h"
using namespace std;

NNRewrite* pModel;
int num_threads=1;
int word_vec_size=100;
int max_line_num=1000000000;
string train_file;
string out_file;
string model_file;
string sub_file;
string word_file;
string word_vec_file;
int gIterNum=1;
long long file_size=0;

typedef pair<string, real> PAIR;
bool cmp_by_value(const PAIR& lhs, const PAIR& rhs) {
    return lhs.second > rhs.second; 
}

typedef pair<string, pair<real,real> > PAIR2;
bool cmp_by_value2(const PAIR2& lhs, const PAIR2& rhs) {
    return lhs.second.first > rhs.second.first;
}


void *TrainModelThread(void *id) {
	pModel->Train((long) id, train_file,file_size);
	return NULL;
}


void TrainModel() {

    pthread_t *pt = (pthread_t *) malloc(num_threads * sizeof(pthread_t));
    //printf("Starting training using file %s\n", train_file.c_str());
    ifstream fin1(train_file.c_str());
    fin1.seekg(0,ios::end);
    file_size = fin1.tellg();
    fin1.close();
    pModel->LoadDicts(sub_file,word_file);
	pModel->SetThreadNum(num_threads);
    for(int i=0; i < gIterNum; i++) {
        cerr << "###################### iter_num:" << i << endl;
        pModel->SetIterNum(i,gIterNum);
	    for (int a = 0; a < num_threads; a++){
		    pthread_create(&pt[a], NULL, TrainModelThread, (void *) a);
        }

	    for (int a = 0; a < num_threads; a++){
		    pthread_join(pt[a], NULL);
        }
	    pModel->SampleLoss();
    }

}

void HdpTrain()
{

    string line;
    string key;
    vector<string> vec;
    map<string,int> sub_cnt;
    sub_file="NNRewrite.subdict";
    train_file="NNRewrite.tmp";
    ofstream out(sub_file.c_str());
    ofstream tmpout(train_file.c_str());
    cerr << "########### hdpTrain" << endl;
    int line_num=0;
    while (getline(cin, line))
    {
        if(line_num%1000000==0){
            cerr << line_num << "\t" << line << endl;
        }
        if(line_num > max_line_num) continue;
        line_num+=1;
        split_str(vec, line, "\t");
        key=vec[0]+"\t"+vec[1];
        map<string, int>::iterator iter=sub_cnt.find(key);
        if(iter==sub_cnt.end()){
            sub_cnt[key]=1;
        }else{
            iter->second+=1;
        }
        tmpout << line << endl;
    }
    tmpout.flush();
    tmpout.close();

    cerr << "save train done, line_num:" << line_num << endl;
    if(line_num==0){
        cerr << "############ empty input " << endl;
        return;
    }
    for (map<string, int>::iterator it = sub_cnt.begin();
            it != sub_cnt.end(); it++) {
        out << it->first << endl;
    }
    out.flush();
    out.close();
    pModel->SetInputType(TEXT);
    cerr << "save subdict done" << endl;
    TrainModel();
    pModel->PrintModel();
    return;
}

void sort_map(map<string,real>& dict, vector<PAIR>& kv)
{
    kv.clear();
    kv.insert(kv.begin(), dict.begin(), dict.end());
    sort(kv.begin(), kv.end(),cmp_by_value);
}

void test1(){

    pModel->LoadModel(model_file);
    string line;
    vector<string> query;
    ostringstream oss;
    map<string,real> sub_dict;
    map<string,real> z2_dict;
    vector<PAIR> kv;
    nn_pack pack;
    pModel->InitPack(pack);
    cout << "################ input query:" << endl;
    while (getline(cin, line)) 
    {
         oss.str("");
         //oss<< "L2 L1 " << line << " R1 R2" << endl;
         oss << line;
         query.clear();
         cout << "OSS######" << oss.str() << endl;
         split_str(query, oss.str(), " ");
         for(size_t i=0; i<query.size(); i++){
            sub_dict.clear();
            z2_dict.clear();
            cout <<"#### word:" << query[i] << endl;
            if(query[i]=="L1" || query[i]=="L2" || query[i]=="R1" || query[i]=="R2")
            {
                continue;
            }
            pModel->Predict(query,i,sub_dict,pack,0);
            pModel->Predict(query,i,z2_dict,pack,1);
            kv.clear();
            kv.insert(kv.begin(), sub_dict.begin(), sub_dict.end());
            sort(kv.begin(), kv.end(),cmp_by_value);
            for(size_t j=0; j<kv.size(); j++){
                cout << query[i] << " " << kv[j].first << " " 
                    << kv[j].second << "\t"<< z2_dict[kv[j].first] << endl; 
            }
         }
         cout << "################ input query:" << endl;
    }

}



void test2() {

    pModel->LoadModel(model_file);
    string line;
    vector<string> query;
    ostringstream oss;
    map<string, pair<real, real> > sub_dict;
    vector<PAIR2> kv;
    nn_pack pack;
    pModel->InitPack(pack);
    cout << "################ input query:" << endl;
    while (getline(cin, line)) {
        oss.str("");
        //oss<< "L2 L1 " << line << " R1 R2" << endl;
        oss << line;
        query.clear();
        cout << "OSS######" << oss.str() << endl;
        split_str(query, oss.str(), " ");
        for (size_t i = 0; i < query.size(); i++) {
            for (size_t j = 0; j < 3; j++) {
                if (i + j >=query.size()) break;
                sub_dict.clear();
                cout <<"#### word:" << query[i] << endl;
                pModel->Predict2(query, i, i + j, sub_dict, pack);
                kv.clear();
                kv.insert(kv.begin(), sub_dict.begin(), sub_dict.end());
                sort(kv.begin(), kv.end(), cmp_by_value2);
                for (size_t j = 0; j < kv.size(); j++) {
                    cout << kv[j].first << "\t" << kv[j].second.first 
                        << "\t" << kv[j].second.second << endl;
                }
            }
        }
        cout << "################ input query:" << endl;
    }

}


void evalfun2() {

    pModel->LoadModel(model_file);
    string line;
    vector<string> items; 
    vector<string> query;

    ostringstream oss;
    map<string, pair<real, real> > sub_dict;
    vector<PAIR2> kv;

    nn_pack pack;
    pModel->InitPack(pack);

    while (getline(cin, line)) {
        split_str(items,line,"\t");
        query.clear();
        split_str(query, items[0], " ");

        sub_dict.clear();
        int begin=atoi(items[3].c_str());
        int end=atoi(items[4].c_str());

        if(query.size()>20 || begin<0){
            cout<< line << "\t0\t0\t10000\t0" << endl;
            continue;
        }

        pModel->Predict2(query, begin, end, sub_dict, pack);

        kv.clear();
        kv.insert(kv.begin(), sub_dict.begin(), sub_dict.end());
        sort(kv.begin(), kv.end(),cmp_by_value2);

        string key=items[1]+"\t"+items[2];
        real self_score=0.0;
        string str1=items[1]+"\t"+items[1];
        if(sub_dict.find(str1) != sub_dict.end()){
             self_score=sub_dict[str1].first;
        }

        int flag=0;
        for(size_t a=0; a<kv.size(); a++){
            if(key==kv[a].first){
                float ratio=0.0;
                if(self_score>0.0) ratio=kv[a].second.first/self_score;
                cout << line << "\t" << kv[a].second.first << "\t" 
                    << kv[a].second.second  << "\t" << a << "\t" << ratio << endl;
                flag=1;
                break;
            }
        }

        if(flag==0) cout<< line << "\t0\t0\t10000\t0" << endl;
    }

}

void evalfun(){

    pModel->LoadModel(model_file);
    string line;
    vector<string> query;
    ostringstream oss;
    map<string,real> sub_dict;
    map<string,real> z2_dict;
    vector<PAIR> kv;
    vector<string> items;
    nn_pack pack;
    pModel->InitPack(pack);
    while (getline(cin, line)) 
    {
         split_str(items,line,"\t");
         oss.str("");
         //oss<< "L2 L1 " << items[0] << " R1 R2";
         oss<< items[0];

         query.clear();
         split_str(query, oss.str(), " ");

         sub_dict.clear();
         z2_dict.clear();
         //int i=atoi(items[3].c_str())+2;
         float self_score=0.0;
         int i=atoi(items[3].c_str());
         if(query.size()>20 || i<0){
             cout<< line << "\t0\t0\t10000\t" << self_score  << endl;
             continue;
         }

         pModel->Predict(query,i,sub_dict,pack,0);
         pModel->Predict(query,i,z2_dict,pack,1);

         string str1=query[i]+"\t"+query[i];
         if(sub_dict.find(str1) != sub_dict.end()){
             self_score=sub_dict[str1];
         }

         kv.clear();
         kv.insert(kv.begin(), sub_dict.begin(), sub_dict.end());
         sort(kv.begin(), kv.end(),cmp_by_value);
         string key=items[1]+"\t"+items[2];
         int flag=0;
         for(size_t a=0; a<kv.size(); a++){
            if(key==kv[a].first){
                cout << line << "\t" << kv[a].second << "\t" 
                     << z2_dict[kv[a].first] << "\t" << a << "\t" << self_score << endl;
                flag=1;
                break;
            }
         }
         if(flag==0) cout<< line << "\t0\t0\t10000\t" << self_score  << endl;
    }
    cerr << "eval done" << endl;

}


void expand(){

    pModel->LoadModel(model_file);
    string line;
    vector<string> items;
    vector<string> query;
    map<string,real> sub_dict;

    vector<PAIR> kv;
    nn_pack pack;
    pModel->InitPack(pack);
    cout << "################ input query1 and query2:" << endl;
    while (getline(cin, line)) {
        split_str(items, line, "\t");
        split_str(query, items[0], " ");
        sub_dict.clear();
        pModel->Expand(query, items[1], sub_dict, pack);
        kv.clear();
        kv.insert(kv.begin(), sub_dict.begin(), sub_dict.end());
        sort(kv.begin(), kv.end(), cmp_by_value);
        for (size_t j = 0; j < kv.size(); j++) {
            if(kv[j].second<0.1) continue;
            cout <<items[0] << "\t" << kv[j].first << "\t" << kv[j].second << "\t" << j << endl;
            if(j>20) break;
        }
        //cout << "################ input query1 and query2:" << endl;
    }

}




int main(int argc, char **argv) {
	int i;
	if (argc == 1) {
		printf("Options:\n");
		printf("Parameters for training:\n");
		printf("\t-train <file>\n");
		printf("\t\tUse text data from <file> to train the model\n");
		printf("\t-sub <file>\n");
		printf("\t\tsub file \n");
	    printf("\t-word <file>\n");
	    printf("\t\t load the word \n");
	    printf("\t-word-vec <file>\n");
	    printf("\t\t load the word-vec \n");

		printf("\t-output <file>\n");
		printf("\t\tUse <file> to save the resulting word vectors \n");
		printf("\t-threads <int>\n");
		printf("\t\tUse <int> threads (default 1)\n");
		printf("\t-iter <int>\n");
		printf("\t\tUse <int> iter (default 1)\n");
        printf("\t-mode <int>\n");
        printf("\t\t NNRewrite: 0 ConcRewrite: 1 (default 0)\n");
        printf("\t-check <int>\n");
        printf("\t\t gradient-check (default 0)\n");
        printf("\t-model-file <file>\n");
        printf("\t\t load the model to predict \n");
        printf("\t-hdp <file>\n");
        printf("\t\t run on the hadoop \n");
        printf("\t-test\n");
        printf("\t\t test the input \n");
        printf("\t-max-line\n");
        printf("\t\t set the max_line_num \n");
        return 0;
	}

	int mode=0;
	int check=0;
    int test=0;
    int hdp=0;
    model_file="";
    word_file="";
    word_vec_file="";
	if ((i = ArgPos((char *)"-size", argc, argv)) > 0) word_vec_size = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-train", argc, argv)) > 0) train_file=argv[i + 1];
	if ((i = ArgPos((char *)"-sub", argc, argv)) > 0) sub_file=argv[i + 1];
	if ((i = ArgPos((char *)"-word", argc, argv)) > 0) word_file=argv[i + 1];
	if ((i = ArgPos((char *)"-word-vec", argc, argv)) > 0) word_vec_file=argv[i + 1];
	if ((i = ArgPos((char *)"-output", argc, argv)) > 0) out_file=argv[i + 1];
	if ((i = ArgPos((char *)"-threads", argc, argv)) > 0) num_threads=atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-iter", argc, argv)) > 0) gIterNum =atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-mode", argc, argv)) > 0) mode =atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-model-file", argc, argv)) > 0) model_file=argv[i + 1];
	if ((i = ArgPos((char *)"-check", argc, argv)) > 0) check =atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-test", argc, argv)) > 0) test =atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-hdp", argc, argv)) > 0) hdp =atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-max-line", argc, argv)) > 0) max_line_num =atoi(argv[i + 1]);
    cerr<< "max_line_num: " << max_line_num << endl;
	if(mode == 0){
	    pModel = new NNRewrite();
	}else if(mode == 1 ){
	    pModel = new ConcRewrite();
	}else if(mode == 2){
	    pModel = new MLPRewrite();
	}else if(mode ==3){
	    pModel = new ConvRewrite();
	}else if(mode == 5){
        pModel = new  LRRewrite();
    }else if (mode == 6)
    {
		pModel = new LRAdvRewrite();
    }
    else{
	    cerr<< " set the mode" << endl;
	    return 0;
	}
    if(test==1){
        test1();
        return 0;
    }
	if(check==1){
	    pModel->GradientCheck();
	    return 0;
	}
    if(test==2){
        evalfun2();
        return 0;
    }

    if(test==3){
        expand();
        return 0;
    }

    if(hdp==1){
       HdpTrain();
    }else{
       TrainModel();
       pModel->SaveModel(out_file);
    }
	return 0;
}

