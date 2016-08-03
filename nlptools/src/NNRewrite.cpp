/*
 * NNRewrite.cpp
 *
 *  Created on: 2014-7-11
 *      Author: zengzengfeng
 */

#include "NNRewrite.h"
#include "pthread.h"

using namespace std;

pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

NNRewrite::NNRewrite() {
    // TODO Auto-generated constructor stub
    word_vec_size = 50;
    sub_vec_size = 50;
    window_size = 5;
    thread_num = 1;
    min_word_freq = 0;
    hidden_size = sub_vec_size;

    vocab_size=0;
    line_num=0;
    line_cnt=0;
    line_trained=0;
    sub_num=0;
    input_type=TEXT;
    alpha=0.01;
    update_word=true;
    label=0;
    iter_num=0;
    max_iter_num=0;
    model_num=1000;
    w_size=0;
    //cerr << "NNRewrite constructor:" << endl;
}

NNRewrite::~NNRewrite() {
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

    for(int i=0; i<model_num; i++){
        if(w_vector[i] != NULL){
            delete w_vector[i];
        }
        if(b_vector[i] != NULL){
            delete b_vector[i];
        }
    }
    //if (w1 != NULL)
    //    delete w1;
    // w1 = NULL;
}
void NNRewrite::SetThreadNum(int threadNum) {
    thread_num = threadNum;
}
void NNRewrite::SetIterNum(int iter, int max_iter) {
    iter_num = iter;
    max_iter_num = max_iter;
}
void NNRewrite::SetInputType(int type){
    input_type=type;
}



void NNRewrite::LoadDicts(const string& subfile,const string& wordfile)
{
    loadSubDict(subfile);
    loadWordDict(wordfile);
}
void NNRewrite::loadWordDict(const string& wordfile) {
    string line;
    vector<string> vec;
    ifstream fin(wordfile.c_str());
    getline(fin, line);
    split_str(vec, line, "\t");
    vocab_size = atoi(vec[1].c_str());
    cerr << "############ vocab_size:" << vocab_size << endl;
    InitNet();
    cerr << "############ InitNet done" << endl;
    int wid = 0;
    int vid = 0;
    while (getline(fin, line)) {
        if (wid >= vocab_size)
            break;
        split_str(vec, line, "\t");
        vocab.push_back(vec[0]);
        word_index[vec[0]]=wid;
        if (vec.size() < word_vec_size + 2)
            continue;

        update_word = false;
        for (int i = 0; i < word_vec_size; i++) {
            word_vec[vid] = atof(vec[i + 2].c_str());
            vid += 1;
        }
        wid += 1;
        //cerr << vec[0] << " \t" << wid << endl;
    }

    fin.close();
    cerr << "############ load word done" << endl;
    cerr << "############ vocab.size:" << vocab.size() <<  " vid:" << vid << endl;
}

void NNRewrite::loadSubDict(const string& subfile)
{
    vector<string> vec;
    vector<string> words;
    string line;

    ifstream subf(subfile.c_str());
    string key;
    string psk="";
    int begin=0;
    int sid=0;
    while (getline(subf, line)) {
        split_str(vec, line, "\t");
        key=vec[0]+"\t"+vec[1];
        sub_string.push_back(key);
        sub_index[key]=sid;
        if(vec[0] != psk){
            pair<int,int> p(begin,sid-1);
            sub_list[psk]=p;
            psk=vec[0];
            begin=sid;
        }
        sid+=1;
    }
    pair<int,int> p(begin,sid-1);
    sub_list[vec[0]]=p;
    sub_num=sub_string.size();
    subf.close();
    cerr << "############ load sub done" << endl;

}

void NNRewrite::InitNet() {

    expTable = new real[EXP_TABLE_SIZE + 1];
    word_vec = new real[vocab_size * word_vec_size];
    //size_t sub_num = sub_list.size();
    //size_t sub_num = sub_index.size();
    cerr<< " InitNet sub_num:" << sub_num << endl;
    sub_vec = new real[sub_num * sub_vec_size];
    //w1 = new real[word_vec_size * sub_vec_size * window_size];


    for(int i=0; i<model_num; i++){
       //real* pw = new real[word_vec_size * sub_vec_size * (window_size - 1)];
       real* pw = new real[w_size];
       real* pb = new real[sub_vec_size];
       w_vector.push_back(pw);
       b_vector.push_back(pb);
    }

    w=w_vector[0];
    b=b_vector[0];

    for (long i = 0; i < vocab_size * word_vec_size; i++) {
        real rand_num = (rand() / (real) RAND_MAX - 0.5) / word_vec_size;
        word_vec[i] = rand_num;
    }
    for (long i = 0; i < sub_num * sub_vec_size; i++) {
        real rand_num = (rand() / (real) RAND_MAX - 0.5) / sub_vec_size;
        sub_vec[i]=rand_num;
    }
    for (long i = 0; i < w_size; i++) {
        real rand_num = (rand() / (real) RAND_MAX - 0.5) / word_vec_size;
        w[i] = rand_num;
    }

    for (int i = 0; i < sub_vec_size; i++){
       // b[i] = (rand() / (real)RAND_MAX - 0.5) / sub_vec_size;
       b[i] = 0;
    }
    expTable = (real *)malloc((EXP_TABLE_SIZE + 1) * sizeof(real));
    for (int i = 0; i < EXP_TABLE_SIZE; i++) {
      expTable[i] = exp((i / (real)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP); // Precompute the exp() table
    }
}
void NNRewrite::InitPack(nn_pack& pack)
{
    
    const unsigned int max_sub_num= 250;
    pack.hidden = vector<real>(hidden_size,0);
	pack.grad_z1.clear();
	for (int k = 0; k < max_sub_num; k++)
	{
		vector<real> tmp;
		for (int m = 0; m < sub_vec_size; m++)
		{
			tmp.push_back(0.0);
		}
		pack.grad_z1.push_back(tmp);
	}
}
void NNRewrite::word2index(vector<string>& words, vector<int>& wids) {


    //transform the word to index
    wids.clear();
    for (size_t i = 0; i < words.size(); i++) {
        map<string, int>::iterator iter = word_index.find(words[i]);
        if (iter == word_index.end()) {
            wids.push_back(0);
        } else {
            wids.push_back(iter->second);
        }
    }
}

real NNRewrite::fast_exp(real f){
    if(f >= MAX_EXP) return f;
    if(f <= -MAX_EXP) return EPSILON;
    return expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]+EPSILON;
}

void NNRewrite::softmax(vector<real>& z, vector<real>& a )
{
    real max_val=-1e12;
    for(size_t i=0; i< z.size(); i++){
        if(z[i]>max_val) max_val=z[i];
    }
    real sum=0.0;
    for(size_t i=0; i< z.size(); i++){
        real v1=exp(z[i]-max_val);
        //real v1=fast_exp(z[i]-max_val);
        //cout<< "v1:" << v1 <<" v2: " << v2 << endl;
        a.push_back(v1);
        sum+=v1;
     }
    for(size_t i=0; i< a.size(); i++){
        a[i]=a[i]/sum;
    }
}

void NNRewrite::getWordVec(vector<int>& wids, vector<real*>& x) {
    x.clear();
    for (size_t i = 0; i < wids.size(); i++) {
        real* xi = word_vec + word_vec_size * wids[i];
        x.push_back(xi);
    }
}
void NNRewrite::getWordVec(vector<int>& wids, vector<real*>& x, size_t begin, size_t end) {
    x.clear();
    for (size_t i = 0; i < wids.size(); i++) {
        if(i>begin && i<= end) continue;
        real* xi = word_vec + word_vec_size * wids[i];
        x.push_back(xi);
    }
}

void NNRewrite::getSubVec(pair<int, int>& range,vector<real* >& u){
    u.clear();
    for(int subid= range.first; subid<= range.second; subid++){
        real* sub = sub_vec+sub_vec_size*subid;
        u.push_back(sub);
    }
}


void NNRewrite::sumFunc(vector<real*>& x, vector<real>& hidden,  nn_pack& pack) {

    for (int c = 0; c < hidden_size; c++)
        hidden[c] = 0;
    for (size_t i = 0; i < x.size(); i++) {
        //if(i==pack.fid) continue;
        for (size_t j = 0; j < hidden_size; j++) {
            hidden[j] += x[i][j];
        }
    }

}

real NNRewrite::forward(nn_pack& pack) {
    pack.z2.clear();
    pack.a2.clear();
    sumFunc(pack.x, pack.hidden, pack);
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



void NNRewrite::backward(nn_pack& pack) {

    pack.delta2 = pack.a2;

    pack.delta2[pack.label] = pack.delta2[pack.label] - 1.0;
    pack.grad_x.clear();
    pack.grad_u.clear();
    float beta=0.0;
    int max=line_cnt*max_iter_num/2;
	alpha = 0.005*(1 - real(line_trained) / real(max + 1));
	if (alpha < 0.0001)
	{
		alpha = 0.0001;
	}
    if(iter_num==0) alpha =0.005;
    //beta=alpha*pack.freq/10;
    //update sub_vec
    for (size_t i = 0; i < pack.u.size(); i++) {
        for (size_t j = 0; j < hidden_size; j++) {
            float g_u= (alpha*pack.delta2[i] * pack.hidden[j]*pack.freq + beta*pack.u[i][j]);
            pack.u[i][j] = pack.u[i][j] - g_u;
            //pack.u[i][j] = pack.u[i][j] - alpha*pack.delta2[i] * pack.hidden[j];
            pack.grad_u.push_back(pack.delta2[i] * pack.hidden[j]);
        }
    }

    if(update_word==false) return;
    //update word_vec
    for (size_t i = 0; i < hidden_size; i++) {
        pack.hidden[i] = 0;
        for (size_t k = 0; k < pack.delta2.size(); k++) {
            pack.hidden[i] += pack.delta2[k] * pack.u[k][i]; //save the gradient of hidden
        }
    }

    for (size_t i = 0; i < pack.x.size(); i++) {
        //if(i==pack.fid) continue;
        for (size_t j = 0; j < word_vec_size; j++) {
            float g_x = pack.freq*alpha*pack.hidden[j] + beta*pack.x[i][j];
            pack.x[i][j] = pack.x[i][j] - g_x;
            //pack.x[i][j] = pack.x[i][j] - alpha*pack.hidden[j];
            pack.grad_x.push_back(pack.hidden[j]);
        }
    }

    line_trained++;

}

void NNRewrite::GradientCheck()
{
    nn_pack pack;
    //pack.hidden = vector<real>(hidden_size,0);
    InitPack(pack);
    pack.label=1;
    pack.fid=2;
    pack.freq=1;

    int word_num=5;
    sub_num=4;
    InitNet();

    real * x = new real[word_vec_size*word_num];
    for(size_t i=0; i< word_vec_size*word_num; i++){
        real rand_num = (rand() / (real) RAND_MAX - 0.5) / word_vec_size;
        x[i]=rand_num;
    }

    real *u = new real[sub_vec_size*sub_num];
    for(size_t i=0; i< sub_vec_size*sub_num; i++){
        real rand_num = (rand() / (real) RAND_MAX - 0.5) / sub_vec_size;
        u[i]=rand_num;
    }

    for(int i=0; i<word_num;i++){
        pack.x.push_back(x+word_vec_size*i);
    }

    for(int i=0; i<sub_num; i++){
        pack.u.push_back(u+sub_vec_size*i);
    }

    real h=1e-4;
    forward(pack);
    backward(pack);

    for(size_t i=0; i< word_vec_size*word_num; i++){
        x[i]=x[i]+h;
        real cost1=forward(pack);
        x[i]=x[i]-2*h;
        real cost2=forward(pack);
        real num_grad=(cost1-cost2)/(2*h);
        x[i]=x[i]+h;
        cout<< "num gradient x:" << num_grad << " gradient x:" << pack.grad_x[i] 
            << " cost1: " << cost1 <<" cost2: " << cost2 << endl;
    }

    for(size_t i=0; i< sub_vec_size*sub_num; i++){
        u[i]=u[i]+h;
        real cost1=forward(pack);
        u[i]=u[i]-2*h;
        real cost2=forward(pack);
        real num_grad=(cost1-cost2)/(2*h);
        u[i]=u[i]+h;
        cout<< "num gradient u:" << num_grad << " gradient u:" << pack.grad_u[i] 
            << " cost1: " << cost1 <<" cost2: " << cost2 << endl;
    }
}

void print_vec(vector<real* >& u, int size)
{
    for(size_t i=0; i< u.size() ; i++){
        for(size_t j=0; j< size; j++){
            cout << u[i][j] << " ";
        }
        cout << endl;
    }
}

void NNRewrite::print_vec(vector<real>& z)
{
    for(size_t i = 0; i < z.size(); i++) {
        cout <<  z[i] << " "; 
    }
    cout << endl;
}

int NNRewrite::parse_pack(vector<string>& vec,nn_pack& pack, int type)
{
    pair<int,int> range;
    int sub_id;
    if(type==TEXT){
         if( vec.size() < 5) return -1;
         split_str(pack.words, vec[3], " ");
         if(pack.words.size()>20) return -1;
         word2index(pack.words, pack.wids);
         vec[1] = vec[0] + "\t" + vec[1];
         sub_id = sub_index[vec[1]];
         range = sub_list[vec[0]];
         pack.fid = atoi(vec[2].c_str());
         pack.freq=atof(vec[4].c_str());
         /*if(pack.fid +2 >= pack.words.size()){
             cerr << "err input:" << vec[1] << "\t"  <<  vec[2] << "\t" << vec[3] << endl;
             return -1;
         }*/
         if(vec.size()>5){
             label=atoi(vec[5].c_str());
         }
     }else{
         split_str(pack.words, vec[0], " ");
         if(pack.words.size()>20) return -1;
         pack.wids.clear();
         for(size_t i=0; i<pack.words.size(); i++){
             pack.wids.push_back(atoi(pack.words[i].c_str()));
         }
         pack.fid=atoi(vec[1].c_str());
         sub_id=atoi(vec[2].c_str());

         range.first=atoi(vec[3].c_str());
         range.second=atoi(vec[4].c_str());
     }
     getWordVec(pack.wids, pack.x);
     getSubVec(range,pack.u);
     pack.label=sub_id - range.first;
     return 0;
}

void NNRewrite::Train(int id, string& train_file,long long file_size) {


    ifstream fin(train_file.c_str());
    long start_index = file_size / (long long) thread_num * (long long) id;
    fin.seekg(start_index);
    long long local_num = 0;
    long long read_size = file_size / thread_num;
    pthread_mutex_lock(&mutex);
    cerr <<"thread" << id << " start_index:"<< start_index
         << " read_size:"<< read_size << " updatex:" << update_word 
         << " input_type:" << input_type << endl;
    pthread_mutex_unlock(&mutex);

    vector<string> vec;
    string line;
    nn_pack pack;
    InitPack(pack);
    int print_step=1;

    //Text format:A B index query ...
    //id format: wids fid subid first second
    long long cur_size=0;
    pair<int, int> range;
    int smpNum=10000;


    getline(fin, line);
    while (getline(fin, line)) {

        if (cur_size > read_size) break;
        cur_size+=line.size()+1;
        local_num++;

        int sub_id=0;
        split_str(vec, line, "\t");

        if(parse_pack(vec,pack,input_type)<0) {
            //cerr << "err input:" << line << endl;
            continue;
        }
        if(local_num%smpNum==0){
            pthread_mutex_lock(&mutex);
            samples.push_back(line);
            pthread_mutex_unlock(&mutex);
        }
        sgdTrain(pack,line,local_num);

    }

    if(iter_num==0){
        pthread_mutex_lock(&mutex);
        cerr << "read_size:" << read_size << "cur_size:" << cur_size << endl;
        line_cnt+=local_num;
        pthread_mutex_unlock(&mutex);
    }

    fin.close();
}
void NNRewrite::SampleLoss(){

    cerr << "########## SampleLoss " << endl;
    nn_pack pack;
    InitPack(pack);
    vector<string> vec;
    float prob_sum=0.0;
    float weight_sum=0.0;
    float weight_cnt=0.0;
    string line;
    for(size_t i=0; i<samples.size(); i++){
        split_str(vec, samples[i], "\t");
        if(parse_pack(vec,pack,input_type)<0) continue;
        forward(pack);
        float t1=pack.a2[pack.label];
        float t2=sigmod(pack.z2[pack.label]);
        prob_sum+=t1;
        weight_sum+=t1*pack.freq;
        weight_cnt+=pack.freq;
        if(input_type==TEXT){
            cerr <<"sample:\t" << samples[i] << "\t" << t1 << "\t" << t2 << endl;
        }else{
            line=vocab[pack.wids[0]];
            for(size_t a=1; a< pack.wids.size(); a++){
                line+=" "+vocab[pack.wids[a]];
            }
            continue;
        }
    }
    if(samples.size()==0) return;
    cerr << "############ aver prob: " << prob_sum/samples.size() << endl;
    cerr << "############ aver prob: " << weight_sum/weight_cnt << endl;
    samples.clear();
}

void NNRewrite::sgdTrain(nn_pack& pack, const string& line, int local_num)
{
    //if(pack.freq>100) pack.freq=100;
    //pack.freq=sqrt(pack.freq)/4;
    pack.freq=1.0;

    int print_step=10000;

    //forward:
    forward(pack);
    float t1=pack.a2[pack.label];
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
        cerr << line << "\t" << t1 << "\t" << pack.a2[pack.label] << endl;
    }
}

void NNRewrite::SaveModel(const string& outfile) {

    ofstream out(outfile.c_str());
    sub_num = sub_index.size();

    cerr << "################### line_cnt:"<< line_cnt << endl;

    out <<"#wordvec\t" << vocab_size << "\t" << word_vec_size << endl;
    out <<"#subvec\t" << sub_num << "\t" << sub_vec_size << endl;

    for (int a = 0; a < vocab_size; a++) {
        out << vocab[a];
        for (int b = 0; b < word_vec_size; b++){
            out << "\t" << word_vec[a * word_vec_size + b];
        }
        out << endl;
    }

    for (map<string,int>::iterator it = sub_index.begin(); it != sub_index.end(); it++)
    {
        real * p= sub_vec+it->second*sub_vec_size;
        out << it->first ;
        for(int i=0;i<sub_vec_size;i++){
            out << "\t" << p[i];
        }
        out << endl;

    }


    out.flush();
    out.close();
}

void NNRewrite::PrintModel(){
    cerr << "################### line_cnt:"<< line_cnt << endl;

    cerr<<"max_iter_num:" << max_iter_num << endl;

    sub_num = sub_index.size();
    cout << "#subvec\t" << sub_num << "\t" << sub_vec_size << endl;
    cout << "#w_num\t1" << endl;
    cout << "#b_num\t1" << endl;
    for (map<string, int>::iterator it = sub_index.begin(); it != sub_index.end(); it++)
    {
        real * p = sub_vec + it->second*sub_vec_size;
        cout << label <<"\t1\t" << it->first;
        for (int i = 0; i < sub_vec_size; i++){
            cout << "\t" << p[i];
        }
        cout << endl;

    }
    cerr << "print subvec done" << endl;


    int wi = 0;
    cout << label <<"\t2\t";
//    for (int m = 0; m < sub_vec_size; m++)
//    {
//
//        for (int i = 0; i < window_size - 1; i++) {
//
//            for (int j = 0; j < word_vec_size; j++)
//            {
//                if(wi>0) cout << "\t";
//                cout << w[wi];
//                wi++;
//            }
//        }
//    }


    for (int wi = 0; wi < w_size; wi++)
    {
        if (wi>0) cout << "\t";
        cout << w[wi];
    }
    cout << endl;


    cerr << "print w done" << endl;

    cout << label <<"\t3\t";
    for (int m = 0; m < sub_vec_size; m++)
    {
        if(m>0) cout << "\t";
        cout  << b[m] ;
    }
    cout << endl;

    cerr << "print b done" << endl;

}

void NNRewrite::LoadModel(const string& infile) {

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
             size_t wlen=w_size;
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

    cerr << psk <<":" << sub_list.size() << endl;
    cerr << "load the model done" << endl;

    return ;

}

void NNRewrite::Predict(vector<string>& query, size_t index, map<string, real>& score, nn_pack& pack, int type){


    if(sub_label.find(query[index])==sub_label.end()){
        return;
    }
    int label=sub_label[query[index]];
    w=w_vector[label];
    b=b_vector[label];

    vector<int> wids;
    word2index(query, wids);

    //nn_pack pack;
    pack.fid = index;
    pack.freq=1;
    pack.label=0;
    //InitPack(pack);

    if(sub_list.find(query[index]) == sub_list.end()){
        return;
    }
    pair<int, int> range = sub_list[query[index]];
    getWordVec(wids, pack.x);
    getSubVec(range,pack.u);
    forward(pack);

    if(type==1){
        string str1=query[index]+"\t"+query[index];
        for(int i=range.first; i<= range.second; i++){
            if(str1 == sub_string[i]){
                int a=i - range.first;
                pack.z2[a]=-1e12;
                //cerr << "equal: " << str1 << ":" << a << endl;
                break;
            }
        }
        //print_vec(pack.z2);
        pack.a2.clear();
        softmax(pack.z2, pack.a2);
        //print_vec(pack.a2);
    }


    for(int i=range.first; i<= range.second; i++){
        pack.label=i - range.first;
        score[sub_string[i]]=pack.a2[pack.label];
        //if(type==1) score[sub_string[i]]= sigmod(pack.z2[pack.label]);
    }
}


void NNRewrite::Predict2(vector<string>& query, size_t begin, size_t end,
        map<string, pair<real,real> >& score, nn_pack& pack)
{
    string key=query[begin];
    for(int i=begin+1; i<= end; i++){
        key=key+"|"+query[i];
    }
    if(sub_label.find(key)==sub_label.end()){
        return;
    }
    //cerr << "key:" << key << " " << begin << " " << end << endl;
    int label=sub_label[key];
    w=w_vector[label];
    b=b_vector[label];

    vector<int> wids;
    word2index(query, wids);

    //nn_pack pack;
    pack.fid = begin;
    pack.freq=1;
    pack.label=0;
    //InitPack(pack);

    if(sub_list.find(key) == sub_list.end()){
        return;
    }
    pair<int, int> range = sub_list[key];
    getWordVec(wids, pack.x, begin, end);
    getSubVec(range,pack.u);
    forward(pack);

    for(int i=range.first; i<= range.second; i++){
        pack.label=i - range.first;
        score[sub_string[i]].first=pack.a2[pack.label];
    }

    string str1 = key + "\t" + key;
    for (int i = range.first; i <= range.second; i++) {
        if (str1 == sub_string[i]) {
            int a = i - range.first;
            pack.z2[a] = -1e12;
            break;
        }
    }
    //print_vec(pack.z2);
    pack.a2.clear();
    softmax(pack.z2, pack.a2);
    //print_vec(pack.a2);

    for(int i=range.first; i<= range.second; i++){
        pack.label=i - range.first;
        score[sub_string[i]].second=pack.a2[pack.label];
    }


}

void NNRewrite::Expand(vector<string>& query1, string& query2,
           map<string, real>& score, nn_pack& pack)
{
    if(sub_label.find(query2)==sub_label.end()){
        return;
    }
    int label=sub_label[query2];
    w=w_vector[label];
    b=b_vector[label];

    vector<int> wids;
    word2index(query1, wids);

    //nn_pack pack;
    pack.fid = 0;
    pack.freq=1;
    pack.label=0;
    //InitPack(pack);

    if(sub_list.find(query2) == sub_list.end()){
        return;
    }
    pair<int, int> range = sub_list[query2];
    getWordVec(wids, pack.x);
    getSubVec(range,pack.u);
    forward(pack);

    for(int i=range.first; i<= range.second; i++){
        pack.label=i - range.first;
        score[sub_string[i]]=pack.a2[pack.label];
    }
}
