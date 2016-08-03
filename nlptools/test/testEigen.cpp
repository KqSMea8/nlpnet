#include <iostream>
#include <vector>
#include <Eigen/Eigen>
#include "typedef.h"
#include "Layer.h"
#include "utils.h"
using namespace Eigen;
using namespace std;

pthread_mutex_t mutex2 = PTHREAD_MUTEX_INITIALIZER;

int thread_num=1;
string train_file;
string test_file;
long long file_size;
MLPNet mlpNet;
int gline_num=0;

int maxIndex(ColVectorXfr& val)
{
    int max_id=-1;
    float max_val=-1.0;
    for(size_t i=0; i< val.rows(); i++){
        if(val[i]>max_val){
            max_id=i;
            max_val=val[i];
        }
    }
    return max_id;
}

void mutex_print(ostringstream& oss)
{
    pthread_mutex_lock(&mutex2);
    cerr << oss.str();
    oss.str("");
    pthread_mutex_unlock(&mutex2);
}
void eval(string& file){

    ifstream fin(file.c_str());
    int index=0;
    map<string,Blob*> &blob_dict =mlpNet._blob_dicts[index];
    IntBlob* input =(IntBlob*) blob_dict["input"];
    IntBlob* label =(IntBlob*) blob_dict["label"];
    DecimBlob* logit =(DecimBlob*) blob_dict["logit"];
    float right_num=0.0;
    vector<string> vec;
    string line;
    int cnt=0;
    while (getline(fin, line)) {
        split_str(vec, line, " ");
        if(vec.size() <4 || vec.size()>20 ) continue;
        for(size_t i=0; i<vec.size()-1; i++){
            int id=atoi(vec[i].c_str());
            if(id<0) id=0;
            input->val[i]=id;
        }
        cnt+=1;
        size_t last=vec.size()-1;
        label->val[0]=atoi(vec[last].c_str());
        forward(mlpNet._networks[index]);
        if (maxIndex(logit->val)==label->val[0]) right_num+=1.0;

    }
    fin.close();
    ostringstream oss;
    oss << "###################### eval acc:" << right_num/cnt << endl;
    mutex_print(oss);

}
void* threadRun(void *id) {

    ifstream fin(train_file.c_str());
    long start_index = file_size / (long long) thread_num * (long long) id;
    fin.seekg(start_index);
    long long local_num = 0;
    long long read_size = file_size / thread_num;
    ostringstream oss;
    oss << "thread" << (long) id << " start_index:" << start_index
        << " read_size:" << read_size << endl;

    mutex_print(oss) ;


    long long cur_size=0;
    string line;
    vector<string> vec;
    int index=(long) id;
    map<string,Blob*> &blob_dict =mlpNet._blob_dicts[index];
    IntBlob* input =(IntBlob*) blob_dict["input"];
    IntBlob* label =(IntBlob*) blob_dict["label"];
    DecimBlob* logit =(DecimBlob*) blob_dict["logit"];
    getline(fin, line);
    float right_num=0.0;
    int line_num =0;
    while (getline(fin, line)) {
        if (cur_size > read_size) break;
        cur_size+=line.size()+1;
        line_num+=1;
        gline_num+=1;
        split_str(vec, line, " ");
        if(vec.size() <4) continue;
        for(size_t i=0; i<vec.size()-1; i++){
            int id=atoi(vec[i].c_str());
            if(id<0){
                //cerr<< "error id:" << id << endl;
                id=0;
            }
            if(i>=20) continue;
            input->val[i]=id;
        }
        

        input->real_size = vec.size();
        if(input->real_size>20) input->real_size=20;
        size_t last=vec.size()-1;
        label->val[0]=atoi(vec[last].c_str());
        forward(mlpNet._networks[index]);
        backward(mlpNet._networks[index]);
        update(mlpNet._networks[index]);

        if (maxIndex(logit->val)==label->val[0]) right_num+=1.0;

        if(gline_num%1000000==0){
            oss << "acc:" << right_num/line_num << " line_num:" << line_num << endl;
            oss << "train_num:" << gline_num << endl;
            right_num=0.0;
            line_num=0;
            mutex_print(oss);
        }

    }
    fin.close();
    oss << "acc:" << right_num/line_num << " line_num:" << line_num << endl;
    mutex_print(oss);
    return NULL;
}

void TrainModel() {

    //mlpNet.InitWeight();
    /*LayerInfo layer0={"input","input", 0,3,0,0,INT};
    LayerInfo layer1={"emb","emb",3,150,50,1203000,DECIM};
    LayerInfo layer2={"fc","fc",150,50,50,150,DECIM};
    LayerInfo layer21={"relu","relu",50,50,50,50,DECIM};
    LayerInfo layer3={"logit","logit",50,3,3,50,DECIM};
    LayerInfo loss={"loss","loss",3,1,0,0,DECIM};

    vector<LayerInfo> layer_conf;
    layer_conf.push_back(layer0);
    layer_conf.push_back(layer1);
    layer_conf.push_back(layer2);
    layer_conf.push_back(layer21);
    layer_conf.push_back(layer3);*/
    
    LayerInfo layer0={"input","input", 0,20,0,0,INT};
    LayerInfo layer1={"emb","emb", 20,1000,50,1203000,DECIM};
    LayerInfo layer2={"conv","conv", 1000,50,50,150,DECIM};
    LayerInfo layer3={"logit","logit",50,3,3,50,DECIM};
    LayerInfo loss={"loss","loss",3,1,0,0,DECIM};

    vector<LayerInfo> layer_conf;
    layer_conf.push_back(layer0);
    layer_conf.push_back(layer1);
    layer_conf.push_back(layer2);
    layer_conf.push_back(layer3);


    mlpNet._layer_conf=layer_conf;
    mlpNet.BuildNets(thread_num);

    ifstream fin1(train_file.c_str());
    fin1.seekg(0, ios::end);
    file_size = fin1.tellg();
    
    for(size_t i=0; i<20; i++){
        cerr << "#### iteration:" << i << endl;
        pthread_t *pt = (pthread_t *) malloc(thread_num * sizeof(pthread_t));
        for (int a = 0; a < thread_num; a++) {
            pthread_create(&pt[a], NULL, threadRun, (void *) a);
        }

        for (int a = 0; a < thread_num; a++) {
            pthread_join(pt[a], NULL);
        }
        eval(test_file);
    }
    mlpNet.Release();
}

void grad_check(){

     MLPNet mlp;

     LayerInfo layer0={"input","input", 0,20,0,0,INT};
     LayerInfo layer1={"emb","emb", 20,1000,50,100,DECIM};
     LayerInfo layer2={"conv","conv", 1000,50,50,150,DECIM};
     LayerInfo layer3={"logit","logit",50,3,3,50,DECIM};
     LayerInfo loss={"loss","loss",3,1,0,0,DECIM};

     vector<LayerInfo> layer_conf;
     layer_conf.push_back(layer0);
     layer_conf.push_back(layer1);
     layer_conf.push_back(layer2);
     layer_conf.push_back(layer3);
     mlp._layer_conf=layer_conf;

     //mlp.InitWeight();
     mlp.BuildNets(1);

     vector<Layer* >&  net = mlp._networks[0];
     map<string, Blob* > _blob_dict = mlp._blob_dicts[0];

     IntBlob* input =(IntBlob*) _blob_dict["input"];
     DecimBlob* ouput =(DecimBlob*) _blob_dict["loss"];
     input->val[0] = 0;
     input->val[1] = 1;
     input->val[2] = 2;
     input->val[3] = 3;
     input->val[4] = 3;
     input->real_size=5;

     forward(net);
     backward(net);

     //number gradient
     decim h=h=1e-4;
     //cout << input->grad << endl;
     /*for(size_t i=0; i<input->val.rows(); i++){
         input->val[i]=input->val[i]+h;
         forward(net);
         decim cost1=ouput->val[0];
         input->val[i]=input->val[i]-2*h;
         forward(net);
         decim cost2=ouput->val[0];
         decim num_grad=(cost1-cost2)/(2*h);
         input->val[i]=input->val[i]+h;
         cout<< "num gradient x:" << num_grad << " gradient x:" << input->grad[i]
                  << " cost1: " << cost1 <<" cost2: " << cost2 << endl;
     }*/
     for (map<string, Weight* >::iterator it = mlp._weight_dict.begin();
                 it != mlp._weight_dict.end(); ++it) {
         Weight* w=it->second;
         for(size_t j=0; j< w->val.cols(); j++){
             for(size_t i=0; i< w->val.rows(); i++){
                 w->val(i,j)=w->val(i,j)+h;
                 forward(net);
                 decim cost1=ouput->val[0];
                 w->val(i,j)=w->val(i,j)-2*h;
                 forward(net);
                 decim cost2=ouput->val[0];
                 decim num_grad=(cost1-cost2)/(2*h);
                 w->val(i,j)=w->val(i,j)+h;
                 bool error = abs(num_grad - w->grad(i,j))>0.0001;
                 if(error) {
                    cout << "gradient error:" << it->first <<  endl; 
                 }
                 if( i==0 && j==0 ){
                     cout<< it->first << " num gradient x:" << num_grad << " gradient x:" << w->grad(i,j)
                         << " cost1: " << cost1 <<" cost2: " << cost2 << endl;
                 }

             }
         }
     }

    mlp.Release();
 }




int main(int argc, char **argv)
{


    /*Vector3f x, y, z;
    Matrix<float,4,1> a;
    Matrix<float,4,2> b;
    cout << "a=" << endl << a << endl;
    cout << "b=" << endl << b << endl;
    cout <<" a.segment(1,2): " << endl << a.segment(1, 2) << endl;
    cout << "a=" << endl << a << endl;
    cout <<" a.segment(1,3): " << endl << a.segment(1, 3) << endl;
    cout << "a=" << endl << a << endl;
    cout <<" a.segment(0,2): " << endl << a.segment(0, 2) << endl;
    cout << "a=" << endl << a << endl;
    //a.segment(1,2)=b;
    cout <<" a.segment(1,2)=b " << endl << a  << endl;*/



    /*MatrixXr R=MatrixXr::Random(3,1);
    MatrixXr R2=MatrixXr::Random(3,1);
    cout << "R=" << endl << R << endl;
    cout << "R2=" << endl << R << endl;
    cout << "R.dot(R2)" << endl;
    cout << R.cwiseProduct(R2) << endl;
    MatrixXr Q=R.transpose();
    cout << "Q=" << endl << Q << endl;
    Q(0,0)=19.7;
    cout << "Q=" << endl << Q << endl;
    cout << "R=" << endl << R << endl;

    DecimBlob  *blob= new DecimBlob(5);
    cout << "blob:" << blob->val << endl;
    cout << "blob[0]:" << blob->val[0] << endl;
    decim a=1.0;
    blob->val[0]=exp(blob->val[0]-a);
    cout << "blob[0]:" << blob->val[0] << endl;*/

    //testNN();
    Test t;
    t.name="hello";
    Test t2=t;
    t2.name="ok";
    cout << t.name << " " << t2.name << endl;
    Test b;
    Test2 c;
    b=c;
    c.test();
    b.test();

    Test2* d=(Test2*)&b;
    d->test();
    if(argc<2){
        MatrixXr a=MatrixXr::Random(3,3);
        cout << "random:" << endl << a << endl;
        MatrixXr b=a.array().square();
        a.array()=2*a.array().square();
        cout << "square:" << endl << b << endl;
        cout << "square*2:" << endl << a << endl;
        cout << "a/b" << endl << a.array()/b.array() << endl;
        grad_check();
        return 1;
    }
    train_file=argv[1];
    test_file=argv[2];
    if(argc>3) thread_num=atoi(argv[3]); 
    TrainModel();
    return 0;

}
