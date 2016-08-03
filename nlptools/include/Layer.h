/*
 * Layer.h
 *
 *  Created on: 2015-12-30
 *      Author: zengzengfeng
 */

#ifndef Layer_H_
#define Layer_H_

#include <iostream>
#include <vector>
#include <map>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <Eigen/Eigen>
#include "utils.h"
#include "typedef.h"
#include <pthread.h>
using namespace std;
using namespace Eigen;

typedef double decim;
typedef Matrix<double, Dynamic, 1> ColVectorXfr;
typedef Matrix<int, Dynamic, 1> ColVectorInt;
//typedef Matrix<double, Dynamic, Dynamic,RowMajor> MatrixXr;
typedef Matrix<double, Dynamic, Dynamic> MatrixXr;
typedef Matrix<int, Dynamic, Dynamic> MatrixInt;

enum {INT, DECIM};
class Blob {

public:
    Blob() {};
    Blob(size_t size) {};
    virtual ~Blob(){};
    virtual int forward(){return 0;};
    virtual int backward(){return 0;};
public:
    int real_size;
};

class DecimBlob: public Blob {

public:
    DecimBlob() {};
    DecimBlob(size_t size) {
        val = MatrixXr::Random(size,1);
        grad = MatrixXr::Random(size,1);
        real_size=size;
    };
    virtual ~DecimBlob(){};
    virtual int forward(){return 0;};
    virtual int backward(){return 0;};
public:
    ColVectorXfr val;
    ColVectorXfr grad;
};

class IntBlob: public Blob {

public:
    IntBlob() {};
    IntBlob(size_t size) {
        val = MatrixInt::Zero(size,1);
        grad = MatrixInt::Zero(size,1);
        real_size=size;
    };
    virtual ~IntBlob(){};
    virtual int forward(){return 0;};
    virtual int backward(){return 0;};
public:
    ColVectorInt val;
    ColVectorInt grad;
};

class Weight {
public:
    Weight(){};
    Weight(size_t x, size_t y){
        val = MatrixXr::Random(x,y)*0.02;
        grad = MatrixXr::Zero(x,y);
        ada = MatrixXr::Ones(x,y);
    };
    void add_grad(size_t idx, ColVectorXfr& grad){}; ///< add sparse grad
    void update(decim eta){
        decim square=eta*eta;
        ada.array()=ada.array()+square*grad.array().square();
        val.array()=val.array()-eta*grad.array()/ada.array().sqrt();
    };
    void update(size_t idx, decim eta){
        decim square=eta*eta;
        ada.array().col(idx)=ada.array().col(idx)
                                +square*grad.col(idx).array().square();
        val.array().col(idx)=val.array().col(idx)
                                -eta*grad.col(idx).array()/ada.col(idx).array().sqrt();
    };

public:
    MatrixXr val;
    MatrixXr grad;
    MatrixXr ada; //for  adaptive updates
};


class Layer {
public:
    Layer(){}
    virtual ~Layer(){}
    virtual void Init(const string& name, Blob* input,
                    Blob* output, map<string,Weight*>& dict) {
    }

    virtual void Forward(){};
    virtual void Backward(){};
    virtual void Update(decim alpha){};

public:
    string _name;

};

class FcLayer : public Layer{
public:
    FcLayer(){}
    void Init(const string& name, Blob* input, Blob* output, map<string,Weight*>& dict) {
         _input = (DecimBlob*)input;
         _output = (DecimBlob*)output;
         _weight = dict[name+"_weight"];
         _bias = dict[name+"_bias"];
         _name = name;
     }

    virtual ~FcLayer(){

    }

    void Forward(){
#ifdef _DEBUG 
        cerr << "forward:" << _name << endl;
#endif
        _output->val = _weight->val * _input->val+_bias->val;
        for(size_t i=0; i<_output->val.rows(); i++){
            decim a=_output->val[i];
            _output->val[i]=active(a);
        }
    };

    void Backward() {
        //col-major
        for(size_t i=0; i<_output->grad.rows(); i++){
            decim a=_output->val[i];
            decim g= acti_der(a);
            _output->grad[i]=g*_output->grad[i];
        }

        for(size_t i=0; i< _weight->grad.cols(); i++){
            decim a=_input->val[i];
            _weight->grad.col(i)=_output->grad*a;
        }
        _bias->grad = _output->grad;

        for(size_t i=0; i<_input->grad.rows(); i++){
            _input->grad[i]=_output->grad.dot(_weight->val.col(i));
        }
    }

    void Update(decim alpha){
        //_weight->val = _weight->val - alpha*_weight->grad;
        //_bias->val = _bias->val - alpha*_bias->grad;
        _weight->update(alpha);
        _bias->update(alpha);
    }
    
    //row-major
    void Backward2() {
        //_weight->grad = _output->grad * _input->val.transpose();
        //cout << "fc backwoard: " << endl;
        MatrixXr trans_val =_input->val.transpose();
        for(size_t i=0; i< _weight->grad.rows(); i++){
            decim a=_output->val[i];
            decim g= acti_der(a);
            _weight->grad.row(i) = _output->grad.row(i) * trans_val*g ;
        }


        //_input-> grad =  _weight->grad.transpose() * _output->grad;

        decim a=_output->val[0];
        decim g= acti_der(a);
        _input->grad =_weight->val.row(0)*_output->grad[0]*g;
        for (size_t i = 1; i < _weight->val.rows(); i++) {
            decim a=_output->val[i];
            decim g= acti_der(a);
            _input->grad +=_weight->val.row(i)*_output->grad[i]*g;
        }

    };

private:
    inline decim active(decim arg){
        if (arg < -6) return -1.0;
        else if (arg > 6) return 1.0;
        return tanh(arg);
    }
    inline decim acti_der(decim sigma) {
        return 1 - (sigma*sigma);
    }

public:
    DecimBlob* _input;
    DecimBlob* _output;
    Weight* _weight;
    Weight* _bias;
};


class ConvLayer : public Layer{
public:
    ConvLayer(){}
    void Init(const string& name, Blob* input, Blob* output, map<string,Weight*>& dict) {
         _input = (DecimBlob*)input;
         _output = (DecimBlob*)output;
         _weight = dict[name+"_weight"];
         _bias = dict[name+"_bias"];
         _name = name;
         _map=MatrixXr::Zero(_output->val.rows(),20);
         _max_id=MatrixXr::Zero(_output->val.rows(),1);
     }

    virtual ~ConvLayer(){

    }

    void Forward(){
        int window=150;
        int stride =50;
        //cout << "_input->real_size=" << _input->real_size << endl;
        int n=(_input->real_size-window)/stride +1;
        for(int i=0;i<n;i++){
            int begin=i*stride;
            _map.col(i) = _weight->val*_input->val.segment(begin,window)+_bias->val;
        }
        //cout << "_map=" << endl << _map << endl;
        //max-pooling
        for(size_t j=0; j<_map.rows();j++){
            decim max=-10000.0;
            size_t max_id=0;
            for(int i=0;i<n;i++){
                if (_map(j,i)>max){
                    max=_map(j,i);
                    max_id=i;
                }
            }
            _output->val[j]=active(max);
            _max_id[j]=max_id;
        }


    };

    void Backward() {
        //col-major
        /*cout << " _weight:" <<  _weight->grad.rows() << ","
              << _weight->grad.cols() << endl;
        cout << " _input:" <<  _input->grad.rows() << ","
              << _input->grad.cols() << endl;*/

        for(size_t i=0; i<_output->grad.rows(); i++){
            decim a=_output->val[i];
            decim g= acti_der(a);
            _output->grad[i]=g*_output->grad[i];
        }

        int stride =50;
        //cout<< "_max_id=" << endl << _max_id.transpose() << endl;
        for (size_t i = 0; i < _weight->grad.cols(); i++) {
            //decim a=_input->val[i];
            for (size_t j = 0; j < _weight->grad.rows(); j++) {
                int begin = _max_id[j] * stride;
                //cout << "begin:" << begin << endl;
                decim a = _input->val[begin + i];
                _weight->grad(j,i) = _output->grad[j] * a;
            }

        }
        _bias->grad = _output->grad;


        /*for(size_t i=0;i<_input->real_size; i++){
            _input->grad[i]=0;
        }*/
        _input->grad.setZero();
        int window=150;
        for(size_t i=0; i<window; i++){
            for (size_t j = 0; j < _weight->grad.rows(); j++) {
                int begin = _max_id[j] * stride;
                _input->grad[begin+i]+=_output->grad[j]*_weight->val(j,i);
            }
        }

        //for(size_t i=0; i<_input->grad.rows(); i++){
             //_input->grad[i]=_output->grad.dot(_weight->val.col(i));
        //}
    }

    void Update(decim alpha){
        //_weight->val = _weight->val - alpha*_weight->grad;
        //_bias->val = _bias->val - alpha*_bias->grad;
        _weight->update(alpha);
        _bias->update(alpha);
    }



private:
    inline decim active(decim arg){
        if (arg < -6) return -1.0;
        else if (arg > 6) return 1.0;
        return tanh(arg);
    }
    inline decim acti_der(decim sigma) {
        return 1 - (sigma*sigma);
    }

public:
    DecimBlob* _input;
    DecimBlob* _output;
    Weight* _weight;
    Weight* _bias;
    MatrixXr _map;
    ColVectorInt _max_id;
};



class ReluLayer : public Layer{
public:
    ReluLayer(){}
    void Init(const string& name, Blob* input, Blob* output, map<string,Weight*>& dict) {
         _input = (DecimBlob*) input;
         _output = (DecimBlob*) output;
         _weight = dict[name+"_weight"];
         _bias = dict[name+"_bias"];
        for(size_t i=0; i< _bias->val.rows(); i++){
            //_bias->val(i,0)=1;
        }
         _name=name;
     }

    virtual ~ReluLayer(){

    }

    void Forward(){
#ifdef _DEBUG 
        cerr << "forward:" << _name << endl;
#endif
        _output->val = _weight->val * _input->val+_bias->val;
        for(size_t i=0; i<_output->val.rows(); i++){
            decim a=_output->val[i];
            _output->val[i]=active(a);
        }
    };

    void Backward() {
        //col-major
        for(size_t i=0; i<_output->grad.rows(); i++){
            decim a=_output->val[i];
            decim g= acti_der(a);
            _output->grad[i]=g*_output->grad[i];
        }

        for(size_t i=0; i< _weight->grad.cols(); i++){
            decim a=_input->val[i];
            _weight->grad.col(i)=_output->grad*a;
        }
        _bias->grad = _output->grad;

        for(size_t i=0; i<_input->grad.rows(); i++){
            _input->grad[i]=_output->grad.dot(_weight->val.col(i));
        }
    }

    void Update(decim alpha){
        //_weight->val = _weight->val - alpha*_weight->grad;
        //_bias->val = _bias->val - alpha*_bias->grad;
        _weight->update(alpha);
        _bias->update(alpha);

    }

private:

    inline decim active(decim x){
        if (x < -0.5 ){
            return 0.0;
        }else{
            return x;
        }
        //return log(1+exp(x));
    }

    inline decim acti_der(decim x) {
        if(x> -0.5){
            return 1.0;
        }else{
            return 0.0;
        }
        
        //return 1.0/(1+exp(-x));
    }

public:
    DecimBlob* _input;
    DecimBlob* _output;
    Weight* _weight;
    Weight* _bias;

};


class LogitLayer : public Layer{
public:
    LogitLayer(){}
    void Init(const string& name, Blob* input, Blob* output, map<string,Weight*>& dict) {
         _input = (DecimBlob*) input;
         _output = (DecimBlob*) output;
         _weight = dict[name+"_weight"];
         _bias = dict[name+"_bias"];
         _name=name;
     }

    virtual ~LogitLayer(){

    }

    void Forward(){

#ifdef _DEBUG 
        cerr << "forward:" << _name << endl;
        cerr << "wsize:" << _weight->val.rows() << " " << _weight->val.cols() << endl;
#endif
        _output->val = _weight->val * _input->val + _bias->val;
        decim max_val=_output->val.maxCoeff(); 
        decim sum=0.0;
        for(size_t i=0; i<_output->val.rows(); i++){
            _output->val[i]=exp(_output->val[i]-max_val);
            sum+=_output->val[i];
        }
        for(size_t i=0; i<_output->val.rows(); i++){
            _output->val[i]=_output->val[i]/sum;
        }
    };
    //col-major
    void Backward(){
#ifdef _DEBUG 
        cerr << "backward:" << _name << endl;
#endif

        for(size_t i=0; i< _weight->grad.cols(); i++){
            _weight->grad.col(i)=_output->grad*_input->val[i];
        }
        _bias->grad = _output->grad;

        for(size_t i=0; i<_input->grad.rows(); i++){
            _input->grad[i]=_output->grad.dot(_weight->val.col(i));
        }
    }

    void Update(decim alpha){
        //_weight->val = _weight->val - alpha*_weight->grad;
        //_bias->val = _bias->val - alpha*_bias->grad;
        _weight->update(alpha);
        _bias->update(alpha);
    }
    //row-major
    void Backward2(){
        //_weight->grad =  _output->grad * _input->val.transpose() ;
        //cout << "LogitLayer backward:" << endl;

        MatrixXr trans_val =_input->val.transpose();
        for(size_t i=0; i< _weight->grad.rows(); i++){
            _weight->grad.row(i) = _output->grad.row(i) * trans_val ;
        }

        //_input-> grad =  _weight->grad.transpose() * _output->grad;
        _input-> grad = _weight->val.row(0)*_output->grad[0];
        for(size_t i=1; i< _weight->val.rows(); i++){
            _input-> grad += _weight->val.row(i)*_output->grad[i];
        }

    };
private:
    DecimBlob* _input;
    DecimBlob* _output;
    Weight* _weight;
    Weight* _bias;
};

class LossLayer : public Layer{
public:
    LossLayer(){}
    void Init(Blob* input, Blob* output, Blob* label) {
         _input = (DecimBlob*) input;
         _output = (DecimBlob*) output;
         _label = (IntBlob*) label;
         _name="Loss";
     }

    virtual ~LossLayer(){

    }

    void Forward(){
        int label=_label->val[0];
        decim a=_input->val[label];
        _output->val[0]=-log(a);
        return;
    };
    void Backward(){
        _input->grad = _input->val;
        int label=_label->val[0];
        _input->grad[label]=_input->grad[label]-1;
    };
    void Update(decim alpha){};

private:
    DecimBlob* _input;
    DecimBlob* _output;
    IntBlob* _label;

};


class EmbLayer : public Layer{
public:
    EmbLayer(){}
    void Init(const string& name, Blob* input, Blob* output, map<string,Weight*>& dict) {
         _input = (IntBlob*) input;
         _output = (DecimBlob*) output;
         _weight = dict[name+"_weight"];
         _name = name;
     }

    virtual ~EmbLayer(){
    }

    void Forward(){
#ifdef _DEBUG 
        cerr << "forward:" << _name << endl;
        cerr << "wsize:" << _weight->val.rows() << " " << _weight->val.cols() << endl;
        cerr << "input:" << _input->val << endl;
#endif
        int len=_weight->val.rows();
        for(size_t i=0; i<_input->real_size; i++){
            int id=_input->val[i];
            int a=i*len;
            //cout << " _output->val.segment(a,len)" << endl <<  _output->val.segment(a,len) << endl;
            //cout << "_weight->val.col(id)" << endl << _weight->val.col(id) << endl;
            _output->val.segment(a,len)=_weight->val.col(id);
        }
        _output->real_size=len*_input->real_size;
    };

    void Backward(){
        int len=_weight->val.rows();
        for(size_t i=0; i<_input->real_size; i++){
            int id=_input->val[i];
            _weight->grad.col(id).setZero();
        }
        for(size_t i=0; i<_input->real_size; i++){
            int id=_input->val[i];
            int a=i*len;
            _weight->grad.col(id)+=_output->grad.segment(a,len);
        }
    };

    void Update(decim alpha){
        for(size_t i=0; i<_input->val.rows(); i++){
            int id=_input->val[i];
            //_weight->val.col(id) =  _weight->val.col(id) - alpha*_weight->grad.col(id);
            _weight->update(id,alpha);
        }
    }
private:
    IntBlob* _input;
    DecimBlob* _output;
    Weight* _weight;
};

void forward(vector<Layer* > _network);
void backward(vector<Layer* > _network);
void update(vector<Layer* > _network);

void forward(vector<Layer* > _network){
    for (size_t i = 0; i < _network.size(); i++) {
        _network[i]->Forward();
        //cout <<"###########forward:" << _network[i]->_name << endl;
    }
}

void backward(vector<Layer* > _network){
    for(size_t i=_network.size()-1; i>= 0; i--){
        _network[i]->Backward();
        //cout <<"###########backward:" << _network[i]->_name << endl;
        if(i==0) break;
     }
}

void update(vector<Layer* > _network){
    for (size_t i = 0; i < _network.size(); i++) {
        _network[i]->Update(0.1);
    }
}

struct LayerInfo
{
    string name;
    string layer;
    int in_size;
    int out_size;
    int weight_rows;
    int weight_cols;
    int out_type;
};

class Test{
public:
    Test() {};
    virtual void test(){ cout<< "test1" << endl; };
public:
    string name;

};

class Test2: public Test{
public:
    Test2() {};
    void test(){ cout << "test2" << endl;}
public:
    string name;
    string type;
};

class MLPNet {
public:
    MLPNet() {
        LayerInfo layer0={"input","input", 0,3,0,0,INT};
        //LayerInfo layer1={"emb",3,150,50,1203000,DECIM};
        LayerInfo layer1={"emb","emb",3,15,5,10,DECIM};
        LayerInfo layer2={"relu","relu",15,5,5,15,DECIM};
        LayerInfo layer3={"fc","fc",5,5,5,5,DECIM};
        LayerInfo layer4={"logit","logit",5,3,3,5,DECIM};

        _layer_conf.push_back(layer0);
        _layer_conf.push_back(layer1);
        _layer_conf.push_back(layer2);
        _layer_conf.push_back(layer3);
        _layer_conf.push_back(layer4);
    }

    ~MLPNet(){

    }

    Layer* newLayer(const string& key){
        Layer* p=NULL;
        if(key=="fc") p= new FcLayer();
        if(key=="logit") p= new LogitLayer();
        if(key=="loss") p= new LossLayer();
        if(key=="emb") p= new EmbLayer();
        if(key=="relu") p= new ReluLayer();
        if(key=="conv") p= new ConvLayer();
        return p;
    }

    void BuildNets(int net_num){


        LayerInfo loss={"loss","loss",3,1,0,0,DECIM};

        for(size_t i=0; i<_layer_conf.size(); i++){
            LayerInfo &info=_layer_conf[i];
            if(info.weight_rows==0) continue;
            Weight *weight =new Weight(info.weight_rows,info.weight_cols);
            Weight *bias =new Weight(info.out_size,1);
            _weight_dict[info.name+"_weight"]=weight;
            _weight_dict[info.name+"_bias"]=bias;
        }

        for (int a = 0; a < net_num; a++) {
            map<string, Blob*> dict;
            vector<Layer*> net;
            for(size_t i=0; i<_layer_conf.size(); i++){
                LayerInfo &info=_layer_conf[i];
                Blob* out=NULL;
                if(info.out_type==INT){
                    out=new IntBlob(info.out_size);
                }else{
                    out=new DecimBlob(info.out_size);
                }
                dict[info.name] = out;
                if(i==0) continue;
                Layer* layer=newLayer(info.layer);
                Blob* in=dict[_layer_conf[i-1].name];
                layer->Init(info.name,in,out,_weight_dict);
                net.push_back(layer);
            }

            size_t last=_layer_conf.size()-1;
            Blob* loss_in=dict[_layer_conf[last].name];
            DecimBlob *loss_out = new DecimBlob(1);
            IntBlob *label = new IntBlob(1);


            LossLayer* loss_layer = new LossLayer();
            loss_layer->Init(loss_in,loss_out,label);

            net.push_back(loss_layer);
            dict[loss.name] = loss_out;
            label->val[0] = 1;
            dict["label"] = label;

            _blob_dicts.push_back(dict);
            _networks.push_back(net);
        }


    }

    void InitWeight(){
        //Weight *emb_weight =new Weight(50,1203000);
        Weight *emb_weight =new Weight(5,10);
        Weight *fc_weight =new Weight(5,15);
        Weight *fc_bias =new Weight(5,1);
        Weight *fc2_weight =new Weight(5,10);
        Weight *fc2_bias =new Weight(5,1);
        Weight *logit_weight =new Weight(3,5);
        Weight *logit_bias =new Weight(3,1);
        _weight_dict["emb_weight"]=emb_weight;
        _weight_dict["fc_weight"]=fc_weight;
        _weight_dict["fc_bias"]=fc_bias;
        _weight_dict["fc2_weight"]=fc2_weight;
        _weight_dict["fc2_bias"]=fc2_bias;
        _weight_dict["logit_weight"]=logit_weight;
        _weight_dict["logit_bias"]=logit_bias;

    }

    void BuildNets2(int net_num) {
        InitWeight();
        for (int i = 0; i < net_num; i++) {
            map<string, Blob*> dict;
            vector<Layer*> net;

            IntBlob *input = new IntBlob(3);
            DecimBlob *emb = new DecimBlob(15);
            DecimBlob *fc = new DecimBlob(5);
            DecimBlob *fc2 = new DecimBlob(5);
            DecimBlob *logit = new DecimBlob(3);
            DecimBlob *output = new DecimBlob(1);
            IntBlob *label = new IntBlob(1);
            label->val[0] = 1;
            input->val[0] = 0;
            input->val[1] = 1;
            input->val[2] = 2;

            dict["input"] = input;
            dict["fc"] = fc;
            dict["fc2"] = fc2;
            dict["logit"] = logit;
            dict["loss"] = output;
            dict["label"] = label;

            EmbLayer* emb_layer = new EmbLayer();
            emb_layer->Init(string("emb"),input, emb, _weight_dict);
            FcLayer* fc_layer = new FcLayer();
            fc_layer->Init(string("fc"),emb, fc, _weight_dict);
            LogitLayer* logit_layer = new LogitLayer();
            logit_layer->Init(string("logit"),fc, logit, _weight_dict);
            LossLayer* loss_layer = new LossLayer();
            loss_layer->Init(logit, output, label);
            net.push_back(emb_layer);
            net.push_back(fc_layer);
            //net.push_back(fc2_layer);
            net.push_back(logit_layer);
            net.push_back(loss_layer);

            _blob_dicts.push_back(dict);
            _networks.push_back(net);
        }

    }

    void Release() {
        for (size_t i = 0; i < _blob_dicts.size(); i++) {
            for (map<string, Blob*>::iterator it = _blob_dicts[i].begin();
                it != _blob_dicts[i].end(); ++it) {
                if (it->second != NULL)
                    delete it->second;
                it->second = NULL;
            }
        }

        for (map<string, Weight*>::iterator it = _weight_dict.begin();
            it != _weight_dict.end(); ++it) {
            if (it->second != NULL)
                delete it->second;
            it->second = NULL;
        }

        for (size_t i = 0; i < _networks.size(); i++) {
            for (size_t j = 0; j < _networks[i].size(); j++) {
                if (_networks[i][j] != NULL)
                    delete _networks[i][j];
                _networks[i][j] = NULL;
            }
        }

    }

public:

    map<string,Weight*> _weight_dict;
    vector<vector<Layer* > > _networks;
    vector<map<string, Blob* > > _blob_dicts;
    vector<LayerInfo> _layer_conf;
    map<string,Layer> _layer_dict;



};


#endif /* Layer_H_ */
