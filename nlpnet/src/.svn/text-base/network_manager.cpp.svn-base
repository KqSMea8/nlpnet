/*
 * network_manager.cpp
 *
 *  Created on: 2016-3-29
 *      Author: zengzengfeng
 */

#include "network_manager.h"

namespace nlpnet {

NetworkManager::NetworkManager() {
    // TODO Auto-generated constructor stub
    _layer_dict["fc"]=new FcLayer();
    _layer_dict["conc_emb"]=new ConcEmbLayer();
    _layer_dict["softmax_logloss"]=new SoftmaxLogLoss();
    _train_num = 0;
    _right_num = 0;

}

NetworkManager::~NetworkManager() {
    // TODO Auto-generated destructor stub
}
int NetworkManager::init_network(const Json::Value& config){
    Json::Value layers = config["layers"];
    std::string ss;
    for(unsigned int i = 0; i < layers.size(); i++){
        Json::Value type = layers[i]["layer_type"];
        ss += "add the layer:"+type.asString()+"\n";
        _layer_list.push_back(_layer_dict[type.asString()]);
        _layer_list[i]->Init(layers[i]);
    }
    fprintf(stdout,"add the layer:%s\n",ss.c_str());

    _output_key = layers[layers.size()-1]["connector"][0]["out"].asString();

    return SUCCESS;
}

int NetworkManager::forward_propagation(NeuronManager* neuron){
    for(size_t i = 0; i < _layer_list.size(); i++){
        //cout << "forward:" << i << endl;
        _layer_list[i]->Forward(neuron);
    }
    vector<int>& label = neuron->label();
    return SUCCESS;
}
int NetworkManager::backward_propagation(NeuronManager* neuron){
    for(int i = _layer_list.size()-1; i >= 0 ; i--){
        _layer_list[i]->Backward(neuron);
    }
    return SUCCESS;
}
int NetworkManager::update(){

    for(size_t i = 0; i < _layer_list.size(); i++){
         _layer_list[i]->Update();
    }
    return SUCCESS;
}

int NetworkManager::report(int thread, int interval, NeuronManager* neuron) {
    vector<int>& label = neuron->label();
    IMatrix& output = neuron->output(_output_key);
    _train_num += label.size();
    for(size_t i = 0; i < label.size(); i++){
        real_t max = output.col(i).maxCoeff();
        if(abs(max-output(label[i],i))<0.00001 ) _right_num+=1;
    }
    if(_train_num % interval == 0){
        fprintf(stdout,"thread:%d, train_num:%d , right_num:%d, acc:%f \n", 
                thread, _train_num, _right_num, _right_num*1.0/_train_num);
    }
    
}

} /* namespace nlpnet */
