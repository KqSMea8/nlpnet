/*
 * neuron.h
 *
 *  Created on: 2016-3-25
 *      Author: zengzengfeng
 */

#ifndef NEURON_H_
#define NEURON_H_
#include "json/json.h"
#include "json/reader.h"
#include "json/writer.h"
#include "utils.h"
namespace nlpnet {
class NeuronManager {
public:
    NeuronManager() : _thread_id(0) {
    }

    NeuronManager(int thread_id) : _thread_id(thread_id) {
    }

    int init_neuron(const Json::Value& config){
        Json::Value layers = config["layers"];
        _batch_dim = config["batch_dim"].asInt();
        for(unsigned int i = 0; i < layers.size(); i++){
            Json::Value type = layers[i]["output_dim"];
            int output_dim = layers[i]["output_dim"].asInt();
            Json::Value cons = layers[i]["connector"];
            for(unsigned int j =0 ; j < cons.size(); j++){
                string name = cons[j]["out"].asString();
                _output[name] = IMatrix(output_dim,1);
                _gradient[name] = IMatrix(output_dim,1);
            }
        }

        return SUCCESS;
    }
    IMatrix& gradient(const std::string& key) {
        return _gradient.at(key);
    }

    IMatrix& output(const std::string& key) {
        //std::map<std::string, IMatrix>::iterator iter = _output.find(key);
        return _output.at(key);
    }
    vector<int>& label(){
        return _label;
    }

    inline void set_output(const std::string& key, const IMatrix& mat) {
        _output[key] = mat;
    }

    inline void set_gradient(const std::string key, const IMatrix& grad) {
        _gradient[key] = grad;
    }

    inline void add_gradient(const std::string key, const IMatrix& grad) {
        if (_gradient.find(key) == _gradient.end()) {
            _gradient[key] = grad;
        }
        else {
            _gradient[key] += grad;
        }
    }

    inline int thread_id() const {
        return _thread_id;
    }

    inline void set_thread_id(int tid) {
        _thread_id = tid;
    }

    inline void clear_feature_index(const std::string& key) {
        _feature_index[key].clear();
    }

    inline void append_feature_index(const std::string& key, const std::vector<int>& index) {
        _feature_index[key].push_back(index);
    }

    inline std::vector<int>& feature_index(const std::string& key, int batch_id) {
        //CHECK_LT(batch_id, _feature_index[key].size());
        return _feature_index[key][batch_id];
    }
    inline std::vector<std::vector<int> >& feature_index(const std::string& key) {
        //CHECK_LT(batch_id, _feature_index[key].size());
        if(_feature_index.find(key) == _feature_index.end()){
            _feature_index[key]=vector<vector<int> >();
        }
        return _feature_index[key];
    }
    inline void set_feature_index(const std::string& key,
            const vector<std::vector<int> >& index) {
        _feature_index[key]=index;
    }

    // fast trial to store the softmax probability
    std::map<std::string, real_t> _value;

private:
    // thread id, every calculation thread use one neuron, so need thread_id to separte mem storage
    int _thread_id;
    int _batch_dim;

    // key-value store
    // word embedding index as feature index
    std::map<std::string, std::vector<std::vector<int> > > _feature_index;
    std::vector<int> _label;
    // each layer's output
    std::map<std::string, IMatrix> _output;
    // each layer's gradient
    std::map<std::string, IMatrix> _gradient;
};

}


#endif /* NEURON_H_ */
