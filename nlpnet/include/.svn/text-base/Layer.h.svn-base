/*
 * Layer.h
 *
 *  Created on: 2015-12-30
 *      Author: zengzengfeng
 */

#ifndef Layers_H_
#define Layers_H_

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
#include "neuron_manager.h"
#include "json/json.h"
#include "json/reader.h"
#include "json/writer.h"
#include "model_manager.h"
#include "optimizer.h"

using namespace std;
using namespace Eigen;

extern int GLOBAL_thread_num;
namespace nlpnet {

class BaseLayer {
public:
    BaseLayer() :
        _in_dim(0), _out_dim(0), _batch_dim(0) {
    }
    virtual void Init(const Json::Value& config) {
        _layer_config = config;
        _in_dim = config["input_dim"].asInt();
        _out_dim = config["output_dim"].asInt();
        _batch_dim = config["batch_dim"].asInt();
        _connector = _layer_config["connector"];
    }
    virtual ~BaseLayer() {
    }
    virtual int Forward(NeuronManager* neuron) {
        return SUCCESS;
    }
    virtual int Backward(NeuronManager* neuron) {
        return SUCCESS;
    }
    virtual int Update() {
        return SUCCESS;
    }

public:
    Json::Value _layer_config;
    Json::Value _connector;
    int _in_dim;
    int _out_dim;
    int _batch_dim;

};

class ConcEmbLayer: public BaseLayer {
public:
    ConcEmbLayer() {
    }
    void Init(const Json::Value& config) {
        BaseLayer::Init(config);
        _layer_config = config;
        Json::Value model_cfg = config["model_config"];
        _emb_key = model_cfg[0]["key"].asString();
        _emb = ModelManager::get_model_ptr(_emb_key);
        //_emb_grad = IMatrix::Zero(_out_dim, 1);
        //cout << "_in_dim:" << _in_dim << endl;
        //cout << "_batch_dim:" << _batch_dim << endl;
        _emb_grad = std::vector<IVector>(_in_dim * _batch_dim,
            IVector::Zero(_emb->rows()));
        //_update_index = std::vector<int>(_in_dim * _batch_dim);
        _optimizer = Optimizer::create(config["opt_config"]);
        _optimizer->register_model(model_cfg[0]["key"].asString(), _emb);

    }

    virtual ~ConcEmbLayer() {
    }

    int Forward(NeuronManager* neuron) {
        //fprintf(stdout,"conc_emb forward,rows:%d,cols:%d\n",_emb->rows(),_emb->cols());
        size_t emb_vec_size = _emb->rows();
        for (unsigned int i = 0; i < _connector.size(); i++) {
            //IMatrix& input = neuron->output(_connector[i]["in"].asString());
            IMatrix& x = neuron->output(_connector[i]["out"].asString());
            for (int b = 0; b < _batch_dim; ++b) {
                // concatenation of all the word embeeding
                vector<int>& wid = neuron->feature_index(
                    _connector[i]["in"].asString(), b);
                for (size_t a = 0; a < wid.size(); a++) {
                    size_t begin = emb_vec_size * a;
                    x.col(b).segment(begin, emb_vec_size) = _emb->col(wid[a]);
                    //x.col(b).noalias() += _emb->col(wid[a]);
                }
            }
        }
        return SUCCESS;
    }

    int Backward(NeuronManager* neuron) {
        _update_index.clear();
        size_t emb_vec_size = _emb->rows();
        for (unsigned int i = 0; i < _connector.size(); i++) {
            string in = _connector[i]["in"].asString();
            string out = _connector[i]["out"].asString();
            IMatrix& gradient = neuron->gradient(out);
            //cout << "emb gradient key:" << out << endl;
            for (int b = 0; b < _batch_dim; ++b) {
                IVector g = gradient.col(b);
                //cout << "emb grad:" << endl << g.transpose() << endl;
                // LOG(INFO) << "bow emb grad = " << g.transpose();
                vector<int>& wid = neuron->feature_index(in, b);
                for (size_t j = 0; j < wid.size(); j++) {
                    size_t begin = emb_vec_size * j;
                    //cout << "begin:" << begin << endl;
                    //cout << "_update_index.size():" << _update_index.size() << endl;
                    //cout << "_emb_grad.size():" << _emb_grad.size() << endl;
                    //cout <<  _emb_grad[_update_index.size()] << endl;

                    _emb_grad[_update_index.size()] = g.segment(begin,
                        emb_vec_size);
                    _update_index.push_back(wid[j]);

                }

            }
        }
        return SUCCESS;
    }

    int Update() {

        for (size_t i = 0; i < _update_index.size(); i++) {
            _optimizer->update(_emb_key, _update_index[i], _emb_grad[i], _emb);
            _emb_grad[i].setZero();
        }
        _update_index.clear();
        return SUCCESS;
    }
private:
    MatrixPtr _emb;
    std::string _emb_key;
    std::vector<IVector> _emb_grad;
    Optimizer* _optimizer;
    std::vector<int> _update_index;

};

class ConvEmbLayer: public BaseLayer {
public:
    ConvEmbLayer() {
    }
    void Init(const Json::Value& config) {
        BaseLayer::Init(config);
        _layer_config = config;
        Json::Value model_cfg = config["model_config"];
        _window = config["window"].asInt();
        _emb_key = model_cfg[0]["key"].asString();
        _weight_key = model_cfg[1]["key"].asString();
        _bias_key = model_cfg[2]["key"].asString();

        _emb = ModelManager::get_model_ptr(_emb_key);
        _W = ModelManager::get_model_ptr(_weight_key);
        _bias = ModelManager::get_model_ptr(_bias_key);

        int max_tokens = 20;
        _emb_grad = std::vector<IVector>(max_tokens * _batch_dim,
            IVector::Zero(_emb->rows()));
        _tmp_grad = std::vector<IVector>(max_tokens,
            IVector::Zero(_emb->rows()));
        _W_grad = IMatrix::Zero(_out_dim, _in_dim);
        _bias_grad = IMatrix::Zero(_out_dim, 1);
        _filter = IMatrix::Zero(_out_dim, _batch_dim);

        _win_vec = IMatrix::Zero(_emb->rows() * _window, 1);
        _max_ids = std::vector<std::vector<int> >(_batch_dim);

        _optimizer = Optimizer::create(config["opt_config"]);
        _optimizer->register_model(_emb_key, _emb);
        _optimizer->register_model(_weight_key, _W);
        _optimizer->register_model(_bias_key, _bias);
    }

    virtual ~ConvEmbLayer() {
    }

    void max_pool(IMatrix& pool, IMatrix& filter, int batch, int id, vector<int>& max_id) {
        if (id == 0) {
            pool.col(batch) = filter.col(batch);
            return;
        }
        for (size_t i = 0; i < pool.rows(); i++) {
            if (filter(i,batch) > pool(i,batch)) {
                pool(i,batch) = filter(i,batch);
                max_id[i] = id;
            }
        }
    }

    void setZero(vector<int>& vec, size_t size){
        vec.clear();
        for(size_t i=0; i<size; i++){
            vec.push_back(0);
        }
    }

    int Forward(NeuronManager* neuron) {
        //fprintf(stdout,"conc_emb forward,rows:%d,cols:%d\n",_emb->rows(),_emb->cols());
        size_t emb_vec_size = _emb->rows();
        for (unsigned int i = 0; i < _connector.size(); i++) {
            IMatrix& output = neuron->output(_connector[i]["out"].asString());
            for (int b = 0; b < _batch_dim; ++b) {
                // Convolution
                vector<int>& tokens = neuron->feature_index(
                    _connector[i]["in"].asString(), b);
                setZero(_max_ids[b], tokens.size());
                for (size_t a = 0; a < tokens.size(); a++) {
                    if (a + _window > tokens.size())
                        break;
                    for (size_t j = 0; j < _window; j++) {
                        size_t begin = emb_vec_size * (a + j);
                        _win_vec.segment(begin, emb_vec_size) = _emb->col(
                            tokens[a + j]);
                    }
                    _filter.col(b) = (*_W) * _win_vec + (*_bias);
                    max_pool(output, _filter, b, a, _max_ids[b]);
                }
            }
        }
        return SUCCESS;
    }

    int Backward(NeuronManager* neuron) {
        _update_index.clear();
        size_t emb_vec_size = _emb->rows();
        for (unsigned int i = 0; i < _connector.size(); i++) {
            string in = _connector[i]["in"].asString();
            string out = _connector[i]["out"].asString();
            IMatrix& gradient = neuron->gradient(out);
            for (int b = 0; b < _batch_dim; ++b) {
                // compute the gradient of tokens
                IVector g = gradient.col(b);
                for (size_t j = 0; j < _W->cols(); j++) {
                    int index_in_window = j / _window;
                    for (size_t k = 0; k < _W->rows(); k++) {
                        int token_id = _max_ids[b][k] + index_in_window;
                        IVector& grad = _tmp_grad[token_id];
                        grad(j % _window, 0) += g[k] * (*_W)(k, j);
                    }
                }
                vector<int>& tokens = neuron->feature_index(
                    _connector[i]["in"].asString(), b);
                for (size_t j = 0; j < tokens.size(); j++) {
                    _emb_grad[_update_index.size()] = _tmp_grad[j];
                    _tmp_grad[j].setZero();
                    _update_index.push_back(tokens[i]);
                }

                //compute gradient of _W
                for (size_t j = 0; j < _W->cols(); j++) {
                    int index_in_window = j / _window;
                    for (size_t k = 0; k < _W->rows(); k++) {
                        int token_id = _max_ids[b][k] + index_in_window;
                        real_t val = (*_emb)(j % _window, tokens[token_id]);
                        _W_grad(k, j) = g[k] * val;
                    }
                }
                _bias_grad += g;
            }
        }
        return SUCCESS;
    }

    int Update() {

        _optimizer->update(_weight_key, _W_grad, _W);
        _W_grad.setZero();

        _optimizer->update(_bias_key, _bias_grad, _bias);
        _bias_grad.setZero();

        for (size_t i = 0; i < _update_index.size(); i++) {
            _optimizer->update(_emb_key, _update_index[i], _emb_grad[i], _emb);
            _emb_grad[i].setZero();
        }
        _update_index.clear();

        return SUCCESS;
    }
private:
    std::string _emb_key;
    std::string _weight_key;
    std::string _bias_key;

    MatrixPtr _emb;
    MatrixPtr _W;
    MatrixPtr _bias;

    std::vector<IVector> _emb_grad;
    std::vector<IVector> _tmp_grad;
    IMatrix _W_grad;
    IMatrix _bias_grad;

    IMatrix _filter;
    IMatrix _pooling;
    Optimizer* _optimizer;
    std::vector<int> _update_index;
    std::vector<std::vector<int> > _max_ids;

    int _window;
    IVector _win_vec;
};

class FcLayer: public BaseLayer {
public:
    FcLayer() {
    }

    virtual ~FcLayer() {

    }

    void Init(const Json::Value& config) {
        BaseLayer::Init(config);
        _layer_config = config;
        Json::Value model_cfg = config["model_config"];
        _weight_key = model_cfg[0]["key"].asString();
        _bias_key = model_cfg[1]["key"].asString();
        _W = ModelManager::get_model_ptr(_weight_key);
        _bias = ModelManager::get_model_ptr(_bias_key);

        _W_grad = IMatrix::Zero(_out_dim, _in_dim);
        _bias_grad = IMatrix::Zero(_out_dim, 1);

        _optimizer = Optimizer::create(config["opt_config"]);
        _optimizer->register_model(_bias_key, _bias);
        _optimizer->register_model(_weight_key, _W);
    }

    int Forward(NeuronManager* neuron) {

        for (unsigned int i = 0; i < _connector.size(); i++) {
            IMatrix& input = neuron->output(_connector[i]["in"].asString());
            IMatrix& output = neuron->output(_connector[i]["out"].asString());
            output = (*_W) * input + (*_bias);
            for (size_t i = 0; i < output.rows(); i++) {
                real_t a = output(i, 0);
                output(i, 0) = active(a);
            }
            //cout << output << endl;
        }
        return SUCCESS;

    }

    int Backward(NeuronManager* neuron) {

        for (unsigned int i = 0; i < _connector.size(); i++) {
            string in = _connector[i]["in"].asString();
            string out = _connector[i]["out"].asString();

            IMatrix& input = neuron->output(in);
            IMatrix& output = neuron->output(out);
            IMatrix& in_grad = neuron->gradient(in);
            IMatrix& out_grad = neuron->gradient(out);

            //col-major
            for (size_t i = 0; i < output.rows(); i++) {
                real_t a = output(i, 0);
                real_t g = acti_der(a);
                out_grad(i, 0) = g * out_grad(i, 0);
            }

            for (size_t i = 0; i < _W_grad.cols(); i++) {
                real_t a = input(i, 0);
                _W_grad.col(i) = out_grad * a;
            }
            _bias_grad = out_grad;

            for (size_t i = 0; i < in_grad.rows(); i++) {
                in_grad(i, 0) = out_grad.col(0).dot((*_W).col(i));
            }

            //cout << in_grad.transpose() << endl;
        }
        return SUCCESS;

    }

    int Update() {
        _optimizer->update(_weight_key, _W_grad, _W);
        _W_grad.setZero();

        _optimizer->update(_bias_key, _bias_grad, _bias);
        _bias_grad.setZero();

        return SUCCESS;
    }

private:
    inline real_t active(real_t arg) {
        if (arg < -6)
            return -1.0;
        else if (arg > 6)
            return 1.0;
        return tanh(arg);
    }
    inline real_t acti_der(real_t sigma) {
        return 1 - (sigma * sigma);
    }

public:
    std::string _weight_key;
    std::string _bias_key;
    MatrixPtr _W;
    MatrixPtr _bias;

    IMatrix _W_grad;
    IMatrix _bias_grad;
    Optimizer* _optimizer;
};

class SoftmaxLogLoss: public BaseLayer {
public:
    SoftmaxLogLoss() {
    }
    void Init(const Json::Value& config) {
        BaseLayer::Init(config);
        _layer_config = config;

    }

    virtual ~SoftmaxLogLoss() {

    }

    int Forward(NeuronManager* neuron) {

        IMatrix& input = neuron->output(_connector[0]["in"].asString());
        IMatrix& output = neuron->output(_connector[0]["out"].asString());

        IMatrix& loss = neuron->output(_connector[1]["out"].asString());
        vector<int>& label = neuron->label();
        loss(0, 0) = 0.0;

        for (int b = 0; b < _batch_dim; b++) {
            real_t max_val = input.col(b).maxCoeff();
            real_t sum = 0.0;
            for (size_t i = 0; i < input.rows(); i++) {
                real_t z = input(i, b) - max_val;
                output(i, b) = exp(z);
                sum += output(i, b);
            }
            for (size_t i = 0; i < output.rows(); i++) {
                output(i, b) = output(i, b) / sum;
            }
            loss(0, 0) += -log(output(label[b], b));
            //cout << "label:" << label[b] << "prob:" << output(label[b],b) << endl; 
        }

        return SUCCESS;
    }

    int Backward(NeuronManager* neuron) {
        IMatrix& in_grad = neuron->gradient(_connector[0]["in"].asString());
        IMatrix& output = neuron->output(_connector[0]["out"].asString());
        in_grad = output;
        vector<int>& label = neuron->label();
        for (size_t i = 0; i < in_grad.cols(); i++) {
            in_grad(label[i], i) = in_grad(label[i], i) - 1.0;
        }
        return SUCCESS;
    }

    int Update() {
        return SUCCESS;
    }
};

}
#endif /* Layers_H_ */
