/******************************************************************
 * 
 * Copyright (c) 2015 Baidu.com, Inc. All Rights Reserved
 * 
 ******************************************************************/


#ifndef ST_NLP_NLPNET_MODEL_MANAGER_H
#define ST_NLP_NLPNET_MODEL_MANAGER_H

#include <fstream>
#include <iostream>
#include <algorithm>
#include <atomic>
#include <fstream>
#include <map>
#include <memory>
//#include <mutex>
//#include <random>
#include <set>
#include <string>
//#include <thread>
#include <vector>
#include <iostream>  
#include <fstream>  
#include <stdlib.h> 
#include "utils.h"
#include "json/json.h"
#include "json/reader.h"
#include "json/writer.h"



namespace nlpnet {
/*! \brief
 * Handle neural network model
 * including create model according to LayerConfig and load/save model
 */
class ModelManager {
public:
    ModelManager() {
    }

    virtual ~ModelManager() {
    }

    // NOTE: call create_model before get_model_ptr
    static int create_model(const Json::Value&  config) {
        /*for (const auto& layer_cfg : network_config.layer_config()) {
            for (const auto& model_cfg : layer_cfg.model_config()) {
                register_model(
                        model_cfg.key(), 
                        model_cfg.row_dim(), 
                        model_cfg.col_dim(), 
                        model_cfg.init_range());
            }
        }*/
        Json::Value network_config = config["layers"];
        for(unsigned int i = 0; i < network_config.size(); i++){
            Json::Value model_cfg = network_config[i][MODEL_CONFIG];
            for(unsigned int j = 0; j < model_cfg.size(); j++){
                register_model(model_cfg[j]["key"].asString(),
                                model_cfg[j]["row_dim"].asInt(),
                               model_cfg[j]["col_dim"].asInt(),
                               model_cfg[j]["init_range"].asFloat()
                );
            }
        }

        return SUCCESS;
    }

    static int load_model(const std::string& model_file,  const Json::Value&  configs) {
        //LOG(INFO) << "ModelManager load model " << model_file;
        std::ifstream fi(model_file.c_str(), std::ios::binary);
        //CHECK(fi) << "failed to read file: " << model_file;
        load_model(fi, configs);
        fi.close();

        return SUCCESS;
    }

    // load model from file according config file
    static int load_model(std::ifstream& fi,  const Json::Value& configs) {
        /*for (const auto& layer_cfg :configs.layer_config()) {
            for (const auto& model_cfg : layer_cfg.model_config()) {
                real_t* model_ptr = get_model_ptr(model_cfg.key())->data();
                fi.read(
                        reinterpret_cast<char *>(model_ptr), 
                        model_cfg.row_dim() * model_cfg.col_dim() * sizeof(real_t));
                LOG(INFO) << "load model " << model_cfg.key() 
                          << " size = " << model_cfg.row_dim() * model_cfg.col_dim();
            }
        }*/

        for(unsigned int i = 0; i < configs.size(); i++){
            Json::Value model_cfg = configs[i][MODEL_CONFIG];
            for(unsigned int j = 0; j < model_cfg.size(); j++){
                std::string key = model_cfg[j]["key"].asString();
                real_t* model_ptr = get_model_ptr(key)->data();
                int row_dim = model_cfg[j]["row_dim"].asInt();
                int col_dim = model_cfg[j]["col_dim"].asInt();
                fi.read(reinterpret_cast<char *>(model_ptr), row_dim * col_dim * sizeof(real_t));
                std::cerr << "read model: " << key << "size = " << row_dim * col_dim << std::endl;
            }
        }

        return SUCCESS;
    }

    static int save_model(const std::string& model_file,  const Json::Value& config) {
        //LOG(INFO) << "ModelManager save model " << model_file;
        std::ofstream fo(model_file.c_str(), std::ios::binary);
        //CHECK(fo) << "failed to write file: " << model_file;
        save_model(fo, config);
        fo.close();
        //LOG(INFO) << "ModelManager save model " << model_file << " done!";

        return SUCCESS;
    }

    static int save_model(std::ofstream& fo,  const Json::Value& configs) {
        /*for (const auto& layer_cfg : config.layer_config()) {
            for (const auto& model_cfg : layer_cfg.model_config()) {
                real_t* model_ptr = get_model_ptr(model_cfg.key())->data();
                fo.write(
                        reinterpret_cast<char *>(model_ptr), 
                        model_cfg.row_dim() * model_cfg.col_dim() * sizeof(real_t));
                LOG(INFO) << "save model " << model_cfg.key() 
                          << " size = " << model_cfg.row_dim() * model_cfg.col_dim();
            }
        }*/
        for(unsigned int i = 0; i < configs.size(); i++){
                 Json::Value model_cfg = configs[i][MODEL_CONFIG];
                 for(unsigned int j = 0; j < model_cfg.size(); j++){
                     std::string key = model_cfg[j]["key"].asString();
                     real_t* model_ptr = get_model_ptr(key)->data();
                     int row_dim = model_cfg[j]["row_dim"].asInt();
                     int col_dim = model_cfg[j]["col_dim"].asInt();
                     fo.write(reinterpret_cast<char *>(model_ptr),
                               row_dim * col_dim * sizeof(real_t));
                     std::cerr << "save model: " << key << "size = " << row_dim * col_dim << std::endl;
                 }
        }

        return SUCCESS;
    }

    static int save_model(const std::string& model_key, std::ofstream& fo) {
        size_t row = get_model_ptr(model_key)->rows();
        size_t col = get_model_ptr(model_key)->cols();
        fo.write(
                reinterpret_cast<char *>(get_model_ptr(model_key)->data()), 
                row * col * sizeof(real_t));

        return SUCCESS;
    }

    static void register_model(const std::string& model_key, int row, int col, real_t init_range) {
       if (_model.find(model_key) != _model.end()) {
            // check model size
            //LOG(ERROR) << "register the model twice!";
            //CHECK_EQ(row, _model[model_key]->rows());
            //CHECK_EQ(col, _model[model_key]->cols());
            return;
        }
        IMatrix* m = new IMatrix(row, col);
        m->setZero();
        std::cerr << "random initialize " << model_key
                  << " (" << -init_range << ", " << init_range << ")" << endl;
        rand_init_array(m->data(), row * col, -init_range, init_range);
        _model[model_key] = MatrixPtr(m);
    }

    static MatrixPtr get_model_ptr(const std::string& model_key) {
        return _model.at(model_key);
    }

    static void report_model( const Json::Value& config) {
        auto report = [](const std::string& name, MatrixPtr p) {
            std::cerr << "[ModelReport]" + name
                << " mean=" << p->mean() 
                << " sum=" << p->sum()
                << " min=" << p->minCoeff()
                << " max=" << p->maxCoeff()
                << " l2-norm=" << sqrt(p->cwiseAbs2().sum() / p->size()) << endl;
        };
        /*for (const auto& layer_cfg : config.layer_config()) {
            for (const auto& model_cfg : layer_cfg.model_config()) {
                report(model_cfg.key(), get_model_ptr(model_cfg.key()));
            }
        }*/

        for (unsigned int i = 0; i < config.size(); i++) {
            Json::Value model_cfg = config[i][MODEL_CONFIG];
            for (unsigned int j = 0; j < model_cfg.size(); j++) {
                std::string key = model_cfg[j]["key"].asString();
                report(key, get_model_ptr(key));
            }
        }
    }

    // use STL map as memory manager
    static std::map<std::string, MatrixPtr> _model; 
};
} // namespace eigennet
#endif
/* vim: set expandtab ts=4 sw=4 sts=4 tw=100: */
