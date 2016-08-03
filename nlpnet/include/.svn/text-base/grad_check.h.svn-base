/*
 * grad_check.h
 *
 *  Created on: 2016-4-10
 *      Author: zengzengfeng
 */

#ifndef GRAD_CHECK_H_
#define GRAD_CHECK_H_



#include <fstream>
#include <iostream>
#include <algorithm>
#include <atomic>
#include <fstream>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include "utils.h"
#include "json/json.h"
#include "json/reader.h"
#include "json/writer.h"

namespace nlpnet {
class GradChecker {
public:
    GradChecker() {
    }

    virtual ~GradChecker() {
    }

    // NOTE: call create_grad before get_grad_ptr
    static int create_grad(const Json::Value&  config) {
        Json::Value network_config = config["layers"];
        for(unsigned int i = 0; i < network_config.size(); i++){
            Json::Value model_cfg = network_config[i][MODEL_CONFIG];
            for(unsigned int j = 0; j < model_cfg.size(); j++){
                register_grad(model_cfg[j]["key"].asString(),
                              model_cfg[j]["row_dim"].asInt(),
                              model_cfg[j]["col_dim"].asInt(),
                              model_cfg[j]["init_range"].asFloat()
                );
            }
        }
        return SUCCESS;
    }

    static void register_grad(const std::string& model_key, int row, int col, real_t init_range) {
       if (_grad.find(model_key) != _grad.end()) {
            return;
        }
        IMatrix* m = new IMatrix(row, col);
        m->setZero();
        _grad[model_key] = MatrixPtr(m);
    }

    static MatrixPtr get_grad_ptr(const std::string& model_key) {
        return _grad.at(model_key);
    }

    // use STL map as memory manager
    static std::map<std::string, MatrixPtr> _grad;
};
} // namespace nlpnet

#endif /* GRAD_CHECK_H_ */
