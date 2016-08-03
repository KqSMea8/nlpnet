/******************************************************************
 * 
 * Copyright (c) 2015 Baidu.com, Inc. All Rights Reserved
 * 
 ******************************************************************/


#ifndef ST_NLPNET_ADAGRAD_OPTIMIZER_H
#define ST_NLPNET_ADAGRAD_OPTIMIZER_H

#include <atomic>
#include <mutex>

#include "optimizer.h"
#include "utils.h"

namespace nlpnet {

class AdagradOptimizer : public Optimizer {
public:
    AdagradOptimizer(const Json::Value& opt_config) :
        Optimizer(opt_config), 
        _accumulation_count(0) {

    }

    void register_model(const std::string& model_key, MatrixPtr model) {
        // initial grad sum = 1
        _grad_sum[model_key] = IMatrix::Ones(
                model->rows(), 
                model->cols());    
        _grad_small_sum[model_key] = IMatrix::Zero(
                model->rows(), 
                model->cols());
    }

    // update dense model
    void update(
            const std::string& key, 
            const IMatrix& grad, 
            MatrixPtr model) {
        // NOTE: ada_grad = grad / sqrt(_grad_sum)
        // acuumulate the gradient divide with batch size
        regularize(model);
        IVector g = grad; // copy grad to g
        clip(g.data(), g.size(), 1e-8, 1);
        _grad_small_sum[key] += g.cwiseAbs2();

        if (++_accumulation_count % ACCUMULATE_THRESHOLD == 0) {
            _grad_sum[key].noalias() += _grad_small_sum[key];
            _grad_small_sum[key].setZero();
        }
        IMatrix ada_grad = grad.cwiseQuotient(_grad_sum[key].cwiseSqrt());
        model->noalias() -= learning_rate() / _batch_size * ada_grad; 
    }

    // update sparse model
    void update(
            const std::string& key, 
            int col_id, 
            const IVector& grad, 
            MatrixPtr model) {
        regularize(model, col_id);
        IVector g = grad; // copy grad to g
        clip(g.data(), g.size(), 1e-8, 1);
        _grad_small_sum[key].col(col_id) += g.cwiseAbs2();

        if (++_accumulation_count % ACCUMULATE_THRESHOLD == 0) {
            _grad_sum[key].col(col_id).noalias() += _grad_small_sum[key].col(col_id);
            _grad_small_sum[key].col(col_id).setZero();
        }
        IVector ada_grad = grad.cwiseQuotient(_grad_sum[key].col(col_id).cwiseSqrt());
        model->col(col_id).noalias() -= learning_rate() / _batch_size * ada_grad;
    }

    void report() {
    }

    volatile size_t _accumulation_count;
    std::map<std::string, IMatrix> _grad_sum;
    std::map<std::string, IMatrix> _grad_small_sum;

    constexpr static size_t ACCUMULATE_THRESHOLD = 1000;
};
} // namespace eigennet
#endif
