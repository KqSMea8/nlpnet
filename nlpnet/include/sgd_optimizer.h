#ifndef ST_NLPNET_SGD_OPTIMIZER_H
#define ST_NLPNET_SGD_OPTIMIZER_H

#include <mutex>
#include "optimizer.h"
#include "grad_check.h"
#include "utils.h"


namespace nlpnet {

class SGDOptimizer : public Optimizer {
public:
    SGDOptimizer(const Json::Value &opt_config) : Optimizer(opt_config) {
    }

    virtual ~SGDOptimizer() {
    }

    void register_model(const std::string& model_name, MatrixPtr model) {
        (void) model;
        _model_name = model_name;
    }

    // update dense model
    void update(
            const std::string& model_key,
            const IMatrix& gradient, 
            MatrixPtr model) {
        if(GradChecker::_grad.size()>0){
            MatrixPtr ptr = GradChecker::get_grad_ptr(model_key);
            *ptr = gradient ;
            return; // do not update model, when gradient check
        }
        regularize(model);
        model->noalias() -= learning_rate() / _batch_size * gradient;
    }

    // update sparse model
    void update(
            const std::string& model_key,
            int col_id, 
            const IVector& gradient, 
            MatrixPtr model) {
        if(GradChecker::_grad.size()>0){
            MatrixPtr ptr = GradChecker::get_grad_ptr(model_key);
            //ptr->col(col_id).noalias() += gradient;
            ptr->col(col_id) = gradient;
            cout << "model key:" << model_key << endl;
            cout << "update spasre:" << endl << ptr->col(col_id).transpose() << endl;
            return; // do not update model, when gradient check
        }
        regularize(model, col_id);
        model->col(col_id).noalias() -= learning_rate() / _batch_size * gradient;
    }

    void report() {
        // pass
    }

private:
    std::string _model_name;
}; 

// SGD with momentum optimizer
class SGDMomentumOptimizer : public Optimizer {
public:
    SGDMomentumOptimizer(const Json::Value& opt_config) : Optimizer(opt_config) {
    }

    virtual ~SGDMomentumOptimizer() {
    }

    void register_model(const std::string& model_name, MatrixPtr model) {
        _model_name = model_name;
        _velocity[model_name] = IMatrix::Zero(model->rows(), model->cols());
    }

    // update dense model
    void update(
            const std::string& key, 
            const IMatrix& gradient, 
            MatrixPtr model) {
        regularize(model);
        _velocity[key] = _mu * _velocity[key] + gradient; 
        model->noalias() -= learning_rate() / _batch_size * _velocity[key];
    }

    // update sparse model
    void update(
            const std::string& key, 
            int col_id, 
            const IVector& gradient, 
            MatrixPtr model) {
        regularize(model, col_id);
        _velocity[key].col(col_id) = _mu * _velocity[key].col(col_id) + gradient; 
        model->col(col_id).noalias() -= learning_rate() / _batch_size *  _velocity[key].col(col_id);
    }

    void report() {
        // TODO: print history velocity
    }

private:
    // maintain velocity for momentum update
    std::map<std::string, IMatrix> _velocity;
    std::string _model_name;

    constexpr static real_t _mu = 0.9;
}; 

// SGD with gradient clipping for RNN
// NOTE: unfinished
class SGDClipingOptimizer : public Optimizer {
public:
    SGDClipingOptimizer(const Json::Value &opt_config) : Optimizer(opt_config) {
        _update_count = 0;
    }

    virtual ~SGDClipingOptimizer() {
    }

    void register_model(const std::string &model_name, MatrixPtr model) {
        (void) model;
        _model_name = model_name;
    }

    // update dense model
    void update(
            const std::string& model_key,
            const IMatrix& gradient, 
            MatrixPtr model) {
        (void) model_key;
        // l2-regularization
        regularize(model);
        IMatrix g_clip = gradient; // copy gradient
        clip(g_clip.data(), g_clip.size(), _min_clip, _max_clip); // clip gradient
        model->noalias() -= learning_rate() / _batch_size * g_clip;
    }

    // update sparse model
    void update(
            const std::string& model_key,
            int col_id, 
            const IVector &gradient, 
            MatrixPtr model) {
        (void) model_key;
        // l2-regularization
        regularize(model, col_id);

        IMatrix g_clip = gradient; // copy gradient
        clip(g_clip.data(), g_clip.size(), _min_clip, _max_clip); // clip gradient
        model->col(col_id).noalias() -= learning_rate() / _batch_size * g_clip;
    }

    void report() {
        // pass
    }

    std::string _model_name;
}; 
} // namespace eigennet
#endif
