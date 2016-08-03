#ifndef ST_NLPNET_OPTIMIZER_H
#define ST_NLPNET_OPTIMIZER_H

#include <string>
#include "json/json.h"
#include "json/reader.h"
#include "json/writer.h"
#include "utils.h"
using namespace Eigen;

namespace nlpnet {

class Optimizer {
public:
    Optimizer(const Json::Value&  opt_config)  {
            _update_count = 0;
            _opt_config = opt_config;
            _learning_rate = opt_config["learning_rate"].asFloat();
            _batch_size =  opt_config["batch_size"].asInt();
    }

    virtual ~Optimizer() {
    }
    
    virtual void register_model(const std::string& model_name, MatrixPtr model) = 0;

    /*! dense update interface, eg. fully connected layer and bias layer */
    virtual void update(
            const std::string& model_name, 
            const IMatrix& grad, 
            MatrixPtr model) = 0;

    /*! sparse update interface, eg. embedding layer */
    virtual void update(
            const std::string& model_name, 
            int row_id, 
            const IVector& grad, 
            MatrixPtr model) = 0;

    // report the gradient
    virtual void report() = 0; 

    /*! get the learning rate */
    real_t learning_rate() {
        // learning rate decay
        /*if (++_update_count > _update_threshold) {
            _update_count = 0;
            _learning_rate *= _decay_factor;
            std::cerr << "[Optimizer]learning rate decay to " << _learning_rate << std::endl;
        }*/
        
        return _learning_rate;
    }

    static Optimizer* create(const Json::Value& opt_config);

protected:
    /*! l2 regularizer as default */
    void regularize(MatrixPtr model) {
        *model -= _lambda * (*model);
    }

    /*! l2 regularizer for sparse vector */
    void regularize(MatrixPtr model, int col_id) {
        model->col(col_id) -= _lambda * model->col(col_id);
    }

    /*! clip array's absolute value to [min, max] */
    void clip(real_t* x, int n, real_t min_val, real_t max_val) {
        for (int i = 0; i < (n / 4) * 4; i += 4) {
            x[i + 0] = std::min(max_val, std::max(min_val, x[i + 0]));
            x[i + 1] = std::min(max_val, std::max(min_val, x[i + 1]));
            x[i + 2] = std::min(max_val, std::max(min_val, x[i + 2]));
            x[i + 3] = std::min(max_val, std::max(min_val, x[i + 3]));
        }
        for (int i = (n / 4) * 4; i < n; ++i) {
            x[i] = std::min(max_val, std::max(min_val, x[i]));
        }
    }

    real_t _learning_rate; // learning rate
    int _batch_size;

    real_t _lambda;     // regularization factor
    real_t _decay_factor; // learning rate decay factor

    // for gradient clipping
    real_t _min_clip;
    real_t _max_clip;

    // maintain update count inside optimizer to triiger learning rate decay
    volatile size_t _update_count;  // locally maintain update count
    size_t _update_threshold; // threshold from config

    Json::Value _opt_config;
};
} // namespace nlpnet
#endif
