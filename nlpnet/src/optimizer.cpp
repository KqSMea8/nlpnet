#include "adagrad_optimizer.h"
#include "optimizer.h"
#include "sgd_optimizer.h"


namespace nlpnet {

Optimizer* Optimizer::create(const Json::Value&  opt_config) {
    // choosing learning method
    if (opt_config["learning_method"] == "sgd") {
        //LOG(INFO) << "create sgd optimizer";
        return new SGDOptimizer(opt_config);
    } else if (opt_config["learning_method"] == "adagrad") {
        //LOG(INFO) << "create adagrad optimizer";
        return new AdagradOptimizer(opt_config);
    } else {
        //LOG(ERROR) << "learning method: " << opt_config.learning_method() << " not existed!";
    }

    return nullptr;
}
} // namespace eigennet
