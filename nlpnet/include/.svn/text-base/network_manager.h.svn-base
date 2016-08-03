/*
 * network_manager.h
 *
 *  Created on: 2016-3-29
 *      Author: zengzengfeng
 */

#ifndef NETWORK_MANAGER_H_
#define NETWORK_MANAGER_H_
#include <map>
#include <vector>
#include "Layer.h"
namespace nlpnet {

class NetworkManager {
public:
    NetworkManager();
    ~NetworkManager();
    int init_network(const Json::Value& config);
    int forward_propagation(NeuronManager* neuron);
    int backward_propagation(NeuronManager* neuron);
    int update();
    int report(int thread, int interval,NeuronManager* neuron);
private:
    std::map<std::string,BaseLayer*> _layer_dict;
    std::vector<BaseLayer*> _layer_list;
    int _train_num;
    int _right_num;
    std::string _output_key;
};

} /* namespace nlpnet */
#endif /* NETWORK_MANAGER_H_ */
