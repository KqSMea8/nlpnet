// string constructor
#include <iostream>
#include <string>
#include <Eigen/Eigen>
#include <pthread.h>
#include "json/json.h"
#include "json/reader.h"
#include "json/writer.h"
#include "optimizer.h"
#include "utils.h"
#include "network_manager.h"
#include "grad_check.h"
#include "file_spliter.h"
#include "data_generator.h"
using namespace Eigen;
using namespace std;
using namespace nlpnet;

int thread_num = 1;
string train_file;
string test_file;
Json::Value model_config;




void* threadRun(void *id) {

    fprintf(stderr,"threadRun%ld\n",long(id));
    NetworkManager network_mgr;
    NeuronManager neuron_mgr;
    network_mgr.init_network(model_config);
    neuron_mgr.init_neuron(model_config);

    FileSpliter file_spliter(train_file, thread_num, long(id));
    DataGenerator gen;
    string data_key = model_config["data"].asString();
    vector<vector<int> >& batch_x = neuron_mgr.feature_index(data_key);
    vector<int>& batch_y = neuron_mgr.label();

    int batch_size = 1;

    while (gen.get_batch(file_spliter,batch_x,batch_y,batch_size) == SUCCESS) {
        ostringstream ss;
        for(size_t i=0; i<batch_x.size(); i++){
            for(size_t j=0; j<batch_x[i].size();j++){
                ss << batch_x[i][j] << " ";
            }
            ss << batch_y[i] << endl;
        }
        //fprintf(stdout,"input:%s",ss.str().c_str());
        network_mgr.forward_propagation(&neuron_mgr);
        network_mgr.backward_propagation(&neuron_mgr);
        network_mgr.report(long(id),1,&neuron_mgr);
        network_mgr.update();
    }

    //fprintf(stderr,"acc:%f,line_num:%d",right_num / line_num, line_num);
    return NULL;
}

void TrainModel(int iter_num) {
    ModelManager::create_model(model_config);
    for (size_t i = 0; i < iter_num; i++) {
        fprintf(stderr,"#### iteration:%d\n",i);
        pthread_t *pt = (pthread_t *) malloc(thread_num * sizeof(pthread_t));
        for (int a = 0; a < thread_num; a++) {
            pthread_create(&pt[a], NULL, threadRun, (void *) a);
        }

        for (int a = 0; a < thread_num; a++) {
            pthread_join(pt[a], NULL);
        }

    }

}

void gradient_check() {

    ModelManager::create_model(model_config);
    GradChecker::create_grad(model_config);

    NetworkManager network_mgr;
    NeuronManager neuron_mgr;
    network_mgr.init_network(model_config);
    neuron_mgr.init_neuron(model_config);

    FileSpliter file_spliter(train_file, 1, 0);
    DataGenerator gen;
    string data_key = model_config["data"].asString();
    vector<vector<int> >& batch_x = neuron_mgr.feature_index(data_key);
    vector<int>& batch_y = neuron_mgr.label();

    int batch_size = 1;
    gen.get_batch(file_spliter, batch_x, batch_y, batch_size);
    network_mgr.forward_propagation(&neuron_mgr);
    network_mgr.backward_propagation(&neuron_mgr);
    network_mgr.update();

    for (map<string, MatrixPtr>::iterator it = ModelManager::_model.begin();
        it != ModelManager::_model.end(); ++it) {
        IMatrix& w_grad = *GradChecker::get_grad_ptr(it->first);
        cout << "######################################### gradient" << endl;
        //cout << it->first << endl << w_grad << endl;

    }
    real_t h = 1e-4;
    for (map<string, MatrixPtr>::iterator it = ModelManager::_model.begin();
        it != ModelManager::_model.end(); ++it) {
        IMatrix& w = *it->second;
        IMatrix& w_grad = *GradChecker::get_grad_ptr(it->first);
        for (size_t j = 0; j < w.cols(); j++) {
            for (size_t i = 0; i < w.rows(); i++) {
                w(i, j) = w(i, j) + h;
                network_mgr.forward_propagation(&neuron_mgr);
                real_t cost1 = neuron_mgr.output("loss")(0, 0);
                w(i, j) = w(i, j) - 2 * h;
                network_mgr.forward_propagation(&neuron_mgr);
                real_t cost2 = neuron_mgr.output("loss")(0, 0);
                real_t num_grad = (cost1 - cost2) / (2 * h);
                w(i, j) = w(i, j) + h;
                bool error = abs(num_grad - w_grad(i, j)) > 0.0001;
                if (error) {
                    cout << "gradient error:" << it->first << endl;
                }
                //if (i == 0 && j == 0) {
                if (w_grad(i, j)>0) {
                    cout << it->first << " num gradient x:" << num_grad
                        << " gradient x:" << w_grad(i, j) << " cost1: "
                        << cost1 << " cost2: " << cost2 << endl;
                }

            }
        }
    }

}

int main(int argc, char **argv) {

    if (argc == 1) {
        printf("Options:\n");
        printf("Parameters for training:\n");
        printf("\t-train <file>\n");
        printf("\t\tUse text data from <file> to train the model\n");
        printf("\t-conf <file>\n");
        printf("\t\tUse json file to configure the model\n");
        printf("\t-gradcheck <int>\n");
        printf("\t\t gradient-check (default 0)\n");
        printf("\t-iter <int>\n");
        printf("\t\t iteration (default 1)\n");
        printf("\t-thread <int>\n");
        printf("\t\t thread (default 1)\n");
        return -1;
    }

    std::string conf_file;
    int grad_check = 0;
    int i=0;
    int iter=1;
    if ((i = arg_pos((char *)"-conf", argc, argv)) > 0) conf_file = argv[i + 1];
    if ((i = arg_pos((char *)"-train", argc, argv)) > 0) train_file=argv[i + 1];
    if ((i = arg_pos((char *)"-gradcheck", argc, argv)) > 0) grad_check=atoi(argv[i + 1]);
    if ((i = arg_pos((char *)"-iter", argc, argv)) > 0) iter=atoi(argv[i + 1]);
    if ((i = arg_pos((char *)"-thread", argc, argv)) > 0) thread_num=atoi(argv[i + 1]);
    Json::Reader reader;
    std::ifstream is;
    is.open(conf_file.c_str(), std::ios::binary);
    if (!reader.parse(is, model_config, false)) {
        fprintf(stderr,"read the config fail\n");
        return -1;
    }
    if(grad_check>0){
        gradient_check();
        return 0;
    }

    TrainModel(iter);
}

