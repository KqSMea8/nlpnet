/*
 * data_generator.h
 *
 *  Created on: 2016-4-1
 *      Author: zengzengfeng
 */

#ifndef DATA_GENERATOR_H_
#define DATA_GENERATOR_H_
#include <vector>
#include <string>
#include "file_spliter.h"
#include "utils.h"
namespace nlpnet{


class DataGenerator{
public:
    DataGenerator(){

    }
    ~DataGenerator(){

    }
    int get_batch(FileSpliter& file, std::vector<std::vector<int> >& batch_x,
                    std::vector<int>& batch_y, size_t batch_size)
    {
        batch_y.clear();
        batch_x.clear();
        for(size_t i=0;i< batch_size; i++){
            _word_index.clear();
            if(file.getline(_line)<0){
                return FAILED;
            }else{
                split_str(_tokens,_line," ");
                for(size_t i=0; i<_tokens.size()-1; i++){
                    _word_index.push_back(atoi(_tokens[i].c_str()));
                }
                batch_x.push_back(_word_index);
                batch_y.push_back(atoi(_tokens[_tokens.size()-1].c_str()));
            }
        }
        return SUCCESS;
    }
private:
    std::vector<std::string> _tokens;
    std::vector<int> _word_index;
    std::string _line;
};
}
#endif /* DATA_GENERATOR_H_ */
