/*
 * file_spliter.cpp
 *
 *  Created on: 2016-3-31
 *      Author: zengzengfeng
 */

#include "file_spliter.h"
#include "utils.h"


namespace nlpnet {

FileSpliter::FileSpliter(const std::string& file_name, int split_num, int index) {
    // TODO Auto-generated constructor stub
    std::ifstream fin(file_name.c_str());
    fin.seekg(0, std::ios::end);
    long long file_size = fin.tellg();
    fin.close();

    _fin_ptr = new std::ifstream(file_name.c_str());
    long long start_index = file_size / (long long) split_num * (long long) index;
    _fin_ptr->seekg(start_index);
    _size = file_size / split_num;
     fprintf(stderr,"file_size:%ld start_index:%ld _size:%ld\n",file_size, start_index,_size);
    _cur_size =0;
    std::string line;
    if( index != 0) {
        //skip the first line, avoid read the line repeatly;
        std::string line;
        std::getline(*_fin_ptr, line);
    }
}

int FileSpliter::getline(std::string& line){
    if(_cur_size>_size){
        _fin_ptr->close();
        return FAILED;
    }
    if(std::getline(*_fin_ptr, line)){
        _cur_size += line.length();
        return SUCCESS;
    }else{
        _fin_ptr->close();
        return FAILED;
    }

}

FileSpliter::~FileSpliter() {
    // TODO Auto-generated destructor stub
    if(_fin_ptr != NULL) delete _fin_ptr;
}

} /* namespace nlpnet */
