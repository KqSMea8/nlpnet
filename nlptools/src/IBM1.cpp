/*
 * IBM1.cpp
 *
 *  Created on: 2014-3-20
 *      Author: zengzengfeng
 */

#include <iostream>
#include <vector>
#include <map>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <sstream>
#include "utils.h"
#include "IBM1.h"
using namespace std;

IBM1::IBM1() {
	// TODO Auto-generated constructor stub

}

IBM1::~IBM1() {
	// TODO Auto-generated destructor stub
}


bool IBM1::rule_align(const string& str1, const string& str2) {
	if (str1 == str2) {
		return true;
	}
	//
	return false;
}

void IBM1::stat_align(vector<string>& tokens1, vector<string>& tokens2,
		int index, float* prob_array) {

	float local_sum = 0.0;
	//align_prob.size()==0 at the first loop
	if (align_prob.size() == 0) {
		local_sum = tokens2.size();
		for (size_t j = 0; j < tokens2.size(); j++) {
			prob_array[j] = 1 / local_sum;
		}
	} else {
		for (size_t j = 0; j < tokens2.size(); j++) {
			string key = tokens1[index] + "|" + tokens2[j];
			prob_array[j] = align_prob[key];
			local_sum += prob_array[j];
		}
		//normalize
		for (size_t j = 0; j < tokens2.size(); j++) {
			prob_array[j] = prob_array[j] / local_sum;
		}

	}
	//adjust weight by word2vec
}
//hl->query
void IBM1::align_one(vector<string>& tokens1, vector<string>& tokens2) {

	int rule_align_bit[tokens1.size()];
	//规则对齐
	for (int i = 0; i < tokens1.size(); i++) {
		rule_align_bit[i] = -1;
		for (int j = 0; j < tokens2.size(); j++) {
			if (rule_align(tokens1[i], tokens2[j])) {
				rule_align_bit[i] = j;
				break;
			}
		}
	}

	//统计对齐
	for (size_t i = 0; i < tokens1.size(); i++) {
		float local_prob[tokens2.size()];
		if (rule_align_bit[i] >= 0) {
			string pair_key = tokens1[i] + "|" + tokens2[rule_align_bit[i]];
			map_plus(align_cnt, pair_key, 1.0);
			map_plus(align_sum, tokens1[i], 1.0);
		} else {
			stat_align(tokens1, tokens2, i, local_prob);
			for (size_t j = 0; j < tokens2.size(); j++) {
				string pair_key = tokens1[i] + "|" + tokens2[j];
				map_plus(align_cnt, pair_key, local_prob[j]);
				//notice！！！
				map_plus(align_sum, tokens2[j], local_prob[j]);
			}
		}

	}

}

void IBM1::cal_align_prob() {
	for (map<string, float>::iterator it = align_cnt.begin();
			it != align_cnt.end(); ++it) {
		vector<string> items;
		split_str(items, it->first, "|");
		it->second = it->second / align_sum[items[0]];
	}
	align_prob = align_cnt;

	//zero the align_cnt
	for (map<string, float>::iterator it = align_cnt.begin();
			it != align_cnt.end(); ++it) {
		it->second = 0.0;
	}
	//zero the align_sum
	for (map<string, float>::iterator it = align_sum.begin();
			it != align_sum.end(); ++it) {
		it->second = 0.0;
	}
}
void IBM1::load_align_prob(const char* filename) {
	ifstream in(filename);
	string line;
	vector<string> items;
	while (getline(cin, line)) {
		split_str(items, line, "\t");
		string key = items[0] + "|" + items[1];
		align_prob[key] = atof(items[2].c_str());
	}

}
void IBM1::save_align_prob(const char* filename) {
	ofstream out(filename);
	for (map<string, float>::iterator it = align_prob.begin();
			it != align_prob.end(); ++it) {
		out << it->first << "\t" << it->second << endl;
	}
}
void IBM1::pipe_run(istream& in) {
	string line;
	vector<string> items;
	vector<string> tokens1;
	vector<string> tokens2;
	while (getline(in, line)) {
		split_str(items, line, "\t");
		split_str(tokens1, items[0], " ");
		split_str(tokens2, items[1], " ");
		align_one(tokens1, tokens2);
	}
	cal_align_prob();

}
void IBM1::disk_run(const char* infile,const char* out_dir,int iter_num){

	for(int i=0;i<iter_num;i++){
		ifstream in(infile);
		ostringstream oss;
		oss<<out_dir<<"/"<<i<<endl;
		pipe_run(in);
		save_align_prob(oss.str().c_str());
		in.close();
	}


}
void IBM1::memory_run(const char* infile,const char* out_dir,int iter_num){

	ifstream in(infile);
	string line;
	vector<string> items;
	vector<vector<string> >  list1;
	vector<vector<string> >  list2;
	vector<string> tokens1;
	vector<string> tokens2;
	while (getline(cin, line)) {
		split_str(items, line, "\t");
		split_str(tokens1, items[0], " ");
		split_str(tokens2, items[1], " ");
		list1.push_back(tokens1);
		list2.push_back(tokens2);
	}
	for(int i=0;i<iter_num;i++){
		for(size_t j=0;i<list1.size();j++){
			align_one(list1[i], list2[i]);
		}
		ostringstream oss;
		oss<<out_dir<<"/"<<i<<endl;
		save_align_prob(oss.str().c_str());
	}
	in.close();

}
