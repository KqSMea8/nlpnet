/*
 * Kmeans.cpp
 *
 *  Created on: 2013-12-9
 *      Author: zengzengfeng
 */

#include "Kmeans.h"
#include <math.h>
#include <queue>
#include <algorithm>
#include <functional>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <iostream>
#include "utils.h"
#include "time.h"

using namespace std;
Kmeans::Kmeans() :
		_data_num(0), _vec_size(0), _multi_label(NULL) {
	// TODO Auto-generated constructor stub

}

Kmeans::~Kmeans() {
	// TODO Auto-generated destructor stub
	if (_multi_label) {
		delete[] _multi_label;
	}
}

void Kmeans::load_data(const char* dataFile) {
	//loaddata
	ifstream in(dataFile);
	string line;
	getline(in, line);
	vector<string> items;
	split_str(items, line, "\t");
	_data_num = atoi(items[0].c_str());
	_vec_size = atoi(items[1].c_str());
	cout << "dataNum:" << _data_num << " vecSize:" << _vec_size << endl;

	_data = vector<float>(_data_num * _vec_size, 0.0);
	vector<float> vec(_vec_size, 0.0);
	long index = 0;
	while (getline(in, line)) {
		split_str(items, line, "\t");
		float sum = 0.0;
		//normalize data
		_tokens.push_back(items[0]);
		for (unsigned i = 0; i < _vec_size; i++) {
			vec[i] = atof(items[i + 1].c_str());
			sum += vec[i] * vec[i];
		}
		for (unsigned i = 0; i < _vec_size; i++) {
			_data[index] = vec[i] / sqrt(sum);
			index++;
		}

	}
	cout << "loading data done..." << endl;
}

void Kmeans::save(const char* outFile) {
	ofstream out(outFile);
	if (_multi_label != NULL) {
		for (int i = 0; i < _data_num; i++) {
			if (_tokens.size() > 0) {
				out << _tokens[i] << "\t";
			}
			for (unsigned j = 0; j < _multi_label->size(); j++) {
				out << _multi_label[i][j].first << "\t"
						<< _multi_label[i][j].second << "\t";
			}
			out << endl;
		}
		return ;
	}

	for (int i = 0; i < _data_num; i++) {
		if (_tokens.size() > 0) {
			out << _tokens[i] << "\t";
		}
		out << _label[i] << endl;
	}
	out.flush();
	out.close();
}

bool compare(const pair<int, float> &v1, const pair<int, float> &v2) {
	return v1.second > v2.second;
}
bool Kmeans::pushPq(vector<pair<int, float> >& pq, pair<int, float> dot,
		unsigned maxNum) {
	if (pq.size() < maxNum) {
		pq.push_back(dot);
		make_heap(pq.begin(), pq.end(), compare);
		return true;
	} else {
		if (dot.second > pq.front().second) {
			pop_heap(pq.begin(), pq.end());
			pq.pop_back();
			pq.push_back(dot);
			push_heap(pq.begin(), pq.end(), compare);
			make_heap(pq.begin(), pq.end(), compare);
			return true;
		} else {
			return false;
		}
	}
}
void Kmeans::cluster(int k, int iter_num) {
	vector<int> pid_list(_data_num);
	_label = vector<int>(_data_num, 0);
	for (int i = 0; i < _data_num; i++) {
		pid_list[i] = i;
	}
	randomize_list(pid_list);
	part_hard_kmeans(pid_list, k, iter_num);
}
void Kmeans::level_cluster(int max_class_size, int branch_num, int iter) {
	vector<int> pid_list(_data_num);
	_label = vector<int>(_data_num, 0);
	for (int i = 0; i < _data_num; i++) {
		pid_list[i] = i;
	}
	randomize_list(pid_list);
	string prefix = "1";
	part_level_kmeans(pid_list, max_class_size, prefix, branch_num, iter);
}
void Kmeans::part_hard_kmeans(vector<int>& pid_list, int k, int iter_num) {

	int closeid;
	int *class_cnt = new int[k];
	float closev, x;
	long a, b, pid, d;

	float* cent = new float[k * _vec_size];
	for (a = 0; a < pid_list.size(); a++) {
		_label[pid_list[a]] = a % k;
	}

	for (int iter = 0; iter < iter_num; iter++) {
		for (b = 0; b < k * _vec_size; b++) {
			cent[b] = 0;
		}
		for (b = 0; b < k; b++) {
			class_cnt[b] = 1;
		}

		//update the center
		for (int i = 0; i < pid_list.size(); i++) {
			pid = pid_list[i];
			for (int d = 0; d < _vec_size; d++) {
				cent[_vec_size * _label[pid] + d] += _data[pid * _vec_size + d];
			}
			class_cnt[_label[pid]]++;
		}
		//normalized the center
		norl_center(k, cent, class_cnt);

		//label the data with the closest cluster id
		for (int i = 0; i < pid_list.size(); i++) {
			pid = pid_list[i];
			closev = -10;
			closeid = 0;
			for (d = 0; d < k; d++) {
				x = 0;
				for (b = 0; b < _vec_size; b++) {
					x += cent[_vec_size * d + b] * _data[pid * _vec_size + b];
				}
				if (x > closev) {
					closev = x;
					closeid = d;
				}
			}
			_label[pid] = closeid;

		}
	}

	delete[] class_cnt;
	delete[] cent;
}
void Kmeans::cal_topk( int top_num) {
	_multi_label = new vector<pair<int, float> > [_data_num];

	//index the class label
	map<int,int> label2id;
	vector<int> id2label;
	int index=0;
	for(int i=0;i<_data_num;i++){
		if(label2id.find(_label[i])==label2id.end()){
			label2id[_label[i]]=index;
			id2label.push_back(_label[i]);
		}
	}
	int k=label2id.size();
	float* cent = new float[k * _vec_size];
	int *class_cnt = new int[k];
	for (int b = 0; b < k * _vec_size; b++) {
		cent[b] = 0;
	}
	for (int b = 0; b < k; b++) {
		class_cnt[b] = 1;
	}
	//update the center
	for (int pid = 0;pid < _data_num; pid++) {
		int label_id = label2id[_label[pid]];
		for (int d = 0; d < _vec_size; d++) {
			cent[_vec_size * label_id + d] += _data[pid * _vec_size + d];
		}
		class_cnt[label_id]++;
	}
	//normalized the center
	norl_center(k, cent, class_cnt);

	//get the top k nearest members
	vector<pair<int, float> > pq;
	for (int i = 0; i < _data_num; i++) {
		//int label_id = single_label[i].second;
		pq.clear();
		for (int d = 0; d < k; d++) {
			float x = 0;
			for (int b = 0; b < _vec_size; b++) {
				x += cent[_vec_size * d + b]
						* _data[i * _vec_size + b];
			}
			int label=id2label[d];
			pair<int, float> dot(label, x);
			pushPq(pq, dot, top_num);
		}
		sort_heap(pq.begin(), pq.end(), compare);
		_multi_label[i] = pq;

	}

	delete[] class_cnt;
	delete[] cent;
}
void Kmeans::part_level_kmeans(vector<int>& pid_list, int max_class_size,
		string prefix, int branch_num, int iter) {
	//int branch_num = 5;
	if(pid_list.size() == 0) return ;

	if (pid_list.size() <= max_class_size) {
		for (int i = 0; i < pid_list.size(); i++) {
			_label[pid_list[i]] = atoi(prefix.c_str());
		}
		return;
	}
	cout << "prefix " << prefix << " size:" << pid_list.size() << endl;

	part_hard_kmeans(pid_list, branch_num, iter);

	vector<vector<int> > pid_branch(branch_num);
	for (int i = 0; i < pid_list.size(); i++) {
		pid_branch[_label[pid_list[i]]].push_back(pid_list[i]);
	}
	for (int i = 0; i < branch_num; i++) {
		if (pid_branch[i].size() == pid_list.size()) {
			cout << "can not be branched!!!" << endl;
			for (int i = 0; i < pid_list.size(); i++) {
				_label[pid_list[i]] = atoi(prefix.c_str());
			}
			return;
		}
		stringstream ss;
		ss << i;
		string new_prefix = prefix + ss.str();
		part_level_kmeans(pid_branch[i], max_class_size, new_prefix, branch_num,
				iter);
	}

}
void Kmeans::norl_center(int& k, float*& cent, int*& class_cnt) {
	//normalized the center
	for (int b = 0; b < k; b++) {
		float closev = 0;
		if(class_cnt[b]==1) continue;
		for (int c = 0; c < _vec_size; c++) {
			cent[_vec_size * b + c] /= class_cnt[b];
			closev += cent[_vec_size * b + c] * cent[_vec_size * b + c];
		}
		closev = sqrt(closev);
		for (int c = 0; c < _vec_size; c++) {
			cent[_vec_size * b + c] /= closev;
		}
	}

}

