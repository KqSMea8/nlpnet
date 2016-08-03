/*
 * Kmeans.h
 *
 *  Created on: 2013-12-9
 *      Author: zengzengfeng
 */

#ifndef KMEANS_H_
#define KMEANS_H_
#include<vector>
#include<string>
using namespace std;
class Kmeans {
public:
	Kmeans();
	~Kmeans();
	void cluster( int k,int iter_num);
	void level_cluster(int max_class_size, int branch_num,int iter);
	void cal_topk(int top_num);
	void load_data(const char* data_file);
	void save(const char* outFile);
private:
	bool pushPq(vector<pair<int, float> >& lattice, pair<int, float> dot,
			unsigned maxNum);
	void norl_center(int& k, float*& cent, int*& class_cnt);
	void part_hard_kmeans(vector<int>& pid_list, int k,int iter_num);
	void part_level_kmeans(vector<int>& pid_list, int max_class_size,string prefix,int branch_num,
				int iter);

private:
	int _data_num;
	int _vec_size;
	//int _max_cluster_size;
	vector<float> _data;
	vector<string> _tokens;
	vector<int> _label;
	vector<pair<int, float> >* _multi_label;

};

#endif /* KMEANS_H_ */
