/*
 * IBM1.h
 *
 *  Created on: 2014-3-20
 *      Author: zengzengfeng
 */

#ifndef IBM1_H_
#define IBM1_H_
using namespace std;

class IBM1 {
public:
	IBM1();
	~IBM1();

	void disk_run(const char* infile,const char* outfile,int iter_num);
	void memory_run(const char* infile,const char* outfile,int iter_num);
	void pipe_run(istream& in);
	void load_align_prob(const char* filename);
	void save_align_prob(const char* filename);
private:
	//对齐一个句对实例
	void align_one(vector<string>& tokens1, vector<string>& tokens2);
	//判断两个词是否可以规则对齐
	bool rule_align(const string& str1, const string& str2);
	void stat_align(vector<string>& tokens1, vector<string>& tokens2, int index,
			float* prob_array);
	void cal_align_prob();

private:
	//A|B prob
	map<string,float> align_prob;
	//A sum(count(A|*))
	map<string, float> align_sum;
	//A|B cnt
	map<string, float> align_cnt;
	map<string,int> align_feature;

};

#endif /* IBM1_H_ */
