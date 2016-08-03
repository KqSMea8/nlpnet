#!/usr/bin/python2.7
# coding=utf-8
# 思路: 将泰语query按照空格切分,统计频次; 高频词取到本地先简单审核一下质量; 
# 然后与本地词典原文进行比较, 不在词典原文中的高频词就可能是有效的待添加词

#------------------------------------------------------------------------------
import sys
import mainConf
import hadoopUtil as hu

#------------------------------------------------------------------------------
# 将泰语query按照空格切分, 并同时输出频次信息 
# hao123子域名  搜索引擎    query   query频次
def mapper1():
    for line in sys.stdin:
        line = line.strip()
        data = line.split("\t")
        if len(data) != 4:
            continue
        data[2] = data[2].strip()
        if len(data[2]) == 0:
            continue
        try:
            count = int(data[3])
        except:
            continue

        if data[0] == "th.hao123.com":
            itemlist = filter(lambda x: len(x) != 0, data[2].split(" "))
            for item in itemlist:
                print "%s\t%d" %(item, count)
        else:
            continue
    return 0

#------------------------------------------------------------------------------
# 将term频次归并; 并去掉低频term
def reducer1():
    MIN_THRS = int(sys.argv[2])
    lastT = None
    count = 0
    for line in sys.stdin:
        line = line.strip()
        data = line.split("\t")
        # term changes
        if lastT != None and lastT != data[0]:
            if count > MIN_THRS:
                print "%s\t%d" %(lastT, count)
            count = 0
        lastT = data[0]
        try:
            count += int(data[1])
        except:
            continue
    # last term
    if count > MIN_THRS:
        print "%s\t%d" %(lastT, count)
    return 0

#------------------------------------------------------------------------------
def setupIO(input):
    if len(input) != 1:
        print >> sys.stderr, "len(input) is wrong"
        return -1
    if input[0].isdigit() == False:
        print >> sys.stderr, "%s is not digit" %(input[0])
        return -1

    hdp.input = []
    hdp.input.append(mainConf.inputLog + "2013")
    hdp.input.append(mainConf.inputLog + "2014")
    hdp.output = mainConf.thaiWorddict

    hdp.mapPara = []
    hdp.mapPara.append(input[0])
    hdp.reducePara = []
    hdp.reducePara.append(input[0])
    return 0

#------------------------------------------------------------------------------
def main():
    hdp.projectName = "nlp_zhanjinbo_1_filter_querylog_" + sys.argv[2]
    hdp.mapper = mapper1
    hdp.reducer = reducer1
    hdp.mainFile = sys.argv[0]
    hdp.mapCapacity = mainConf.mapCapacity
    hdp.reduceCapacity = mainConf.reduceCapacity
    hdp.reduceTask = mainConf.reduceTask
    hdp.fileL = ["*.py"]
    hdp.otherD = []
    hdp.otherD.append("mapred.min.split.size=999999999999")
    hdp.otherD.append("mapred.job.priority=HIGH")
    hdp.keyFields = 1
    hdp.partitioner = "Keybase"
    hdp.partitionFields = 1
    hdp.emailAlarm = mainConf.emailList
    hdp.customFun = setupIO
    hdp.execute(sys.argv)

#------------------------------------------------------------------------------
# python 1_filter_querylog.py -e ${min_thrs} -r
if __name__ == "__main__":
    hdp = hu.job()
    sys.exit(main())

