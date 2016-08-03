#!/usr/bin/python2.7
# coding=gb18030

#------------------------------------------------------------------------------
import os
import sys
import mainConf

#------------------------------------------------------------------------------
def upload_data(locPath, HDFSPath):
    """ �������ļ��ϴ���HDFS """
    cmd = mainConf.hadoopPath + " fs -test -e " + HDFSPath
    print cmd
    if os.system(cmd) != 0:
        cmd = mainConf.hadoopPath + " fs -put " + locPath + " " + HDFSPath
        print cmd
        os.system(cmd)

def detect_data():
    """ ���ʵ���Դ�Ƿ������HDFS�����������ϴ� """
    cmd = mainConf.hadoopPath + " fs -test -e " + mainConf.binPath
    print cmd
    if os.system(cmd) != 0:
        cmd = mainConf.hadoopPath + " fs -mkdir " + mainConf.binPath
        print cmd
        os.system(cmd)
    upload_data(mainConf.python27PathLoc, mainConf.python27Path)

#------------------------------------------------------------------------------
# python 0_begin.py
if __name__ == "__main__":
    sys.exit(detect_data())

