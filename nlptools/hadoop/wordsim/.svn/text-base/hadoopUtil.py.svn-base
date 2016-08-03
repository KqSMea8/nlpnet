#!/usr/bin/evn python
# coding=utf-8

#------------------------------------------------------------------------------
import os
import re
import sys
import time
import getopt
import subprocess as sp
import mainConf

#------------------------------------------------------------------------------
class job:
    def __init__(self):
        self.hadoop = mainConf.hadoopPath
        self.mapper = None
        self.reducer = None
        self.mapPara = []
        self.reducePara = []
        self.input = []
        self.output = None
        self.mainFile = None
        self.projectName = None
        self.debugMapCmd = None
        self.debugInputFormat = "cat"
        self.mapCapacity = 500
        self.reduceCapacity = 200
        self.reduceTask = 10
        self.debugLength = 100
        self.pripority = "NORMAL"
        self.smsAlarm = None    # alarm by sms
        self.emailAlarm = None  # alarm by email
        self.otherD = []        # other -D options
        self.others = []        # other options e.g. -cacheArchive /.../worddict.tar.gz#worddict
        self.keyFields = 1
        self.partitionFields = 1
        self.fileL = ["*.py"]
        self.partitioner = "Keybase"    # or "Int"
        self.useHCE = False
        self.debugTmpFile = None
        self.forceGet = False   # force getting log from hadoop

        self.customFun = None

    def sanityCheck(self):
        assert len(self.input) > 0, "Input path is empty."
        assert self.output != None, "Output path is empty."
        assert self.mainFile != None, "Main program file is empty."
        assert self.projectName != None, "Project name is empty."

    def setDebugTmpFile(self):
        os.system("mkdir -p /tmp/hadoop_debug_data/")
        self.debugTmpFile = "/tmp/hadoop_debug_data/%s"%(self.projectName.replace(" ", "_"))

    def getExeCommand(self):
        if self.useHCE:
            cmdStr = self.hadoop + " streamoverhce "
        else:
            cmdStr = self.hadoop + " streaming "

        cmdStr += " -D mapred.job.name=" + self.projectName.replace(" ", "_") + \
                " -D mapred.job.priority=" + self.pripority + \
                " -D mapred.job.map.capacity=" + "%d" %self.mapCapacity + \
                " -D mapred.job.reduce.capacity=" + "%d"%self.reduceCapacity + \
                " -D mapred.reduce.tasks=" + "%d" %self.reduceTask + \
                " -D stream.num.map.output.key.fields=" + "%d" %self.keyFields + \
                " -D num.key.fields.for.partition=" + "%d" %self.partitionFields

        if self.otherD != None:
            for opt in self.otherD:
                cmdStr += " -D " + opt

        if self.partitioner.upper() == "KEYBASE":
            cmdStr += " -partitioner org.apache.hadoop.mapred.lib.KeyFieldBasedPartitioner "
        elif self.partitioner.upper()=="INT":
            cmdStr += " -partitioner com.baidu.sos.mapred.lib.IntHashPartitioner "
        else:
            cmdStr += " -partitioner %s " %(self.partitioner)

        for s in self.input:
            cmdStr += " -input " + s

        cmdStr += " -output " + self.output
        cmdStr += " -cacheArchive " + mainConf.python27Path + "#python "

        if self.others:
            if isinstance(self.others, list):
                for opt in self.others:
                    cmdStr += " " + opt
            else: # string
                cmdStr += " " + self.others

        for f in self.fileL:
            cmdStr += " -file " + f

        if callable(self.mapper): # is a function
            para = ""
            for n in self.mapPara:
                para = para + " -e " + n
            cmdStr += " -mapper \"python/python2.7/bin/python2.7 " + self.mainFile + para + " -p map\""
        else:
            cmdStr += " -mapper \"" + self.mapper + "\""

        if callable(self.reducer): # is a function
            para = ""
            for n in self.reducePara:
                para = para + " -e " + n
            cmdStr += " -reducer \"python/python2.7/bin/python2.7 " + self.mainFile + para + " -p reduce\""
        else:
            cmdStr += " -reducer \"" + self.reducer + "\" "

        return cmdStr

    def getRmCommand(self):
        cmdStr = self.hadoop + " fs -rmr " + self.output
        return cmdStr

    def fetchDebugData(self):
        if (not self.forceGet) and os.path.exists(self.debugTmpFile): # using last data
            pass
        else:
            if self.debugInputFormat == "zip":
                unzipCmd = "zcat"
            elif self.debugInputFormat == "lzma":
                unzipCmd = "lzcat"
            else:
                unzipCmd = "cat"
            os.system("rm -rf " + self.debugTmpFile)
            # preparing data source
            for p in self.input:
                cmdStr = self.hadoop + " fs -cat " + p + "/* 2>/dev/null | " + unzipCmd + " 2>/dev/null "\
                        " | head -n " + "%d"%self.debugLength + " >>" + self.debugTmpFile + " 2>/dev/null "
                os.system(cmdStr)

    def debugMap(self, isRun=True):
        self.fetchDebugData()
        if callable(self.mapper): # is a function
                self.debugMapCmd = "cat "+ self.debugTmpFile + " | python " + self.mainFile + " -p map "
        else:
                self.debugMapCmd = "cat "+ self.debugTmpFile + " | " + self.mapper
        if isRun:
            os.system(self.debugMapCmd)

    def debugReduce(self):
        self.debugMap(False)
        if callable(self.reducer): # is a function
            cmd = self.debugMapCmd + " | sort | python " + self.mainFile + " -p reduce"
        else:
            cmd = self.debugMapCmd + " | sort | " + self.reducer
        os.system(cmd)

    def show(self):
        cmd = self.hadoop + " fs -cat " + self.output + "/*"
        os.system(cmd)

    def showInput(self):
        self.fetchDebugData()
        cmd = "cat "+ self.debugTmpFile
        os.system(cmd)

    def ls(self):
        cmd = self.hadoop + " fs -ls " + self.output
        print cmd
        sys.stdout.flush()
        os.system(cmd)

    def __printbetter(self,cmdStr):
        cmdStr = cmdStr.replace(" -", "\n\t -")
        cmdStr = cmdStr.replace("\n\t -p ", " -p ")
        print cmdStr

    def alarm(self):
        if self.smsAlarm != None:
            alarmStr = "hadoop task failed. Task name:" + self.projectName
            smsCmd = "gsmsend -s emp01.baidu.com:15003 " + self.smsAlarm + "\@\"" + alarmStr + ".\""
            print smsCmd
            os.system(smsCmd)
        if self.emailAlarm != None:
            alarmStr = "\"hadoop task failed. Task name:" + self.projectName + "\""
            emailCmd = "echo " + alarmStr + " | mail -s \"[warning] error\" " + self.emailAlarm
            print emailCmd
            os.system(emailCmd)

    def checkexit(self,inputPath):
        cmd = self.hadoop + " fs -test -e " + inputPath
        return os.system(cmd)

    def run(self):
        self.sanityCheck()
        rmCmd = self.getRmCommand()
        print rmCmd
        sys.stdout.flush()
        os.system(rmCmd)
        exeCmd = self.getExeCommand()
        print exeCmd
        self.__printbetter(exeCmd)
        sys.stdout.flush()
        rescode=os.system(exeCmd)
        sys.stdout.flush()
        time.sleep(5)
        # try to count the output folder. If task failed, will return 255
        lsCmd = self.hadoop + " fs -count " + self.output
        p = sp.Popen(lsCmd, shell = True, stdout=sp.PIPE, stderr=None)
        output = p.communicate()[0]
        if p.returncode == 0 and rescode == 0:
            data = re.match(r'^\s+(\d+)\s+(\d+)\s+(\d+).*$', output)
            byte = int(data.group(3))
            num = int(data.group(2))
            if byte > 0 and num == self.reduceTask:
                print "\nTask sucesses. Output information:"
                print "\tNumber of file:\t "+str(data.group(2))
                if byte > 1024*1024*1024:
                    print "\tNumber of byte:\t %.2f GB" %(byte/(1024*1024*1024.0))
                elif byte > 1024*1024:
                    print "\tNumber of byte:\t %.2f MB" %(byte/(1024*1024.0))
                elif byte > 1024:
                    print "\tNumber of byte:\t %.2f KB" %(byte/1024.0)
                else:
                    print "\tNumber of byte:\t %d" %(byte)
            else:
                if byte <= 0:
                    print "\nTask failed with output empty."
                if num != self.reduceTask:
                    print "\nTask failed with incomplete output."
                self.alarm()
                return -1
        else:
            print "Task failed !!!"
            self.alarm()
        print
        sys.stdout.flush()
        if p.returncode == 0 and rescode == 0:
            return 0
        else:
            return 255
       
    def usage(self):
        print "Usage: python_program [-hsdr] "
        print "    -h --help        give this help"
        print "    -s --show        cat hadoop output"
        print "    -i --input       cat hadoop input"
        print "    -l --ls          ls hadoop output"
        print "    -d --debug       have to be followed by \"map\" or \"reduce\" "
        print "    -f --force       force getting debug data. It has to be used with -d"
        print "    -r --run         run hadoop program"
        print "    -p --hadoop      run hadoop program (internal use only)"
        print "    -e --extra       parameters are input of customized function"

    def execute(self, argv):
        self.setDebugTmpFile()
        strCmd = ""
        strParam = ""
        extParam = []
        try:
            opts, args = getopt.getopt(argv[1:], "hd:fsilrp:e:", \
                ["help", "debug=", "force", "show", "input", "ls", "run", "hadoop=", "extra="])
        except getopt.error, msg:
            print msg
            print "for help use --help"
            sys.exit(2)
        # process options
        if len(opts) == 0:
            self.usage()
            sys.exit(0)
        for o, a in opts:
            if o in ("-p", "--hadoop"):
                if a == "map":
                    self.mapper()
                elif a == "reduce":
                    self.reducer()
                sys.exit(0)
            if o in ("-d", "--debug"):
                strCmd += "d"
                strParam = a
            elif o in ("-f", "--force"):
                strCmd += "f"
            elif o in ("-s", "--show"):
                strCmd += "s"
            elif o in ("-i", "--input"):
                strCmd += "i"
            elif o in ("-l", "--ls"):
                strCmd += "l"
            elif o in ("-r", "--run"):
                strCmd += "r"
            elif o in ("-h", "--help"):
                strCmd += "h"
            elif o in ("-e", "--extra"):
                extParam.append(a)

        # execute a customized funcation
        if callable(self.customFun):
            self.customFun(extParam)
        if "h" in strCmd:
            self.usage()
            sys.exit(0)
        elif "s" in strCmd:
            self.show()
        elif "i" in strCmd:
            self.showInput()
        elif "l" in strCmd:
            self.ls()
        elif "d" in strCmd:
            if "f" in strCmd:
                self.forceGet = True
            if strParam == "map":
                self.debugMap()
            elif strParam == "reduce":
                self.debugReduce()
            else:
                self.usage()
            self.forceGet = False
        elif "r" in strCmd:
            rslt = self.run()
            sys.exit(rslt)

