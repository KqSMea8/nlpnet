#coding:gbk
import sys
# word    feature freq
def map(line):
    items=line.split("\t")
    if len(items)<3:
        sys.stderr.write("err:"+line+"\n")
        return 
    print items[1]+"\t"+items[0]+"\t"+items[2]



word_dist={}
pair_cnt={}
past_feature=""
alpha=0.001
#input: feature    word    freq
#calculate prob(word|feature)
#output: target source feature prob
def reduce(line):
    items=line.split("\t")
    global past_feature
    global word_dist
    if items[0]!=past_feature:
        amount=0.0
        for k,v in word_dist.items():
            amount+=v
        for k,v in word_dist.items():
            prob=v/amount
            word_dist[k]=prob
        wlist=[]
        for w in word_dist.keys():
            if word_dist[w]>alpha:
                wlist.append(w)
        #send the fea to itself and the larger one
        for i in range(len(wlist)):
            prob=word_dist[wlist[i]]
            print wlist[i]+"\t"+past_feature+"\t"+str(prob)
            for j in range(i+1,len(wlist)):
                pair=wlist[i]+"\t"+wlist[j]
                if wlist[j]>wlist[i]:
                    pair=wlist[j]+"\t"+wlist[i]
                if pair not in pair_cnt:
                    pair_cnt[pair]=1
                else:
                    pair_cnt[pair]+=1
        past_feature=items[0]
        word_dist.clear()
    word_dist[items[1]]=int(items[2]) 

if __name__ == '__main__':
    while True:
        try:
            line=raw_input().strip()
            if sys.argv[1]=="map":
                map(line)
            else:
                reduce(line)
        except EOFError:
            break
    if sys.argv[1]=="reduce":
        reduce("xx\txx\t1")
        for k,v in pair_cnt.items():
            print k
