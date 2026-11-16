import numpy as np
from common import *

def main():
    a,_=load(); X,Z,y,tr,te,mu,sd=split_standard(a); d={'train_rows':len(tr),'train_positives':int(y[tr].sum()),'test_rows':len(te),'test_positives':int(y[te].sum())}
    w,b,h=logistic_fit(Z,y,tr,.1,1000); d.update(loss_epoch_0=h[0],loss_epoch_1=h[1],loss_epoch_10=h[10],loss_epoch_100=h[100],loss_epoch_1000=h[1000],bias=b,test_bce=bce(y[te],sigmoid(Z[te]@w+b)))
    for i,v in enumerate(w,1): d[f'weight_X{i}']=v; d[f'odds_ratio_X{i}']=np.exp(v)
    for eta in (.01,.1,1.):
        _,_,hh=logistic_fit(Z,y,tr,eta,200); d[f'loss_epoch_200_eta_{eta}']=hh[200]
    prob=sigmoid(Z[te]@w+b)
    for tau in (.5,.3):
        m=metrics(y[te],(prob>=tau).astype(int))
        for k,v in m.items(): d[f'tau_{tau}_{k}']=v
    d['blind_accuracy']=float(np.mean(y[te]==0))
    rows=[]
    for tau in np.arange(.1,1.,.1):
        m=metrics(y[te],(prob>=tau).astype(int)); rows.append([round(float(tau),1),m['tp'],m['fp'],m['tn'],m['fn'],m['recall'],m['fp']/(m['fp']+m['tn']) if m['fp']+m['tn'] else 0])
    write_csv('part2_thresholds.csv',['tau','tp','fp','tn','fn','tpr','fpr'],rows)
    cv=[]
    for k in range(1,22,2):
        scores=[]
        for fold in range(5):
            val=np.arange(len(tr))%5==fold; ref=~val
            pred=knn_predict(Z[tr][ref],y[tr][ref],Z[tr][val],k); scores.append(metrics(y[tr][val],pred)['f1'])
        cv.append([k,float(np.mean(scores))])
    write_csv('part2_knn_cv.csv',['K','mean_cv_f1'],cv); best=min(cv,key=lambda r:(-r[1],r[0]))[0]; d['knn_best_k']=best
    kp=knn_predict(Z[tr],y[tr],Z[te],best); km=metrics(y[te],kp)
    for k,v in km.items(): d[f'knn_{k}']=v
    ptr=sigmoid(Z[tr]@w+b); choices=[]
    for tau in np.arange(.05,1.,.05):
        mm=metrics(y[tr],(ptr>=tau).astype(int)); choices.append((50*mm['fp']+400*mm['fn'],tau))
    cost,tau=min(choices,key=lambda q:(q[0],q[1])); d.update(cost_tau=tau,cost_train_at_tau=cost)
    mt=metrics(y[te],(prob>=tau).astype(int)); m5=metrics(y[te],(prob>=.5).astype(int))
    d.update({'cost_test_at_tau':50*mt['fp']+400*mt['fn'],'cost_test_at_0.5':50*m5['fp']+400*m5['fn'],'cost_test_flag_nobody':400*int(y[te].sum())})
    write_results('part2_results.txt',d)
if __name__=='__main__': main()
