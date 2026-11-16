import csv, os, math
import numpy as np
ROOT=os.path.dirname(os.path.abspath(__file__))
DATA=os.path.join(ROOT,'data','records.csv'); OUT=os.path.join(ROOT,'outputs')
os.makedirs(OUT,exist_ok=True)

def load():
    with open(DATA,newline='') as f:
        rows=list(csv.DictReader(f))
    names=list(rows[0].keys()); a={k:np.array([float(r[k]) for r in rows]) for k in names if k!='record_id'}
    return a,names

def write_results(name,d):
    with open(os.path.join(OUT,name),'w') as f:
        for k,v in d.items():
            if isinstance(v,(str,bool,np.bool_)): s=('yes' if bool(v) else 'no') if isinstance(v,(bool,np.bool_)) else v
            elif isinstance(v,(int,np.integer)): s=str(int(v))
            else: s=f'{float(v):.4f}'
            f.write(f'{k} = {s}\n')

def write_csv(name,header,rows):
    with open(os.path.join(OUT,name),'w',newline='') as f:
        w=csv.writer(f); w.writerow(header)
        for row in rows:
            w.writerow([f'{x:.4f}' if isinstance(x,(float,np.floating)) else x for x in row])

def split_standard(a):
    y=a['loan_default'].astype(int); zero=np.where(y==0)[0]; one=np.where(y==1)[0]
    tr=np.sort(np.r_[zero[:7*len(zero)//10],one[:7*len(one)//10]])
    te=np.sort(np.r_[zero[7*len(zero)//10:],one[7*len(one)//10:]])
    X=np.column_stack([a[k] for k in ['account_age_months','failed_txn_count_90d','credit_utilisation_ratio','avg_monthly_inflow_inr_thousands','repayment_score','irregular_salary_flag','autopay_flag']])
    mu=X[tr].mean(axis=0); sd=X[tr].std(axis=0,ddof=1); Z=(X-mu)/sd
    return X,Z,y,tr,te,mu,sd

def sigmoid(z): return 1/(1+np.exp(-np.clip(z,-500,500)))
def bce(y,p):
    p=np.clip(p,1e-15,1-1e-15); return float(-np.mean(y*np.log(p)+(1-y)*np.log(1-p)))
def metrics(y,pred):
    tp=int(np.sum((y==1)&(pred==1))); fp=int(np.sum((y==0)&(pred==1))); tn=int(np.sum((y==0)&(pred==0))); fn=int(np.sum((y==1)&(pred==0)))
    div=lambda a,b:a/b if b else 0.0
    pr=div(tp,tp+fp); re=div(tp,tp+fn); return dict(tp=tp,fp=fp,tn=tn,fn=fn,accuracy=div(tp+tn,len(y)),precision=pr,recall=re,f1=div(2*pr*re,pr+re),specificity=div(tn,tn+fp))
def logistic_fit(Z,y,idx,eta=.1,steps=1000,track=False):
    X=Z[idx]; yy=y[idx]; w=np.zeros(X.shape[1]); b=0.; hist={0:bce(yy,sigmoid(X@w+b))}
    for ep in range(1,steps+1):
        p=sigmoid(X@w+b); err=p-yy; w-=eta*(X.T@err)/len(yy); b-=eta*np.mean(err)
        if ep in (1,10,100,200,1000): hist[ep]=bce(yy,sigmoid(X@w+b))
    return w,b,hist

def knn_predict(refX,refy,qX,k,return_prob=False):
    preds=[]; probs=[]
    for q in qX:
        dist=np.sqrt(np.sum((refX-q)**2,axis=1)); ids=np.argsort(dist,kind='stable')[:k]; p=float(np.mean(refy[ids])); probs.append(p); preds.append(int(p>=((k+1)//2)/k))
    return (np.array(preds),np.array(probs)) if return_prob else np.array(preds)
