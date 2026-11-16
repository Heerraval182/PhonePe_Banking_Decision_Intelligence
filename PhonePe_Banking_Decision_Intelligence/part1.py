import numpy as np
from common import load,write_results

def line(x,y):
    xm=x.mean(); ym=y.mean(); slope=np.sum((x-xm)*(y-ym))/np.sum((x-xm)**2); intercept=ym-slope*xm
    fit=intercept+slope*x; return slope,intercept,fit

def main():
    a,_=load(); y=a['monthly_spend_inr_thousands']; x=a['avg_monthly_inflow_inr_thousands']; n=len(y); d={}
    mean=y.mean(); sd=y.std(ddof=1); se=sd/np.sqrt(n)
    d.update(n=n,mean_value=mean,sd_value=sd,se_value=se,ci95_low=mean-1.96*se,ci95_high=mean+1.96*se)
    z=(mean-8.0)/se; d.update(z_stat=z,reject_h0=abs(z)>1.96,n_required=math_ceil((1.96*sd/.5)**2))
    slope,inter,fit=line(x,y); resid=y-fit; tss=np.sum((y-mean)**2); ess=np.sum((fit-mean)**2); rss=np.sum(resid**2)
    d.update(reg_slope=slope,reg_intercept=inter,tss=tss,ess=ess,rss=rss,tss_equals_ess_plus_rss=abs(tss-ess-rss)<1e-6*tss,r2=1-rss/tss,rmse=np.sqrt(rss/n),mae=np.mean(np.abs(resid)))
    xs=(x-x.mean())/x.std(ddof=1); m=c=0.; costs={0:np.mean(y*y)}
    for ep in range(1,501):
        e=y-(m*xs+c); gm=-2*np.mean(xs*e); gc=-2*np.mean(e); m-=.1*gm; c-=.1*gc
        if ep in (1,10,100,500): costs[ep]=np.mean((y-(m*xs+c))**2)
    gs=m/x.std(ddof=1); gi=c-m*x.mean()/x.std(ddof=1)
    d.update(gd_J0=costs[0],gd_J1=costs[1],gd_J10=costs[10],gd_J100=costs[100],gd_J500=costs[500],gd_slope=gs,gd_intercept=gi,gd_slope_gap=abs(gs-slope))
    order=np.argsort(fit,kind='stable'); d.update(resid_sd_lower_raw=np.std(resid[order[:400]],ddof=1),resid_sd_upper_raw=np.std(resid[order[400:]],ddof=1),resid_sd_ratio_raw=np.std(resid[order[400:]],ddof=1)/np.std(resid[order[:400]],ddof=1))
    lx=np.log(x); ly=np.log(y); ls,li,lfit=line(lx,ly); lr=ly-lfit; ltss=np.sum((ly-ly.mean())**2); lrss=np.sum(lr**2)
    lo=np.argsort(lfit,kind='stable'); llo=np.std(lr[lo[:400]],ddof=1); lhi=np.std(lr[lo[400:]],ddof=1)
    d.update(loglog_intercept=li,loglog_slope=ls,loglog_r2=1-lrss/ltss,loglog_rmse=np.sqrt(lrss/n),resid_sd_lower_log=llo,resid_sd_upper_log=lhi,resid_sd_ratio_log=lhi/llo)
    cols=[np.ones(n),np.log(x),a['account_age_months'],a['failed_txn_count_90d'],a['credit_utilisation_ratio'],a['repayment_score'],a['irregular_salary_flag'],a['autopay_flag']]
    mat=np.column_stack(cols); beta=np.linalg.solve(mat.T@mat,mat.T@ly); pred=mat@beta; mrss=np.sum((ly-pred)**2); mr2=1-mrss/ltss
    for i,v in enumerate(beta): d[f'mr_beta_{i}']=v
    d.update(mr_r2=mr2,mr_adj_r2=1-(1-mr2)*(n-1)/(n-8),loglog_adj_r2=1-(1-(1-lrss/ltss))*(n-1)/(n-2))
    write_results('part1_results.txt',d)

def math_ceil(x): return int(np.ceil(x))
if __name__=='__main__': main()
