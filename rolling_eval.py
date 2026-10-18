import pandas as pd, numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from pathlib import Path
p=Path('/mnt/data/foresight_actual')
w=pd.read_csv(p/'data/processed/sales_daily.csv',parse_dates=['date'])
w['week']=w.date.dt.to_period('W').dt.start_time
w=w.groupby(['sku_id','week'],as_index=False).agg(units_sold=('units_sold','sum'),promo_flag=('promo_flag','max')).sort_values(['sku_id','week'])
g=w.groupby('sku_id',group_keys=False)
for lag in [1,2,4,8]: w[f'lag_{lag}']=g.units_sold.shift(lag)
w['rolling_4']=g.units_sold.transform(lambda s:s.shift(1).rolling(4).mean()); w['rolling_8']=g.units_sold.transform(lambda s:s.shift(1).rolling(8).mean()); w['week_of_year']=w.week.dt.isocalendar().week.astype(int); w['month']=w.week.dt.month; w['baseline']=g.units_sold.shift(4)
F=['lag_1','lag_2','lag_4','lag_8','rolling_4','rolling_8','week_of_year','month','promo_flag']; v=w.dropna(subset=F+['units_sold','baseline']); maxweek=v.week.max()
folds=[maxweek-pd.Timedelta(weeks=21),maxweek-pd.Timedelta(weeks=14),maxweek-pd.Timedelta(weeks=7)]
rows=[]
def wp(a,p): return abs(a-p).sum()/abs(a).sum()
for cut in folds:
    tr=v[v.week<cut]; te=v[(v.week>=cut)&(v.week<cut+pd.Timedelta(weeks=7))]
    if len(tr)<10000 or te.empty: continue
    m=HistGradientBoostingRegressor(max_iter=100,learning_rate=.08,max_leaf_nodes=31,l2_regularization=1,random_state=42)
    m.fit(tr[F],tr.units_sold); pred=np.maximum(m.predict(te[F]),0)
    rows.append({'fold_start':cut,'n_test':len(te),'baseline_wape':wp(te.units_sold,te.baseline),'model_wape':wp(te.units_sold,pred)})
r=pd.DataFrame(rows); r.to_csv(p/'outputs/rolling_origin_backtest.csv',index=False); print(r.to_string(index=False)); print('means',r[['baseline_wape','model_wape']].mean().to_dict())
