from pathlib import Path
import pandas as pd, numpy as np, joblib
from sklearn.ensemble import HistGradientBoostingRegressor
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[1]; proc=root/'data'/'processed'; out=root/'outputs'; charts=out/'charts'; models=root/'models'
models.mkdir(exist_ok=True); charts.mkdir(exist_ok=True)

df=pd.read_csv(proc/'analysis_ready.csv',parse_dates=['date'])
df['week']=df['date'].dt.to_period('W').dt.start_time
weekly=df.groupby(['sku_id','week'],as_index=False).agg(units_sold=('units_sold','sum'),revenue=('revenue','sum'),promo_flag=('promo_flag','max'))
weekly=weekly.sort_values(['sku_id','week'])
g=weekly.groupby('sku_id',group_keys=False)
for lag in [1,2,4,8]: weekly[f'lag_{lag}']=g.units_sold.shift(lag)
weekly['rolling_4']=g.units_sold.transform(lambda s:s.shift(1).rolling(4).mean())
weekly['rolling_8']=g.units_sold.transform(lambda s:s.shift(1).rolling(8).mean())
weekly['week_of_year']=weekly.week.dt.isocalendar().week.astype(int)
weekly['month']=weekly.week.dt.month
# seasonal-naive 4-week baseline
weekly['baseline']=g.units_sold.shift(4)
features=['lag_1','lag_2','lag_4','lag_8','rolling_4','rolling_8','week_of_year','month','promo_flag']
valid=weekly.dropna(subset=features+['units_sold','baseline']).copy()
# hold out final 8 weeks globally
cutoff=valid.week.max()-pd.Timedelta(weeks=7)
train=valid[valid.week<cutoff]; test=valid[valid.week>=cutoff]
model=HistGradientBoostingRegressor(max_iter=250,learning_rate=.06,max_leaf_nodes=31,l2_regularization=1.0,random_state=42)
model.fit(train[features],train.units_sold)
test=test.copy(); test['prediction']=np.maximum(model.predict(test[features]),0)
def wape(y,p):
    return float(np.abs(y-p).sum()/np.abs(y).sum())
metrics=pd.DataFrame([{'baseline_wape':wape(test.units_sold,test.baseline),'model_wape':wape(test.units_sold,test.prediction),'baseline_mae':float(np.abs(test.units_sold-test.baseline).mean()),'model_mae':float(np.abs(test.units_sold-test.prediction).mean())}])
metrics['model_beats_baseline']=metrics.model_wape<metrics.baseline_wape
metrics.to_csv(out/'model_metrics.csv',index=False)
joblib.dump(model,models/'forecast_model.joblib')
weekly.to_csv(out/'weekly_training_data.csv',index=False)

# Forecast 8 weeks recursively per SKU; use model only if it wins, otherwise baseline seasonal repeat.
use_model=bool(metrics.loc[0,'model_beats_baseline'])
future=[]
for sku,g0 in weekly.groupby('sku_id'):
    g0=g0.sort_values('week')
    hist=list(g0.units_sold.astype(float).values)
    if len(hist)<8: continue
    last=g0.week.max()
    for h in range(1,9):
        wk=last+pd.Timedelta(weeks=h)
        if use_model:
            row=pd.DataFrame([{
                'lag_1':hist[-1],'lag_2':hist[-2],'lag_4':hist[-4],'lag_8':hist[-8],
                'rolling_4':np.mean(hist[-4:]),'rolling_8':np.mean(hist[-8:]),
                'week_of_year':int(wk.isocalendar().week),'month':wk.month,'promo_flag':0
            }])
            pred=max(float(model.predict(row[features])[0]),0)
        else:
            pred=hist[-4]
        future.append({'sku_id':sku,'forecast_week':wk,'forecast_units':pred})
        hist.append(pred)
future=pd.DataFrame(future)
future.to_csv(out/'weekly_forecast.csv',index=False)

# Risk based on latest inventory snapshot.
inv=pd.read_csv(proc/'inventory_snapshots.csv',parse_dates=['date'])
inv=inv.sort_values('date').groupby('sku_id',as_index=False).tail(1)
risk=future.groupby('sku_id',as_index=False).agg(horizon_demand=('forecast_units','sum'),avg_weekly=('forecast_units','mean'))
risk=risk.merge(inv[['sku_id','on_hand_units','on_order_units','lead_time_days','reorder_point']],on='sku_id',how='left')
risk['lead_time_weeks']=risk.lead_time_days.fillna(7)/7
risk['lead_time_demand']=risk.avg_weekly*risk.lead_time_weeks
risk['stockout_gap']=risk.lead_time_demand-risk.on_hand_units.fillna(0)-risk.on_order_units.fillna(0)
risk['stockout_risk']=risk.stockout_gap>0
risk['overstock_units']=risk.on_hand_units.fillna(0)-risk.horizon_demand
risk['overstock_risk']=risk.overstock_units>0
risk['action']=np.select([risk.stockout_risk & ~risk.overstock_risk,risk.overstock_risk & ~risk.stockout_risk,risk.stockout_risk & risk.overstock_risk],['REORDER NOW','MARKDOWN / CLEAR','WATCH / VOLATILE'],default='HEALTHY')
sku=pd.read_csv(proc/'sku_master.csv')
risk=risk.merge(sku[['sku_id','category','subcategory','unit_price','cost_price','brand']],on='sku_id',how='left')
risk['sales_at_risk_rupees']=risk.stockout_gap.clip(lower=0)*risk.unit_price.fillna(0)
risk['locked_capital_rupees']=risk.overstock_units.clip(lower=0)*risk.cost_price.fillna(0)
risk.to_csv(out/'risk_scoring.csv',index=False)

# EDA tables/charts
summary=pd.DataFrame({'column':df.columns,'dtype':[str(df[c].dtype) for c in df.columns],'missing':[int(df[c].isna().sum()) for c in df.columns],'unique':[int(df[c].nunique()) for c in df.columns]})
summary.to_csv(out/'data_quality_summary.csv',index=False)
top=df.groupby('sku_id',as_index=False).units_sold.sum().sort_values('units_sold',ascending=False).head(20); top.to_csv(out/'top_movers.csv',index=False)
dead=df.groupby('sku_id',as_index=False).units_sold.sum().sort_values('units_sold').head(20); dead.to_csv(out/'dead_stock_candidates.csv',index=False)
cat=df.groupby('category',as_index=False).agg(units_sold=('units_sold','sum'),revenue=('revenue','sum')).sort_values('units_sold',ascending=False); cat.to_csv(out/'category_performance.csv',index=False)
monthly=df.assign(month=df.date.dt.to_period('M').dt.to_timestamp()).groupby('month',as_index=False).units_sold.sum(); monthly.to_csv(out/'monthly_demand.csv',index=False)
# charts
plt.figure(figsize=(11,5)); plt.plot(monthly.month,monthly.units_sold); plt.title('Monthly Total Demand'); plt.xlabel('Month'); plt.ylabel('Units Sold'); plt.tight_layout(); plt.savefig(charts/'monthly_demand.png',dpi=150); plt.close()
plt.figure(figsize=(10,6)); plt.bar(top.sku_id.astype(str),top.units_sold); plt.xticks(rotation=75); plt.title('Top 20 SKUs by Units Sold'); plt.tight_layout(); plt.savefig(charts/'top_20_skus.png',dpi=150); plt.close()
plt.figure(figsize=(10,5)); plt.bar(cat.category.astype(str),cat.units_sold); plt.xticks(rotation=45,ha='right'); plt.title('Demand by Category'); plt.tight_layout(); plt.savefig(charts/'category_demand.png',dpi=150); plt.close()
# model vs baseline summary by week
comp=test.groupby('week',as_index=False).agg(actual=('units_sold','sum'),baseline=('baseline','sum'),model=('prediction','sum')); comp.to_csv(out/'backtest_weekly.csv',index=False)
plt.figure(figsize=(11,5)); plt.plot(comp.week,comp.actual,label='Actual'); plt.plot(comp.week,comp.baseline,label='Seasonal-naive'); plt.plot(comp.week,comp.model,label='Model'); plt.legend(); plt.title('Backtest: Actual vs Baseline vs Model'); plt.tight_layout(); plt.savefig(charts/'backtest.png',dpi=150); plt.close()
print(metrics.to_string(index=False)); print('forecast rows',len(future),'risk rows',len(risk),'model used',use_model); print('risk actions',risk.action.value_counts().to_dict()); print('sales at risk',risk.sales_at_risk_rupees.sum(),'locked',risk.locked_capital_rupees.sum())
