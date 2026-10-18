from pathlib import Path
import pandas as pd, numpy as np, json
root=Path(__file__).resolve().parents[1]
raw=root/'data/raw'; proc=root/'data/processed'; out=root/'outputs'; charts=out/'charts'
for d in [proc,out,charts]: d.mkdir(parents=True,exist_ok=True)

# Read and aggregate 10m transaction file in chunks
parts=[]
for chunk in pd.read_csv(raw/'sales_transactions.csv', chunksize=250_000, parse_dates=['date']):
    chunk['quantity']=pd.to_numeric(chunk['quantity'], errors='coerce').fillna(0).clip(lower=0)
    chunk['unit_price']=pd.to_numeric(chunk['unit_price'], errors='coerce').fillna(0)
    chunk['total_value']=pd.to_numeric(chunk['total_value'], errors='coerce').fillna(chunk['quantity']*chunk['unit_price']).clip(lower=0)
    chunk['discount_pct']=pd.to_numeric(chunk['discount_pct'], errors='coerce').fillna(0)
    chunk['promo_flag']=chunk['promo_id'].fillna('').astype(str).ne('').astype(int)
    g=chunk.groupby(['date','sku_id'],as_index=False).agg(units_sold=('quantity','sum'), revenue=('total_value','sum'), unit_price=('unit_price','mean'), promo_flag=('promo_flag','max'))
    parts.append(g)
print('chunks',len(parts))
sales=pd.concat(parts,ignore_index=True).groupby(['date','sku_id'],as_index=False).agg(units_sold=('units_sold','sum'),revenue=('revenue','sum'),unit_price=('unit_price','mean'),promo_flag=('promo_flag','max'))
sales.to_csv(proc/'sales_daily.csv',index=False)

sku=pd.read_csv(raw/'sku_master.csv')
sku['launch_date']=pd.NaT
sku.to_csv(proc/'sku_master.csv',index=False)

# Calendar generated from sales date range + promotion periods
start=sales.date.min().normalize(); end=sales.date.max().normalize()
cal=pd.DataFrame({'date':pd.date_range(start,end,freq='D')})
cal['week']=cal.date.dt.isocalendar().week.astype(int)
cal['month']=cal.date.dt.month
cal['season']=cal.date.dt.month.map({12:'Winter',1:'Winter',2:'Winter',3:'Spring',4:'Spring',5:'Spring',6:'Summer',7:'Summer',8:'Summer',9:'Autumn',10:'Autumn',11:'Autumn'})
cal['is_holiday']=0
promo=pd.read_csv(raw/'promotions.csv',parse_dates=['start_date','end_date'])
cal['promo_event']=''
for _,r in promo.iterrows():
    m=(cal.date>=r.start_date)&(cal.date<=r.end_date)
    cal.loc[m,'promo_event']=r.promo_name
cal.to_csv(proc/'calendar.csv',index=False)

# Inventory source is a current snapshot by store; aggregate to SKU and attach snapshot date.
inv=pd.read_csv(raw/'inventory_snapshot.csv',parse_dates=['last_restock_date'])
snapshot_date=sales.date.max().normalize()
inv_s=inv.groupby('sku_id',as_index=False).agg(on_hand_units=('stock_on_hand','sum'),reorder_point=('reorder_point','sum'),safety_stock=('safety_stock','sum'))
inv_s['date']=snapshot_date
inv_s['on_order_units']=0
inv_s['lead_time_days']=7
inv_s=inv_s[['date','sku_id','on_hand_units','on_order_units','lead_time_days','reorder_point','safety_stock']]
inv_s.to_csv(proc/'inventory_snapshots.csv',index=False)

# Analysis-ready join
analysis=sales.merge(sku,on='sku_id',how='left').merge(cal,on='date',how='left')
analysis.to_csv(proc/'analysis_ready.csv',index=False)

# EDA summary
summary=pd.DataFrame({'column':analysis.columns,'dtype':[str(analysis[c].dtype) for c in analysis.columns],'missing':[int(analysis[c].isna().sum()) for c in analysis.columns],'unique':[int(analysis[c].nunique()) for c in analysis.columns]})
summary.to_csv(out/'data_quality_summary.csv',index=False)
print('sales_daily rows',len(sales),'skus',sales.sku_id.nunique(),'dates',sales.date.min(),sales.date.max())
print('analysis rows',len(analysis))
print(summary.to_string(index=False))
