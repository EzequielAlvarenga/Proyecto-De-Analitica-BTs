"""Pipeline original BTS. Ejecutar desde la raíz: python src/pipeline.py."""
from pathlib import Path
import json, hashlib, zipfile, pickle
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.dummy import DummyClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, brier_score_loss, ConfusionMatrixDisplay
ROOT=Path(__file__).resolve().parents[1]
for x in ['data/processed','models','reports/figures']: (ROOT/x).mkdir(parents=True,exist_ok=True)
RAW=ROOT/'data/raw/bts_2024_01.zip'
URL='https://transtats.bts.gov/PREZIP/On_Time_Reporting_Carrier_On_Time_Performance_1987_present_2024_1.zip'
if not RAW.exists():
    from urllib.request import urlretrieve
    urlretrieve(URL,RAW)
cols=['FlightDate','DayofMonth','DayOfWeek','Reporting_Airline','Flight_Number_Reporting_Airline','Origin','Dest','CRSDepTime','CRSElapsedTime','Distance','ArrDelay','DepDelay','Cancelled','Diverted','ArrDel15','CarrierDelay','WeatherDelay','NASDelay','SecurityDelay','LateAircraftDelay']
with zipfile.ZipFile(RAW) as z:
    d=pd.read_csv(z.open(next(n for n in z.namelist() if n.endswith('.csv'))),usecols=cols)
    (ROOT/'data/raw/diccionario_bts.html').write_bytes(z.read('readme.html'))
audit={'raw_rows':len(d),'raw_nulls':d.isna().sum().to_dict(),'source':URL,'sha256':hashlib.sha256(RAW.read_bytes()).hexdigest()}
# Los duplicados exactos se eliminan; la clave de vuelo se audita por separado.
audit['exact_duplicates']=int(d.duplicated().sum());d=d.drop_duplicates().copy()
key=['FlightDate','Reporting_Airline','Flight_Number_Reporting_Airline','Origin','Dest','CRSDepTime']
audit['duplicate_keys']=int(d.duplicated(key).sum())
if audit['duplicate_keys']: raise ValueError('Revisar claves de vuelo repetidas antes de seguir')
d['FlightDate']=pd.to_datetime(d.FlightDate)
audit['cancelled']=int(d.Cancelled.eq(1).sum());audit['diverted']=int(d.Diverted.eq(1).sum())
# No imputar la etiqueta ni confundir cancelaciones con puntualidad.
e=d.loc[d.Cancelled.eq(0)&d.Diverted.eq(0)].copy()
audit['missing_target']=int(e.ArrDelay.isna().sum());e=e.dropna(subset=['ArrDelay'])
# HHMM es hora local; 2400 equivale a medianoche. No restar horas locales entre aeropuertos.
hh=e.CRSDepTime//100;mm=e.CRSDepTime%100
bad=(hh>24)|(hh<0)|(mm>59)|((hh==24)&(mm!=0))|(e.Distance<=0)|(e.CRSElapsedTime<=0)
audit['physical_invalid']=int(bad.sum());e=e.loc[~bad].copy()
e['hour']=(e.CRSDepTime//100)%24;e['distance_km']=e.Distance*1.609344
# Retraso negativo significa llegada anticipada, no error físico.
e['delayed']=(e.ArrDelay>=15).astype(int)
audit['label_mismatches']=int((e.ArrDel15.notna()&(e.ArrDel15!=e.delayed)).sum())
q1,q3=e.ArrDelay.quantile([.25,.75]);lo=q1-1.5*(q3-q1);hi=q3+1.5*(q3-q1)
audit['iqr_bounds']=[lo,hi];audit['iqr_flags_retained']=int(((e.ArrDelay<lo)|(e.ArrDelay>hi)).sum())
audit['eligible_rows']=len(e);audit['delay_rate']=float(e.delayed.mean())
# Las causas ausentes quedan ausentes; no se usan para predecir.
e.to_csv(ROOT/'data/processed/vuelos.csv',index=False)
e[['ArrDelay','DepDelay','distance_km','CRSElapsedTime']].describe(percentiles=[.5,.9,.95,.99]).to_csv(ROOT/'reports/descriptivos.csv')
# Contraste por DÍA para evitar tratar miles de vuelos como réplicas independientes.
m=e.loc[e.hour.between(6,11)].groupby('FlightDate').delayed.mean()
v=e.loc[e.hour.between(18,23)].groupby('FlightDate').delayed.mean()
delta=(v-m).dropna(); test=stats.ttest_1samp(delta,0)
ci=stats.t.interval(.95,len(delta)-1,loc=delta.mean(),scale=stats.sem(delta))
audit['hypothesis']={'days':len(delta),'evening_minus_morning':float(delta.mean()),'t':float(test.statistic),'p':float(test.pvalue),'ci95':list(ci),'daily_lag1_autocorr':float(delta.autocorr())}
# Partición cronológica. Ningún transformador ve validación o prueba al ajustarse.
train=e[e.DayofMonth<=18];val=e[e.DayofMonth.between(19,24)];testset=e[e.DayofMonth>=25]
num=['DepDelay','hour','DayOfWeek','distance_km','CRSElapsedTime'];cat=['Origin','Dest','Reporting_Airline'];features=num+cat
pre=ColumnTransformer([('num',SimpleImputer(strategy='median'),num),('cat',Pipeline([('impute',SimpleImputer(strategy='most_frequent')),('onehot',OneHotEncoder(handle_unknown='ignore'))]),cat)])
def metrics(y,p):
    yhat=(p>=.5).astype(int)
    return { 'roc_auc':float(roc_auc_score(y,p)),'pr_auc':float(average_precision_score(y,p)),'accuracy':float(accuracy_score(y,yhat)),'precision':float(precision_score(y,yhat,zero_division=0)),'recall':float(recall_score(y,yhat,zero_division=0)),'f1':float(f1_score(y,yhat,zero_division=0)),'brier':float(brier_score_loss(y,p)),'confusion_matrix':confusion_matrix(y,yhat).tolist()}
candidates=[]
for depth in [2,3,4]:
    pipe=Pipeline([('pre',pre),('tree',DecisionTreeClassifier(max_depth=depth,min_samples_leaf=1500,random_state=42))])
    pipe.fit(train[features],train.delayed)
    candidates.append((roc_auc_score(val.delayed,pipe.predict_proba(val[features])[:,1]),depth))
best=max(candidates)[1]
# Se reajusta en entrenamiento más validación una vez elegida la profundidad.
fit=pd.concat([train,val]);model=Pipeline([('pre',pre),('tree',DecisionTreeClassifier(max_depth=best,min_samples_leaf=1500,random_state=42))]);model.fit(fit[features],fit.delayed)
p=model.predict_proba(testset[features])[:,1]
audit['split']={'train':len(train),'validation':len(val),'test':len(testset)};audit['candidates']=[{'depth':b,'validation_auc':a} for a,b in candidates];audit['selected_depth']=best
audit['test_metrics']=metrics(testset.delayed,p);audit['baseline']=metrics(testset.delayed,np.repeat(fit.delayed.mean(),len(testset)))
# IC por bootstrap de DÍAS del conjunto de prueba; pocos días, incertidumbre limitada.
rng=np.random.default_rng(42);days=testset.DayofMonth.to_numpy();ys=testset.delayed.to_numpy();aucs=[]
for _ in range(300):
    chosen=rng.choice(np.unique(days),size=7,replace=True);ix=np.concatenate([np.where(days==day)[0] for day in chosen]);aucs.append(roc_auc_score(ys[ix],p[ix]))
audit['test_auc_day_bootstrap_ci95']=np.quantile(aucs,[.025,.975]).tolist()
with open(ROOT/'models/arbol.pkl','wb') as f:pickle.dump(model,f)
names=model['pre'].get_feature_names_out();(ROOT/'models/reglas.txt').write_text(export_text(model['tree'],feature_names=list(names)),encoding='utf8')
# Gráficas con unidades y población explícitas.
plt.rcParams.update({'figure.figsize':(9,3.8),'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
def save(name):plt.tight_layout();plt.savefig(ROOT/f'reports/figures/{name}.png',dpi=170);plt.close()
e.ArrDelay.clip(-60,180).hist(bins=60,color='#167d9a');plt.xlabel('Retraso de llegada (min); extremos agrupados en -60 y 180');plt.ylabel('Vuelos');save('distribucion')
e.groupby('hour').delayed.mean().mul(100).plot(marker='o',color='#167d9a');plt.xlabel('Hora local programada de salida');plt.ylabel('Llegadas con retraso (%)');save('hora')
e.groupby('FlightDate').delayed.mean().mul(100).plot(marker='o',color='#167d9a');plt.xlabel('Fecha de enero de 2024');plt.ylabel('Llegadas con retraso (%)');save('diario')
corr=e[['ArrDelay','DepDelay','hour','distance_km','CRSElapsedTime']].corr(method='spearman');corr.to_csv(ROOT/'reports/correlaciones.csv');plt.imshow(corr,vmin=-1,vmax=1,cmap='coolwarm');plt.xticks(range(5),['Llegada','Salida','Hora','Distancia','Duración'],rotation=20);plt.yticks(range(5),['Llegada','Salida','Hora','Distancia','Duración']);plt.colorbar(label='Correlación de Spearman');save('correlacion')
ConfusionMatrixDisplay.from_predictions(testset.delayed,p>=.5,display_labels=['Puntual','Retrasado'],colorbar=False,values_format='d');plt.title('Prueba temporal del 25 al 31 de enero');plt.xlabel('Predicción');plt.ylabel('Resultado real');save('confusion')
imp=pd.Series(model['tree'].feature_importances_,index=names).sort_values(ascending=False);imp.to_csv(ROOT/'reports/importancias.csv');imp.head(7).sort_values().plot.barh(color='#167d9a');plt.xlabel('Reducción normalizada de impureza');save('importancia')
causes=['CarrierDelay','WeatherDelay','NASDelay','SecurityDelay','LateAircraftDelay'];e.loc[e.delayed.eq(1),causes].sum().to_csv(ROOT/'reports/causas_minutos.csv')
(ROOT/'reports/resultados.json').write_text(json.dumps(audit,indent=2,default=float),encoding='utf8')
print(json.dumps(audit,indent=2,default=float))
