"""Abrir con streamlit run dashboard.py desde la carpeta del proyecto."""
from pathlib import Path
import json
import pandas as pd
import streamlit as st
ROOT=Path(__file__).resolve().parent
st.set_page_config(page_title='Puntualidad BTS',layout='wide')
st.title('Retrasos de llegada en vuelos de Estados Unidos')
st.caption('Enero de 2024 · BTS · Operaciones completadas sin desvío · Retraso ≥ 15 min')
@st.cache_data
def load():return pd.read_csv(ROOT/'data/processed/vuelos.csv',parse_dates=['FlightDate'])
d=load();airports=st.sidebar.multiselect('Origen',sorted(d.Origin.unique()));airlines=st.sidebar.multiselect('Aerolínea',sorted(d.Reporting_Airline.unique()));hours=st.sidebar.slider('Hora local programada',0,23,(0,23))
f=d[d.hour.between(*hours)]
if airports:f=f[f.Origin.isin(airports)]
if airlines:f=f[f.Reporting_Airline.isin(airlines)]
if f.empty:st.warning('No hay vuelos con esos filtros.');st.stop()
a,b,c=st.columns(3);a.metric('Vuelos',f'{len(f):,}');b.metric('Retrasados',f'{f.delayed.mean():.1%}');c.metric('Mediana de retraso',f'{f.ArrDelay.median():.0f} min')
st.subheader('Variación diaria');st.line_chart(f.groupby('FlightDate').delayed.mean().mul(100),y_label='Retraso (%)')
st.subheader('Hora programada de salida');st.bar_chart(f.groupby('hour').delayed.mean().mul(100),y_label='Retraso (%)')
st.subheader('Aeropuertos con al menos 100 vuelos');g=f.groupby('Origin').delayed.agg(['mean','size']);st.dataframe(g[g['size']>=100].sort_values('mean',ascending=False).rename(columns={'mean':'Proporción retrasada','size':'Vuelos'}))
st.subheader('Validación del modelo en datos futuros');r=json.loads((ROOT/'reports/resultados.json').read_text());st.json(r['test_metrics']);st.caption('Estas métricas corresponden a toda la prueba temporal y no cambian con los filtros. La predicción se emite al salir de la puerta y usa la demora de salida ya observada. No autoriza decisiones de seguridad.')
st.download_button('Descargar selección CSV',f.to_csv(index=False),'seleccion.csv','text/csv')
