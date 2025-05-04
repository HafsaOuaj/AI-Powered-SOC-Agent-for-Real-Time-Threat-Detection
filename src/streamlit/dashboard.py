import streamlit as st
import pandas as pd
import json
from streamlit_autorefresh import st_autorefresh
import pickle



st.set_page_config(page_title="AI SOC Dashboard", page_icon=":guardsman:", layout="wide")
st_autorefresh(interval=1000, limit=None, key="refresh")

st.title("🔒 AI-Powered SOC Agent Dashboard")

with open("resources/data_preprocessing/label_encoders.pkl","rb") as f:
    labels_encoder=pickle.load(f)

def load_alerts():
    alerts = []
    with open("alerts.json", "r") as f:
        for line in f:
            try:
                alerts.append(json.loads(line))
            except:
                continue
    return pd.DataFrame(alerts)
df = load_alerts()
st.subheader("🗂️ Full Alert Log")
col1, col2 = st.columns(2)

# Stats
with col1:
    st.subheader("📈 Detected Attacks")
    attack_cat = [df.iloc[i]['log']['attack_cat'] for i in range(len(df))]
    durs = [df.iloc[-i]['log']['dload'] for i in range(20,1,-1)]
    attack_cats = [labels_encoder["attack_cat"].inverse_transform([int(i)])[0] for i in attack_cat]
    st.bar_chart(pd.Series(attack_cats).value_counts())
    st.line_chart(durs)

with col2:
    st.subheader("🕒 Recent Alerts")
    st.dataframe(df.sort_values(by='timestamp', ascending=False).head(10))

# Full Table
st.subheader("🗂️ Full Alert Log")
st.dataframe(df)