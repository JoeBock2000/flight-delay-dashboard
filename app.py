import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
import joblib
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Flight Delay: Factor Discovery & Prediction", layout="wide", initial_sidebar_state="expanded")

st.markdown('''<style>
.main .block-container {padding-top: 2rem; padding-bottom: 2rem;}
[data-testid="stSidebar"] {background-color: #0F1E3D;}
[data-testid="stSidebar"] * {color: #E8EEF7;}
h1, h2, h3 {color: #0F1E3D;}
.metric-card {background:#F4F7FB;border-radius:12px;padding:1rem 1.25rem;border:1px solid #E1E8F0;}
.insight-box {background:#E6F1FB;border-radius:12px;padding:1rem 1.25rem;border-left:4px solid #378ADD;}
.warn-box {background:#FAEEDA;border-radius:12px;padding:1rem 1.25rem;border-left:4px solid #BA7517;}
.conclusion-box {background:#0F1E3D;border-radius:12px;padding:1.25rem 1.5rem;color:#FFFFFF;}
.conclusion-box p {color:#FFFFFF;}
</style>''', unsafe_allow_html=True)

DATA_PATH = "flight_delay_clean.parquet"
MODEL_PATH = "xgboost_model.json"
ENCODERS_PATH = "encoders.pkl"
# Only the columns the dashboard actually uses. Loading the full 70-column parquet
# and taking the OPERATED copy peaks at ~2.7 GB and gets OOM-killed on Streamlit Cloud.
NEEDED_COLUMNS = ["OUTCOME","DEP_DELAY","DEP_HOUR","ORIGIN","AIRLINE_CODE","DAILY_TRAFFIC","CLUSTER","TIME_OF_DAY","FOG","SEASON","IS_WEEKEND","MONTH","DAY_OF_WEEK","WSF2","THUNDER","HAZE_SMOKE","PRCP","TMAX","TMIN","DISTANCE","AWND","SNOW","SNWD"]

@st.cache_data(show_spinner="Loading flight data...")
def load_data():
    df = pd.read_parquet(DATA_PATH, columns=NEEDED_COLUMNS)
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype("category")
    return df

@st.cache_resource
def load_model():
    m = xgb.XGBClassifier()
    m.load_model(MODEL_PATH)
    return m

@st.cache_resource
def load_encoder_bundle():
    return joblib.load(ENCODERS_PATH)

def build_encoders():
    return load_encoder_bundle()['encoders']

@st.cache_data
def airport_reference(df):
    return df.groupby("ORIGIN").agg(DAILY_TRAFFIC=("DAILY_TRAFFIC","mean"),WSF2=("WSF2","mean"),AWND=("AWND","mean"),TMAX=("TMAX","mean"),TMIN=("TMIN","mean"),PRCP=("PRCP","mean"),SNOW=("SNOW","mean"),SNWD=("SNWD","mean"),FOG=("FOG","mean"),HAZE_SMOKE=("HAZE_SMOKE","mean"),DISTANCE=("DISTANCE","mean"))

df = load_data()
model = load_model()
encoders = build_encoders()
MODEL_FEATURES = load_encoder_bundle()['feature_order']
airport_ref = airport_reference(df)
OPERATED = df[df["OUTCOME"].isin(["On-time","Delayed"])]
BASELINE_DELAY = (OPERATED["OUTCOME"]=="Delayed").mean()*100

st.sidebar.markdown("### Flight Delay")
st.sidebar.caption("Factor Discovery & Prediction")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate",["Overview","Prediction","Factors","Patterns","Mechanisms","Synthesis"],label_visibility="collapsed")
st.sidebar.markdown("---")
st.sidebar.caption(f"Dataset: {len(df):,} flights\n\n10 US hub airports\n\nBTS + NOAA, 2019-2023")

if page == "Overview":
    st.title("Flight Delay: Factor Discovery & Prediction Dashboard")
    st.caption("A data mining approach to understanding flight disruption | BTS + NOAA, 2019-2023")
    c1,c2,c3,c4 = st.columns(4)
    total=len(df); delayed=(df["OUTCOME"]=="Delayed").sum(); cancelled=(df["OUTCOME"]=="Cancelled").sum()
    c1.markdown(f'<div class="metric-card"><small>Flights analysed</small><h2>{total:,}</h2></div>',unsafe_allow_html=True)
    c2.markdown(f'<div class="metric-card"><small>Delayed</small><h2>{delayed/total*100:.1f}%</h2></div>',unsafe_allow_html=True)
    c3.markdown(f'<div class="metric-card"><small>Cancelled</small><h2>{cancelled/total*100:.1f}%</h2></div>',unsafe_allow_html=True)
    c4.markdown(f'<div class="metric-card"><small>Best model ROC-AUC</small><h2>0.71</h2></div>',unsafe_allow_html=True)
    st.subheader("The central finding")
    st.markdown('<div class="conclusion-box"><p style="margin:0;">Flight disruption is <b>multifactorial</b>. No single factor dominates. Prediction is inherently limited (best model reaches only 0.71 ROC-AUC even after tuning), which is precisely why understanding the contributing <b>factors and circumstances</b> is the more valuable pursuit.</p></div>',unsafe_allow_html=True)
    st.subheader("Outcome distribution")
    fig,ax=plt.subplots(figsize=(9,3.2))
    counts=df["OUTCOME"].value_counts()
    colors={"On-time":"#1D9E75","Delayed":"#D85A30","Cancelled":"#BA7517","Diverted":"#534AB7"}
    ax.barh(counts.index[::-1],counts.values[::-1],color=[colors.get(x,"#888") for x in counts.index[::-1]])
    for i,v in enumerate(counts.values[::-1]): ax.text(v,i,f" {v:,} ({v/total*100:.1f}%)",va="center",fontsize=9)
    ax.set_xlabel("Number of flights"); ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout(); st.pyplot(fig)

elif page == "Prediction":
    st.title("Delay Prediction")
    st.caption("Estimate delay risk from flight parameters. Discovery, not this tab, is where the value lies.")
    left,right = st.columns([1,1])
    with left:
        st.subheader("Flight parameters")
        origin=st.selectbox("Origin airport",sorted(df["ORIGIN"].unique()))
        dep_hour=st.slider("Departure hour",5,23,19)
        month=st.slider("Month",1,12,7)
        airline=st.selectbox("Airline",sorted(df["AIRLINE_CODE"].unique()))
        distance=st.slider("Flight distance (miles)",100,3000,800,step=50)
        st.markdown("**Weather conditions**")
        wc1,wc2=st.columns(2)
        with wc1: thunder=st.checkbox("Thunderstorm"); fog=st.checkbox("Fog")
        with wc2: precip=st.slider("Precipitation (in)",0.0,3.0,0.0,step=0.1); high_traffic=st.checkbox("Peak traffic day")
    def season_from_month(m):
        if m in (12,1,2): return "Winter"
        if m in (3,4,5): return "Spring"
        if m in (6,7,8): return "Summer"
        return "Autumn"
    def tod_from_hour(h):
        if 5<=h<12: return "Morning"
        if 12<=h<17: return "Afternoon"
        if 17<=h<21: return "Evening"
        return "Night"
    ref=airport_ref.loc[origin]; season=season_from_month(month); tod=tod_from_hour(dep_hour)
    traffic=ref["DAILY_TRAFFIC"]*(1.15 if high_traffic else 1.0)
    row={"TIME_OF_DAY":encoders["TIME_OF_DAY"].transform([tod])[0],"FOG":1 if fog else 0,"SEASON":encoders["SEASON"].transform([season])[0],"IS_WEEKEND":0,"AIRLINE_CODE":encoders["AIRLINE_CODE"].transform([str(airline)])[0],"MONTH":month,"DEP_HOUR":dep_hour,"DAY_OF_WEEK":5,"ORIGIN":encoders["ORIGIN"].transform([origin])[0],"WSF2":ref["WSF2"],"THUNDER":1 if thunder else 0,"HAZE_SMOKE":1 if ref["HAZE_SMOKE"]>0.5 else 0,"PRCP":precip,"TMAX":ref["TMAX"],"TMIN":ref["TMIN"],"DISTANCE":distance,"AWND":ref["AWND"],"DAILY_TRAFFIC":traffic,"SNOW":ref["SNOW"],"SNWD":ref["SNWD"]}
    X_row=pd.DataFrame([row])[MODEL_FEATURES]
    proba=float(model.predict_proba(X_row)[0,1])*100
    with right:
        st.subheader("Estimated delay risk")
        st.markdown(f'<div class="conclusion-box"><small style="color:#9DB4D4;">Delay probability</small><h1 style="color:#FFFFFF;margin:0.2rem 0;">{proba:.0f}%</h1></div>',unsafe_allow_html=True)
        st.progress(min(proba/100,1.0))
        st.markdown(f'<div class="warn-box">Baseline delay rate is {BASELINE_DELAY:.0f}%. This model reaches only ROC-AUC 0.71, so treat this as an indicative signal, not a definitive forecast. The value of this project is in understanding the factors.</div>',unsafe_allow_html=True)
        st.caption("Features derived automatically from your inputs and airport averages:")
        st.dataframe(pd.DataFrame({"Derived feature":["Season","Time of day","Airport traffic (avg)","Typical wind","Typical temp max"],"Value":[season,tod,f"{traffic:.0f}/day",f"{ref['WSF2']:.1f}",f"{ref['TMAX']:.0f}F"]}),hide_index=True,use_container_width=True)

elif page == "Factors":
    st.title("What drives each disruption type")
    st.caption("SHAP-derived importance shows delays and cancellations have different signatures")
    st.markdown('<div class="insight-box">Key insight: delays are an <b>intraday, congestion-driven</b> phenomenon (departure hour dominates), while cancellations are a <b>monthly, weather-driven</b> one (month and precipitation dominate).</div>',unsafe_allow_html=True)
    shap_data=pd.DataFrame({"Feature":["MONTH","DEP_HOUR","DISTANCE","SEASON","DAILY_TRAFFIC","AIRLINE_CODE","PRCP","AWND","TMIN","TMAX","WSF2","TIME_OF_DAY"],"Delayed":[0.0278,0.0423,0.0108,0.0085,0.0135,0.0096,0.0057,0.0076,0.0070,0.0065,0.0065,0.0093],"Cancelled":[0.0638,0.0093,0.0111,0.0188,0.0128,0.0108,0.0177,0.0092,0.0080,0.0084,0.0070,0.0041]})
    outcome=st.radio("Show factors driving:",["Delayed","Cancelled","Compare both"],horizontal=True)
    fig,ax=plt.subplots(figsize=(10,5))
    d=shap_data.sort_values("Delayed" if outcome!="Cancelled" else "Cancelled",ascending=True)
    y=np.arange(len(d))
    if outcome=="Delayed":
        ax.barh(y,d["Delayed"],color="#378ADD"); ax.set_xlabel("Mean |SHAP value| - impact on Delayed")
    elif outcome=="Cancelled":
        ax.barh(y,d["Cancelled"],color="#D85A30"); ax.set_xlabel("Mean |SHAP value| - impact on Cancelled")
    else:
        ax.barh(y-0.2,d["Delayed"],height=0.4,color="#378ADD",label="Delayed"); ax.barh(y+0.2,d["Cancelled"],height=0.4,color="#D85A30",label="Cancelled"); ax.set_xlabel("Mean |SHAP value|"); ax.legend()
    ax.set_yticks(y); ax.set_yticklabels(d["Feature"]); ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout(); st.pyplot(fig)

elif page == "Patterns":
    st.title("Disruption patterns and profiles")
    st.caption("Association rules and the flight clusters that emerge from the data")
    st.subheader("Association rules")
    rules=pd.DataFrame({"Conditions":["Summer + Evening","Heavy precipitation","Summer + Very high traffic","Airline WN","Evening","Very high gusts","Hot temperature","Long distance"],"Outcome":["Delayed"]*8,"Support":[0.044,0.021,0.031,0.028,0.056,0.048,0.052,0.051],"Confidence":[0.317,0.271,0.257,0.249,0.245,0.227,0.215,0.206],"Lift":[1.72,1.46,1.39,1.35,1.33,1.23,1.17,1.11]})
    min_lift=st.slider("Minimum lift",1.0,1.8,1.0,step=0.05)
    filtered=rules[rules["Lift"]>=min_lift].sort_values("Lift",ascending=False)
    st.dataframe(filtered,hide_index=True,use_container_width=True)
    fig,ax=plt.subplots(figsize=(10,4))
    dd=filtered.sort_values("Lift")
    ax.barh(dd["Conditions"],dd["Lift"],color="#378ADD"); ax.axvline(1.0,color="#D85A30",linestyle="--",label="Baseline"); ax.set_xlabel("Lift"); ax.legend(); ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout(); st.pyplot(fig)
    st.markdown("---"); st.subheader("Flight profiles (k-means clustering)")
    if "CLUSTER" in df.columns:
        cs=df.groupby("CLUSTER").agg(size=("OUTCOME","size"),delay_rate=("OUTCOME",lambda x:(x=="Delayed").mean()*100),tmax=("TMAX","mean")).round(1)
        names={}
        for cid in cs.index:
            t=cs.loc[cid,"tmax"]; names[cid]="Winter weather" if t<45 else ("Warm / summer" if t>70 else "Transitional")
        cols=st.columns(len(cs))
        for i,cid in enumerate(cs.index):
            cols[i].markdown(f'<div class="metric-card"><small>{names[cid]}</small><h2>{cs.loc[cid,"delay_rate"]:.1f}%</h2><small>delay rate · {int(cs.loc[cid,"size"]):,} flights</small></div>',unsafe_allow_html=True)

elif page == "Mechanisms":
    st.title("How disruption works")
    st.caption("The mechanisms behind delays: propagation, severity, and airport efficiency")
    st.subheader("Delay propagation through the operating day")
    hourly=OPERATED.groupby("DEP_HOUR").agg(delay_rate=("OUTCOME",lambda x:(x=="Delayed").mean()*100),avg_delay=("DEP_DELAY","mean"),flights=("OUTCOME","size")).reset_index()
    hourly=hourly[hourly["flights"]>1000]
    fig,ax1=plt.subplots(figsize=(11,4))
    ax1.plot(hourly["DEP_HOUR"],hourly["delay_rate"],marker="o",color="#D85A30")
    ax1.set_xlabel("Scheduled departure hour"); ax1.set_ylabel("Delay rate (%)",color="#D85A30"); ax1.tick_params(axis="y",labelcolor="#D85A30")
    ax2=ax1.twinx(); ax2.plot(hourly["DEP_HOUR"],hourly["avg_delay"],marker="s",color="#378ADD")
    ax2.set_ylabel("Average delay (minutes)",color="#378ADD"); ax2.tick_params(axis="y",labelcolor="#378ADD")
    ax1.spines[["top"]].set_visible(False); ax2.spines[["top"]].set_visible(False)
    plt.tight_layout(); st.pyplot(fig)
    st.markdown('<div class="insight-box">Both delay rate and average magnitude climb in lockstep through the day, from ~7.7% at 6am to 26.8% by 8pm. The signature of delay propagation.</div>',unsafe_allow_html=True)
    st.markdown("---"); st.subheader("Airport efficiency, not traffic volume, drives delays")
    ast=OPERATED.groupby("ORIGIN").agg(delay_rate=("OUTCOME",lambda x:(x=="Delayed").mean()*100),avg_traffic=("DAILY_TRAFFIC","mean"),flights=("OUTCOME","size")).reset_index()
    fig,ax=plt.subplots(figsize=(10,6))
    ax.scatter(ast["avg_traffic"],ast["delay_rate"],s=ast["flights"]/200,alpha=0.6,color="#378ADD",edgecolor="black")
    for _,r in ast.iterrows(): ax.annotate(r["ORIGIN"],(r["avg_traffic"],r["delay_rate"]),fontsize=10,fontweight="bold",ha="center",va="center")
    ax.set_xlabel("Average daily traffic (flights/day)"); ax.set_ylabel("Delay rate (%)"); ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout(); st.pyplot(fig)
    st.markdown('<div class="insight-box">Atlanta handles the highest traffic yet has among the lowest delay rates. Operational efficiency, not volume, is decisive.</div>',unsafe_allow_html=True)

elif page == "Synthesis":
    st.title("Bringing the methods together")
    st.caption("Where the three methods agree, and what they conclude")
    st.subheader("Factor agreement across methods")
    tri=pd.DataFrame({"Factor":["Time of day","Season / Month","Weather","Airport traffic","Temperature","Distance"],"Association":[1.00,0.93,0.85,0.00,0.88,0.84],"SHAP":[0.65,1.00,0.55,0.13,0.31,1.00],"Clustering":[0.02,0.40,0.50,0.06,1.00,0.05]})
    fig,ax=plt.subplots(figsize=(11,5))
    x=np.arange(len(tri)); w=0.25
    ax.bar(x-w,tri["Association"],w,label="Association",color="#D85A30")
    ax.bar(x,tri["SHAP"],w,label="SHAP",color="#378ADD")
    ax.bar(x+w,tri["Clustering"],w,label="Clustering",color="#1D9E75")
    ax.set_xticks(x); ax.set_xticklabels(tri["Factor"],rotation=15); ax.set_ylabel("Normalised evidence strength"); ax.legend(); ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout(); st.pyplot(fig)
    st.subheader("Central conclusion")
    st.markdown('<div class="conclusion-box"><p style="margin:0;">Flight disruption is <b>multifactorial</b>. No single factor dominates. Prediction is inherently limited (best model 0.71 ROC-AUC even after tuning). Three complementary methods triangulate on time-of-day, season, and weather as the principal drivers, with delays being congestion-driven and cancellations weather-driven.</p></div>',unsafe_allow_html=True)
