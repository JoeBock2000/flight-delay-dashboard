import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
import joblib
import streamlit.components.v1 as components
from sklearn.metrics import accuracy_score, roc_auc_score
import html as html_lib
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
FLOWCHART_PATH = "flowchart_sequential.html"
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

# Dark "analysis panel" for CSV-upload results. Styling is taken verbatim from
# analysis_panel.html; every value is computed from the real uploaded data.
PANEL_STYLE = """<style>
  html,body{margin:0;padding:0;background:#F4F7FB;}
  *{box-sizing:border-box;}
  .analysis{
    font-family:-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
    max-width:820px;margin:14px auto;border-radius:16px;padding:22px 24px 26px;
    color:#E8EEF7;
    background:linear-gradient(145deg,#1c2230 0%,#141821 45%,#1a2030 100%);
    border:1px solid #2b3346;
    box-shadow:0 10px 30px rgba(10,15,25,.35), inset 0 1px 0 rgba(255,255,255,.04);
  }
  .analysis h3{margin:0 0 4px;font-size:16px;font-weight:600;color:#F4F8FF;letter-spacing:.01em;}
  .analysis .sub{margin:0 0 18px;font-size:12.5px;color:#9DB0CC;}
  .metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:20px;}
  @media(max-width:640px){.metrics{grid-template-columns:repeat(2,1fr);}}
  .metric{
    background:linear-gradient(160deg,#232b3c,#1a2130);
    border:1px solid #313a4f;border-radius:12px;padding:12px 14px;
    box-shadow:inset 0 1px 0 rgba(255,255,255,.05);
  }
  .metric .k{font-size:11px;text-transform:uppercase;letter-spacing:.06em;color:#8697b4;}
  .metric .v{font-size:23px;font-weight:600;margin-top:4px;color:#EAF1FF;}
  .metric .v small{font-size:13px;color:#9DB0CC;font-weight:400;}
  .grid2{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin-bottom:20px;}
  @media(max-width:640px){.grid2{grid-template-columns:1fr;}}
  .card{
    background:linear-gradient(160deg,#20283700,#1a212f);
    border:1px solid #2b3346;border-radius:12px;padding:14px 16px 16px;
  }
  .card h4{margin:0 0 12px;font-size:12.5px;font-weight:600;color:#C4D2E8;letter-spacing:.02em;}
  .cm{display:grid;grid-template-columns:88px 1fr 1fr;grid-auto-rows:auto;gap:6px;font-size:12px;}
  .cm .hd{color:#8697b4;display:flex;align-items:center;justify-content:center;padding:4px;text-align:center;}
  .cm .rl{color:#8697b4;display:flex;align-items:center;padding:4px 6px;}
  .cell{border-radius:9px;padding:14px 8px;text-align:center;}
  .cell .n{font-size:20px;font-weight:600;}
  .cell .t{font-size:10.5px;opacity:.8;margin-top:2px;}
  .tp{background:linear-gradient(160deg,#12463a,#0e3a30);color:#7FE3C0;border:1px solid #1c6b57;}
  .tn{background:linear-gradient(160deg,#123a52,#0e2f42);color:#8BC7F0;border:1px solid #1c567a;}
  .fp{background:linear-gradient(160deg,#4a2f12,#3a260e);color:#F0C08B;border:1px solid #7a531c;}
  .fn{background:linear-gradient(160deg,#4a1f24,#3a171b);color:#F0959B;border:1px solid #7a2b33;}
  .hist{display:flex;align-items:flex-end;gap:4px;height:120px;padding-top:6px;}
  .bar{flex:1;background:linear-gradient(180deg,#4A90D9,#2b5f96);border-radius:4px 4px 0 0;min-height:2px;position:relative;}
  .bar.hot{background:linear-gradient(180deg,#E0844A,#a85d2b);}
  .axis{display:flex;justify-content:space-between;font-size:10px;color:#8697b4;margin-top:6px;}
  .legend{font-size:11px;color:#9DB0CC;margin-top:8px;}
  .legend b{color:#EAF1FF;font-weight:600;}
  .tbl-wrap{max-height:260px;overflow:auto;border:1px solid #2b3346;border-radius:12px;}
  table{width:100%;border-collapse:collapse;font-size:12px;}
  thead th{position:sticky;top:0;background:#1a2130;color:#8697b4;font-weight:600;
    text-align:left;padding:9px 12px;border-bottom:1px solid #2b3346;font-size:11px;
    text-transform:uppercase;letter-spacing:.04em;}
  tbody td{padding:8px 12px;border-bottom:1px solid #232b3a;color:#D5E0F0;}
  tbody tr:last-child td{border-bottom:none;}
  .pill{display:inline-block;padding:2px 9px;border-radius:20px;font-size:11px;font-weight:600;}
  .pill.hi{background:#4a1f24;color:#F0959B;}
  .pill.mid{background:#4a2f12;color:#F0C08B;}
  .pill.lo{background:#123a52;color:#8BC7F0;}
  .ok{color:#7FE3C0;}
  .miss{color:#F0959B;}
</style>"""

def render_analysis_panel(y_true, y_prob, preview_df):
    yprob = np.asarray(y_prob, dtype=float)
    n = int(len(yprob))
    mean_pred = float(yprob.mean())*100 if n else 0.0
    counts, _ = np.histogram(yprob, bins=np.linspace(0.0, 1.0, 11))
    mx = int(counts.max()) if counts.size and counts.max() > 0 else 1
    bars = "".join(
        f'<div class="bar{" hot" if i>=5 else ""}" style="height:{c/mx*100:.1f}%" title="{int(c)} flights"></div>'
        for i, c in enumerate(counts))

    has_labels = y_true is not None
    metric_mid = ""
    conf_card = ""
    if has_labels:
        yt = np.asarray(y_true, dtype=float)
        m = ~np.isnan(yt)
        yt_l = yt[m].astype(int)
        yp_l = yprob[m]
        n_l = int(len(yt_l))
        pred_l = (yp_l >= 0.5).astype(int)
        TP = int(((pred_l == 1) & (yt_l == 1)).sum())
        TN = int(((pred_l == 0) & (yt_l == 0)).sum())
        FP = int(((pred_l == 1) & (yt_l == 0)).sum())
        FN = int(((pred_l == 0) & (yt_l == 1)).sum())
        acc = accuracy_score(yt_l, pred_l)*100 if n_l else 0.0
        auc = roc_auc_score(yt_l, yp_l) if (n_l and len(np.unique(yt_l)) > 1) else None
        auc_str = f"{auc:.3f}" if auc is not None else "n/a"
        metric_mid = (
            f'<div class="metric"><div class="k">ROC-AUC</div><div class="v">{auc_str}</div></div>'
            f'<div class="metric"><div class="k">Accuracy @0.5</div><div class="v">{acc:.1f}<small>%</small></div></div>')
        conf_card = (
            '<div class="card"><h4>Confusion matrix</h4><div class="cm">'
            '<div></div><div class="hd">Predicted on-time</div><div class="hd">Predicted delayed</div>'
            '<div class="rl">Actual on-time</div>'
            f'<div class="cell tn"><div class="n">{TN}</div><div class="t">true on-time</div></div>'
            f'<div class="cell fp"><div class="n">{FP}</div><div class="t">false alarm</div></div>'
            '<div class="rl">Actual delayed</div>'
            f'<div class="cell fn"><div class="n">{FN}</div><div class="t">missed delay</div></div>'
            f'<div class="cell tp"><div class="n">{TP}</div><div class="t">caught delay</div></div>'
            '</div>'
            f'<div class="legend">The model catches <b>{TP} of {TP+FN}</b> real delays.</div></div>')

    metrics_html = (
        f'<div class="metric"><div class="k">Flights scored</div><div class="v">{n}</div></div>'
        f'{metric_mid}'
        f'<div class="metric"><div class="k">Mean predicted</div><div class="v">{mean_pred:.1f}<small>%</small></div></div>')

    hist_card = (
        '<div class="card"><h4>Predicted probability distribution</h4>'
        f'<div class="hist">{bars}</div>'
        '<div class="axis"><span>0%</span><span>25%</span><span>50%</span><span>75%</span><span>100%</span></div>'
        f'<div class="legend">Most predictions cluster low, near the <b>{mean_pred:.0f}%</b> mean &mdash; the model rarely commits to &ldquo;delayed&rdquo;.</div></div>')

    middle = f'<div class="grid2">{conf_card}{hist_card}</div>' if has_labels else hist_card

    has_out = "OUTCOME" in preview_df.columns
    thead = ('<tr><th>Origin</th><th>Hour</th><th>Airline</th><th>Prob.</th><th>Actual</th><th>Hit</th></tr>'
             if has_out else '<tr><th>Origin</th><th>Hour</th><th>Airline</th><th>Prob.</th></tr>')
    rows_html = ""
    for _, r in preview_df.head(12).iterrows():
        origin = html_lib.escape(str(r.get("ORIGIN", "")))
        try:
            hour = str(int(round(float(r.get("DEP_HOUR")))))
        except (TypeError, ValueError):
            hour = html_lib.escape(str(r.get("DEP_HOUR", "")))
        airline = html_lib.escape(str(r.get("AIRLINE_CODE", "")))
        p = float(r.get("Delay probability (%)", 0.0))
        cls = "hi" if p >= 40 else ("mid" if p >= 25 else "lo")
        cells = (f'<td>{origin}</td><td>{hour}</td><td>{airline}</td>'
                 f'<td><span class="pill {cls}">{p:.0f}%</span></td>')
        if has_out:
            actual_raw = str(r.get("OUTCOME", ""))
            actual = html_lib.escape(actual_raw)
            if actual_raw in ("On-time", "Delayed"):
                hit = (p >= 50) == (actual_raw == "Delayed")
                tick = "✓" if hit else "✗"
                hit_cell = f'<td class="{"ok" if hit else "miss"}">{tick}</td>'
            else:
                hit_cell = '<td></td>'
            cells += f'<td>{actual}</td>{hit_cell}'
        rows_html += f'<tr>{cells}</tr>'
    table_card = (
        '<div class="card"><h4>Per-flight results (first 12)</h4>'
        f'<div class="tbl-wrap"><table><thead>{thead}</thead><tbody>{rows_html}</tbody></table></div></div>')

    if has_labels:
        title = "Model performance on your uploaded data"
        sub = f"{n} flights scored against your labels &mdash; data the model did not see during training"
    else:
        title = "Prediction analysis for your uploaded flights"
        sub = f"{n} flights scored from your uploaded CSV (no OUTCOME column provided, so no accuracy metrics)"

    return (
        '<!DOCTYPE html><html><head><meta charset="utf-8">' + PANEL_STYLE + '</head><body>'
        '<div class="analysis">'
        f'<h3>{title}</h3><p class="sub">{sub}</p>'
        f'<div class="metrics">{metrics_html}</div>'
        f'{middle}{table_card}'
        '</div></body></html>')

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
page = st.sidebar.radio("Navigate",["Overview","System","Prediction","Factors","Patterns","Mechanisms","Synthesis"],label_visibility="collapsed")
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

elif page == "System":
    st.title("System architecture")
    st.caption("The dashboard runs on a static, curated dataset stored alongside the app. The architecture extends to real-time ingestion via an API and database as a future step.")
    with open(FLOWCHART_PATH, encoding="utf-8") as f:
        html_content = f.read()
    components.html(html_content, height=720, scrolling=False)

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

    st.markdown("---")
    st.subheader("Batch model testing (CSV upload)")
    st.caption("Score many flights at once. Include an optional OUTCOME column (On-time/Delayed) to measure model performance on your own unseen data.")
    REQUIRED_COLS=["ORIGIN","DEP_HOUR","MONTH","AIRLINE_CODE"]
    template=pd.DataFrame([
        {"ORIGIN":"ATL","DEP_HOUR":19,"MONTH":7,"AIRLINE_CODE":"WN","DISTANCE":800,"THUNDER":0,"FOG":0,"PRCP":0.0,"OUTCOME":"Delayed"},
        {"ORIGIN":"ORD","DEP_HOUR":8,"MONTH":1,"AIRLINE_CODE":"AA","DISTANCE":600,"THUNDER":0,"FOG":1,"PRCP":0.2,"OUTCOME":"On-time"},
    ])
    st.download_button("Download template CSV",template.to_csv(index=False).encode("utf-8"),file_name="flight_test_template.csv",mime="text/csv")
    uploaded=st.file_uploader("Upload a flights CSV",type=["csv"])
    if uploaded is not None:
        try:
            up=pd.read_csv(uploaded)
        except Exception as e:
            up=None; st.error(f"Could not read the CSV file - {e}.")
        if up is not None:
            missing=[c for c in REQUIRED_COLS if c not in up.columns]
            if missing:
                st.error(f"Uploaded CSV is missing required column(s): {', '.join(missing)}. Required columns are ORIGIN, DEP_HOUR, MONTH, AIRLINE_CODE.")
            else:
                try:
                    dist_mean=float(df["DISTANCE"].mean())
                    feat_rows=[]
                    for _,r in up.iterrows():
                        o=str(r["ORIGIN"]); dh=int(r["DEP_HOUR"]); mo=int(r["MONTH"]); al=str(r["AIRLINE_CODE"])
                        dist=float(r["DISTANCE"]) if ("DISTANCE" in up.columns and pd.notna(r["DISTANCE"])) else dist_mean
                        th=int(r["THUNDER"]) if ("THUNDER" in up.columns and pd.notna(r["THUNDER"])) else 0
                        fo=int(r["FOG"]) if ("FOG" in up.columns and pd.notna(r["FOG"])) else 0
                        pr=float(r["PRCP"]) if ("PRCP" in up.columns and pd.notna(r["PRCP"])) else 0.0
                        ref_r=airport_ref.loc[o]; se=season_from_month(mo); td=tod_from_hour(dh)
                        feat_rows.append({"TIME_OF_DAY":encoders["TIME_OF_DAY"].transform([td])[0],"FOG":fo,"SEASON":encoders["SEASON"].transform([se])[0],"IS_WEEKEND":0,"AIRLINE_CODE":encoders["AIRLINE_CODE"].transform([al])[0],"MONTH":mo,"DEP_HOUR":dh,"DAY_OF_WEEK":5,"ORIGIN":encoders["ORIGIN"].transform([o])[0],"WSF2":ref_r["WSF2"],"THUNDER":th,"HAZE_SMOKE":1 if ref_r["HAZE_SMOKE"]>0.5 else 0,"PRCP":pr,"TMAX":ref_r["TMAX"],"TMIN":ref_r["TMIN"],"DISTANCE":dist,"AWND":ref_r["AWND"],"DAILY_TRAFFIC":ref_r["DAILY_TRAFFIC"],"SNOW":ref_r["SNOW"],"SNWD":ref_r["SNWD"]})
                    X_batch=pd.DataFrame(feat_rows)[MODEL_FEATURES]
                    probs=model.predict_proba(X_batch)[:,1]*100
                    results=up.copy(); results["Delay probability (%)"]=probs.round(1)
                    if "OUTCOME" in up.columns:
                        lab=up["OUTCOME"].isin(["On-time","Delayed"])
                        y_true=np.where(lab.values,(up["OUTCOME"].astype(str)=="Delayed").values,np.nan)
                    else:
                        y_true=None
                    components.html(render_analysis_panel(y_true,probs/100.0,results),height=900,scrolling=False)
                    st.dataframe(results,hide_index=True,use_container_width=True)
                except Exception as e:
                    st.error(f"Could not process the CSV - {e}. Check that ORIGIN and AIRLINE_CODE values appear in the training data and that numeric columns contain valid values.")

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
