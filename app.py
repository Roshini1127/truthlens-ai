
import re, html
from datetime import datetime
from urllib.parse import quote_plus, urlparse
import xml.etree.ElementTree as ET
import pandas as pd
import requests
import streamlit as st
import altair as alt
import re, html
from datetime import datetime
from urllib.parse import quote_plus, urlparse
import xml.etree.ElementTree as ET
import pandas as pd
import requests
import streamlit as st
import altair as alt
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

BASE_DIR = Path(__file__).resolve().parent

st.set_page_config(
    page_title="TruthLens AI",
    page_icon=str(BASE_DIR / "truthlens_ai_app_icon.png"),
    layout="wide"
)
st.image(str(BASE_DIR / "truthlens_ai_logo.png"), width=280)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
html,body,[class*="css"]{font-family:'DM Sans',sans-serif}
.stApp{background:radial-gradient(ellipse at 0 0,rgba(124,58,237,.24),transparent 34%),radial-gradient(ellipse at 100% 0,rgba(192,38,211,.15),transparent 32%),linear-gradient(135deg,#10091f,#17102b 52%,#0e0a1a);color:#f4efff}
[data-testid="stHeader"]{background:rgba(16,9,31,.86)}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#21133d,#171027 65%,#100b1d);border-right:1px solid #8b5cf633}
[data-testid="stSidebar"] *{color:#f2eaff}
h1,h2,h3,h4{font-family:'Space Grotesk',sans-serif!important;color:#f8f4ff!important}
p,li,label,.stMarkdown,.stCaption{color:#e3d9f5}
.hero{padding:1.5rem 1.7rem;border:1px solid #c4b5fd35;border-radius:24px;color:white;background:linear-gradient(115deg,#4c1d95,#6d28d9 48%,#a21caf);box-shadow:0 14px 35px #00000055;margin-bottom:1rem}
.hero h1{color:white!important;font-size:clamp(2rem,4vw,3rem);margin:0}.hero p{color:#f5eaff;margin:.45rem 0 0}
.metric{background:linear-gradient(145deg,#24183d,#1b132e);border:1px solid #a78bfa55;border-radius:17px;padding:1rem;min-height:100px;box-shadow:0 8px 22px #00000022}
.mlabel{color:#c4b5fd;font-size:.82rem;font-weight:600}.mvalue{font-size:1.65rem;font-weight:700;color:#e9d5ff}
div.stButton>button,div.stDownloadButton>button{border-radius:12px;border:1px solid #c4b5fd55;color:white;font-weight:700;background:linear-gradient(100deg,#6d28d9,#7c3aed,#a21caf);transition:all .2s ease}
div.stButton>button:hover,div.stDownloadButton>button:hover{border-color:#e9d5ff;box-shadow:0 0 0 2px #a78bfa44;transform:translateY(-1px)}
[data-testid="stTextInput"] input,[data-testid="stTextArea"] textarea,[data-testid="stNumberInput"] input,[data-testid="stSelectbox"] div[data-baseweb="select"],[data-testid="stMultiSelect"] div[data-baseweb="select"]{background:#211735;color:#f8f4ff;border-color:#6d529b}
[data-testid="stTextInput"] input::placeholder,[data-testid="stTextArea"] textarea::placeholder{color:#aa9bbf}
[data-testid="stMetric"]{background:#211735;border:1px solid #8b5cf644;padding:12px;border-radius:14px}
[data-testid="stMetricLabel"], [data-testid="stMetricValue"]{color:#f3e8ff}
[data-testid="stAlert"]{background:#211735;border:1px solid #8b5cf655;color:#f3e8ff}
[data-testid="stDataFrame"],[data-testid="stTable"]{border:1px solid #8b5cf644;border-radius:12px;overflow:hidden}
hr{border-color:#6d529b55}
</style>
<div class="hero"><div>🔎 TRUTHLENS AI · VERSION 5.1</div><h1>See the claim. Check the evidence.</h1><p>Live news coverage, source comparison, vivid evidence indicators and an ML learning lab.</p></div>
""", unsafe_allow_html=True)

RSS = "https://news.google.com/rss/search?q={}&hl=en-IN&gl=IN&ceid=IN:en"
SUPPORT = ["confirmed","officially announced","official statement","according to","reported","launched","successfully","approved","published","data shows"]
DEBUNK = ["fact check","fact-check","false claim","fake news","misleading","no evidence","not true","rumour","rumor","debunk","fabricated","denied the claim"]
def clean(s): return re.sub(r"\s+"," ",s or "").strip()
def search_news(query, limit=8):
    res=requests.get(RSS.format(quote_plus(query)),headers={"User-Agent":"TruthLensAI/5.0"},timeout=15); res.raise_for_status()
    root=ET.fromstring(res.content); out=[]; seen=set()
    for it in root.findall(".//item")[:limit]:
        title=clean(it.findtext("title")); link=clean(it.findtext("link")); date=clean(it.findtext("pubDate"))
        desc=html.unescape(re.sub(r"<[^>]+>"," ",it.findtext("description") or ""))
        src=it.find("source"); source=clean(src.text if src is not None else "") or urlparse(link).netloc.replace("www.","")
        key=re.sub(r"[^a-z0-9]","",title.lower())
        if title and link and key not in seen:
            seen.add(key); out.append({"title":title,"link":link,"source":source,"date":date,"snippet":clean(desc)[:600]})
    return out
def analyze(claim, articles):
    corpus=" ".join(f"{a['title']} {a['snippet']} {a['source']}" for a in articles).lower()
    words=set(re.findall(r"[a-z0-9₹]+",claim.lower()))-{"the","and","for","with","from","that","this","will","has","have","are","was","were","india","indian"}
    words={w for w in words if len(w)>2}; cwords=set(re.findall(r"[a-z0-9₹]+",corpus))
    overlap=len(words & cwords)/max(1,len(words))
    sh=sum(corpus.count(x) for x in SUPPORT); dh=sum(corpus.count(x) for x in DEBUNK)
    s=min(100,round(overlap*65+min(25,sh*3))); d=min(100,round(min(70,dh*12)))

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

st.set_page_config(page_title="TruthLens AI", page_icon="🔎", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
html,body,[class*="css"]{font-family:'DM Sans',sans-serif}
.stApp{background:radial-gradient(ellipse at 0 0,rgba(124,58,237,.24),transparent 34%),radial-gradient(ellipse at 100% 0,rgba(192,38,211,.15),transparent 32%),linear-gradient(135deg,#10091f,#17102b 52%,#0e0a1a);color:#f4efff}
[data-testid="stHeader"]{background:rgba(16,9,31,.86)}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#21133d,#171027 65%,#100b1d);border-right:1px solid #8b5cf633}
[data-testid="stSidebar"] *{color:#f2eaff}
h1,h2,h3,h4{font-family:'Space Grotesk',sans-serif!important;color:#f8f4ff!important}
p,li,label,.stMarkdown,.stCaption{color:#e3d9f5}
.hero{padding:1.5rem 1.7rem;border:1px solid #c4b5fd35;border-radius:24px;color:white;background:linear-gradient(115deg,#4c1d95,#6d28d9 48%,#a21caf);box-shadow:0 14px 35px #00000055;margin-bottom:1rem}
.hero h1{color:white!important;font-size:clamp(2rem,4vw,3rem);margin:0}.hero p{color:#f5eaff;margin:.45rem 0 0}
.metric{background:linear-gradient(145deg,#24183d,#1b132e);border:1px solid #a78bfa55;border-radius:17px;padding:1rem;min-height:100px;box-shadow:0 8px 22px #00000022}
.mlabel{color:#c4b5fd;font-size:.82rem;font-weight:600}.mvalue{font-size:1.65rem;font-weight:700;color:#e9d5ff}
div.stButton>button,div.stDownloadButton>button{border-radius:12px;border:1px solid #c4b5fd55;color:white;font-weight:700;background:linear-gradient(100deg,#6d28d9,#7c3aed,#a21caf);transition:all .2s ease}
div.stButton>button:hover,div.stDownloadButton>button:hover{border-color:#e9d5ff;box-shadow:0 0 0 2px #a78bfa44;transform:translateY(-1px)}
[data-testid="stTextInput"] input,[data-testid="stTextArea"] textarea,[data-testid="stNumberInput"] input,[data-testid="stSelectbox"] div[data-baseweb="select"],[data-testid="stMultiSelect"] div[data-baseweb="select"]{background:#211735;color:#f8f4ff;border-color:#6d529b}
[data-testid="stTextInput"] input::placeholder,[data-testid="stTextArea"] textarea::placeholder{color:#aa9bbf}
[data-testid="stMetric"]{background:#211735;border:1px solid #8b5cf644;padding:12px;border-radius:14px}
[data-testid="stMetricLabel"], [data-testid="stMetricValue"]{color:#f3e8ff}
[data-testid="stAlert"]{background:#211735;border:1px solid #8b5cf655;color:#f3e8ff}
[data-testid="stDataFrame"],[data-testid="stTable"]{border:1px solid #8b5cf644;border-radius:12px;overflow:hidden}
hr{border-color:#6d529b55}
</style>
<div class="hero"><div>🔎 TRUTHLENS AI · VERSION 5.1</div><h1>See the claim. Check the evidence.</h1><p>Live news coverage, source comparison, vivid evidence indicators and an ML learning lab.</p></div>
""", unsafe_allow_html=True)

RSS = "https://news.google.com/rss/search?q={}&hl=en-IN&gl=IN&ceid=IN:en"
SUPPORT = ["confirmed","officially announced","official statement","according to","reported","launched","successfully","approved","published","data shows"]
DEBUNK = ["fact check","fact-check","false claim","fake news","misleading","no evidence","not true","rumour","rumor","debunk","fabricated","denied the claim"]
def clean(s): return re.sub(r"\s+"," ",s or "").strip()
def search_news(query, limit=8):
    res=requests.get(RSS.format(quote_plus(query)),headers={"User-Agent":"TruthLensAI/5.0"},timeout=15); res.raise_for_status()
    root=ET.fromstring(res.content); out=[]; seen=set()
    for it in root.findall(".//item")[:limit]:
        title=clean(it.findtext("title")); link=clean(it.findtext("link")); date=clean(it.findtext("pubDate"))
        desc=html.unescape(re.sub(r"<[^>]+>"," ",it.findtext("description") or ""))
        src=it.find("source"); source=clean(src.text if src is not None else "") or urlparse(link).netloc.replace("www.","")
        key=re.sub(r"[^a-z0-9]","",title.lower())
        if title and link and key not in seen:
            seen.add(key); out.append({"title":title,"link":link,"source":source,"date":date,"snippet":clean(desc)[:600]})
    return out
def analyze(claim, articles):
    corpus=" ".join(f"{a['title']} {a['snippet']} {a['source']}" for a in articles).lower()
    words=set(re.findall(r"[a-z0-9₹]+",claim.lower()))-{"the","and","for","with","from","that","this","will","has","have","are","was","were","india","indian"}
    words={w for w in words if len(w)>2}; cwords=set(re.findall(r"[a-z0-9₹]+",corpus))
    overlap=len(words & cwords)/max(1,len(words))
    sh=sum(corpus.count(x) for x in SUPPORT); dh=sum(corpus.count(x) for x in DEBUNK)
    s=min(100,round(overlap*65+min(25,sh*3))); d=min(100,round(min(70,dh*12)))
    if not articles: status="UNVERIFIED — NO SEARCH RESULTS"; explanation="No matching articles were returned. This does not mean the claim is true or false."
    elif dh>=2 and d>=s: status="POSSIBLE DEBUNKING EVIDENCE — REVIEW SOURCES"; explanation="Some results contain fact-checking or debunking language. Open the sources and confirm they address the exact same claim."
    elif overlap>=.45 and s>=35 and d<25: status="RELATED COVERAGE FOUND — NOT PROVEN TRUE"; explanation="Search results share terms with the claim. Keyword overlap shows relevance, not factual correctness."
    else: status="INCONCLUSIVE — HUMAN REVIEW NEEDED"; explanation="Search results do not provide a clear match. Try a shorter claim and inspect primary sources or reputable fact-checkers."
    return {"support":s,"debunk":d,"overlap":round(overlap*100),"status":status,"explanation":explanation}
def normalize_label(x):
    x=str(x).strip().lower()
    return {"true":"REAL","real":"REAL","mostly-true":"REAL","mostly true":"REAL","false":"FAKE","fake":"FAKE","pants-fire":"FAKE","mostly-false":"FAKE","mostly false":"FAKE"}.get(x,x.upper())
def train(data):
    df=data.copy(); df.columns=[str(c).strip().lower() for c in df.columns]
    if not {"text","label"}.issubset(df.columns): raise ValueError("CSV needs columns named text and label.")
    df=df[["text","label"]].dropna(); df["text"]=df.text.astype(str).str.strip(); df["label"]=df.label.map(normalize_label)
    df=df[(df.text!="") & df.label.isin(["REAL","FAKE"])]
    if len(df)<10 or df.label.nunique()<2 or df.label.value_counts().min()<2: raise ValueError("Need at least 10 rows and at least 2 examples for each REAL/FAKE class.")
    xtr,xte,ytr,yte=train_test_split(df.text,df.label,test_size=max(.25,min(.4,2/len(df))),random_state=42,stratify=df.label)
    model=Pipeline([("tfidf",TfidfVectorizer(ngram_range=(1,2),max_features=12000)),("classifier",LogisticRegression(max_iter=1000,class_weight="balanced"))])
    model.fit(xtr,ytr); pred=model.predict(xte)
    return model,df,{"accuracy":accuracy_score(yte,pred),"report":pd.DataFrame(classification_report(yte,pred,output_dict=True,zero_division=0)).T,"cm":pd.DataFrame(confusion_matrix(yte,pred,labels=["FAKE","REAL"]),index=["Actual FAKE","Actual REAL"],columns=["Predicted FAKE","Predicted REAL"]),"ntrain":len(ytr),"ntest":len(yte)}

if "history" not in st.session_state: st.session_state.history=[]
if "last" not in st.session_state: st.session_state.last=None
if "model" not in st.session_state: st.session_state.model=None
if "metrics" not in st.session_state: st.session_state.metrics=None
with st.sidebar:
    st.title("🔎 TruthLens AI"); page=st.radio("Navigate",["Claim checker","Check history","ML lab","Screenshot OCR","Guide"])
    st.caption("Scores are evidence signals, not truth probabilities.")
if page=="Claim checker":
    st.subheader("✨ Live claim explorer")
    claim=st.text_area("Paste a headline or statement",height=110,placeholder="Example: Chandrayaan-3 landed on the Moon in August 2023.")
    limit=st.slider("Articles to compare",5,10,8)
    use_model=st.checkbox("Also show my trained ML model prediction")
    if st.button("Verify claim against live news",type="primary",use_container_width=True):
        if not claim.strip(): st.warning("Enter a claim first.")
        else:
            try:
                with st.spinner("Searching live news coverage..."):
                    arts=search_news(claim.strip(),limit); a=analyze(claim.strip(),arts)
                rec={"time":datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z"),"claim":claim.strip(),**a,"articles":arts}
                st.session_state.last=rec; st.session_state.history.insert(0,rec); st.session_state.history=st.session_state.history[:50]
            except Exception as e: st.error(f"News search failed. Check your internet connection and try again. Details: {e}")
    r=st.session_state.last
    if r:
        st.divider(); st.subheader(r["status"])
        cols=st.columns(3)
        for col,label,val in zip(cols,["Evidence support","Debunking language","Related-term overlap"],[r["support"],r["debunk"],r["overlap"]]):
            col.markdown(f'<div class="metric"><div class="mlabel">{label}</div><div class="mvalue">{val}/100</div></div>',unsafe_allow_html=True)
        st.caption("These heuristic indicators are not calibrated probabilities and do not prove a claim true or false.")
        chart=alt.Chart(pd.DataFrame({"Indicator":["Evidence support","Debunking language","Term overlap"],"Score":[r["support"],r["debunk"],r["overlap"]]})).mark_bar(cornerRadiusEnd=8).encode(
            x=alt.X("Score:Q",scale=alt.Scale(domain=[0,100]),title="Indicator (0–100)"),y=alt.Y("Indicator:N",sort="-x",title=None),
            color=alt.Color("Indicator:N",scale=alt.Scale(domain=["Evidence support","Debunking language","Term overlap"],range=["#7C5CFC","#E879C7","#58B8D8"]),legend=None),tooltip=["Indicator","Score"]).properties(height=170)
        st.altair_chart(chart,use_container_width=True)
        st.markdown("### 🧠 Explanation"); st.info(r["explanation"])
        if use_model:
            if st.session_state.model is None: st.warning("Train a model in the ML lab first.")
            else:
                pred=st.session_state.model.predict([r["claim"]])[0]; probs=st.session_state.model.predict_proba([r["claim"]])[0]
                st.write("Experimental ML prediction:",pred)
                st.bar_chart(pd.DataFrame({"Model score":probs},index=st.session_state.model.named_steps["classifier"].classes_))
                st.caption("Learned model scores can be unreliable when training data is small or biased.")
        st.markdown("### 📰 Live articles and evidence links")
        if not r["articles"]: st.warning("No articles found. Try a shorter search phrase.")
        for i,a in enumerate(r["articles"],1):
            with st.container(border=True):
                st.markdown(f"**{i}. [{a['title']}]({a['link']})**")
                st.caption(f"{a['source']} · {a['date'] or 'Date not supplied'}")
                st.write(a["snippet"] or "No snippet available; open the original article.")
        if r["articles"]:
            st.markdown("#### Source comparison")
            st.dataframe(pd.DataFrame([{"Source":a["source"],"Headline":a["title"],"Published":a["date"],"Link":a["link"]} for a in r["articles"]]),
                         use_container_width=True,hide_index=True,column_config={"Link":st.column_config.LinkColumn("Open source")})
        row={"checked_at":r["time"],"claim":r["claim"],"status":r["status"],"support_indicator":r["support"],"debunking_indicator":r["debunk"],"term_overlap":r["overlap"],"articles_found":len(r["articles"]),"article_links":" | ".join(a["link"] for a in r["articles"])}
        st.download_button("⬇️ Download this report (CSV)",pd.DataFrame([row]).to_csv(index=False).encode(),file_name="truthlens_report.csv",mime="text/csv")
elif page=="Check history":
    st.title("🕘 Check history"); st.caption("History is stored in the current session only. Download it before the session resets.")
    if not st.session_state.history: st.info("Run a claim check to start your history.")
    else:
        rows=[{"checked_at":r["time"],"claim":r["claim"],"status":r["status"],"support_indicator":r["support"],"debunking_indicator":r["debunk"],"articles_found":len(r["articles"])} for r in st.session_state.history]
        df=pd.DataFrame(rows); st.dataframe(df,use_container_width=True,hide_index=True)
        st.download_button("Download all history CSV",df.to_csv(index=False).encode(),file_name="truthlens_history.csv",mime="text/csv")
        choice=st.selectbox("Open previous check",[f"{i+1}. {r['claim'][:90]}" for i,r in enumerate(st.session_state.history)])
        old=st.session_state.history[[f"{i+1}. {r['claim'][:90]}" for i,r in enumerate(st.session_state.history)].index(choice)]
        st.write(old["status"]); st.write(old["explanation"])
        for a in old["articles"]: st.markdown(f"- [{a['title']}]({a['link']}) — {a['source']}")
        if st.button("Clear history"): st.session_state.history=[]; st.session_state.last=None; st.rerun()
elif page=="ML lab":
    st.title("🧪 ML training & evaluation")
    st.write("Train a TF-IDF + Logistic Regression model from a labelled CSV and evaluate it on a held-out test split.")
    st.warning("The included sample dataset is tiny and illustrative. It demonstrates the ML workflow, not real-world accuracy. Use a reputable, documented dataset for project evaluation.")
    st.code('text,label\\n"Chandrayaan-3 landed on the Moon in 2023",REAL\\n"The Moon is made of cheese",FAKE',language="csv")
    uploaded=st.file_uploader("Upload training CSV (columns: text,label)",type=["csv"])
    if uploaded:
        try:
            df=pd.read_csv(uploaded); st.dataframe(df.head(8),use_container_width=True)
            if st.button("Train and evaluate model",type="primary"):
                model,cleaned,met=train(df); st.session_state.model=model; st.session_state.metrics=met
                st.success(f"Trained with {len(cleaned)} labelled rows.")
        except Exception as e: st.error(str(e))
    else:
        if st.button("Train using included demo dataset"):
            try:
                import os
                df=pd.read_csv(os.path.join(os.path.dirname(__file__),"data","demo_training_data.csv"))
                model,cleaned,met=train(df); st.session_state.model=model; st.session_state.metrics=met
                st.success(f"Demo model trained on {len(cleaned)} rows. Do not interpret its scores as real-world performance.")
            except Exception as e: st.error(f"Training failed: {e}")
    if st.session_state.metrics:
        m=st.session_state.metrics; c1,c2,c3=st.columns(3)
        c1.metric("Test accuracy",f"{m['accuracy']*100:.1f}%"); c2.metric("Training rows",m["ntrain"]); c3.metric("Test rows",m["ntest"])
        st.markdown("#### Precision, recall and F1-score"); st.dataframe(m["report"],use_container_width=True)
        st.markdown("#### Confusion matrix"); st.dataframe(m["cm"],use_container_width=True)
        st.download_button("Download metrics CSV",m["report"].to_csv().encode(),file_name="truthlens_ml_metrics.csv",mime="text/csv")
elif page=="Screenshot OCR":
    st.title("🖼️ Screenshot text extraction")
    st.write("Upload a screenshot to extract text. The OCR engine requires Tesseract to be installed.")
    imgfile=st.file_uploader("Choose PNG/JPG screenshot",type=["png","jpg","jpeg"])
    if imgfile:
        st.image(imgfile,use_container_width=True)
        try:
            from PIL import Image
            import pytesseract
            pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
            text=pytesseract.image_to_string(Image.open(imgfile)).strip()
            if text:
                st.text_area("Extracted text — copy into Claim checker",text,height=180)
                st.download_button("Download extracted text",text,file_name="extracted_claim.txt")
            else: st.warning("No text detected. Try a clearer image.")
        except Exception:
            st.error("Tesseract OCR is not installed or unavailable. Windows guide: https://github.com/UB-Mannheim/tesseract/wiki")
elif page=="Guide":
    st.title("📘 How to use TruthLens AI")
    st.markdown("""
- **Claim checker:** paste a claim and search current Google News RSS coverage.
- **Source comparison:** inspect source names, dates, headlines, snippets and original links.
- **History:** export previous checks; session history is not a permanent database.
- **Screenshot OCR:** use the Screenshot OCR page. Install Tesseract on Windows first.
- **ML lab:** upload CSV with `text,label` columns; labels should be `REAL` and `FAKE`. The app reports held-out accuracy, precision, recall, F1 and a confusion matrix.
- **Important:** search indicators and model predictions can be wrong. Check primary sources and reputable fact-checkers before deciding.
""")
