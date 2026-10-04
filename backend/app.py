from pathlib import Path
from datetime import datetime, timedelta, timezone
import base64, hashlib, hmac, json, os, secrets

from dotenv import load_dotenv

load_dotenv()

import joblib
import pandas as pd
import mysql.connector
from mysql.connector import Error
from fastapi import Depends, FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"
REPORT_DIR.mkdir(exist_ok=True)

DB_HOST = os.getenv("DB_HOST", "127.0.0.1")
DB_PORT = int(os.getenv("DB_PORT", "3306"))
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_NAME = os.getenv("DB_NAME", "ai_health_dashboard")
SECRET = os.getenv("HEALTH_APP_SECRET", "change-this-secret-in-production")
TOKEN_HOURS = 12

def mysql_conn(database=True):
    args = {"host": DB_HOST, "port": DB_PORT, "user": DB_USER, "password": DB_PASSWORD}
    if database:
        args["database"] = DB_NAME
    return mysql.connector.connect(**args)

def init_db():
    try:
        conn = mysql_conn(False)
        cur = conn.cursor()
        cur.execute(f"CREATE DATABASE IF NOT EXISTS `{DB_NAME}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
        cur.close(); conn.close()
        conn = mysql_conn(True); cur = conn.cursor()
        cur.execute("""CREATE TABLE IF NOT EXISTS users (
            id INT AUTO_INCREMENT PRIMARY KEY, username VARCHAR(50) NOT NULL UNIQUE,
            email VARCHAR(120) NOT NULL UNIQUE, password_hash TEXT NOT NULL,
            full_name VARCHAR(100) DEFAULT '', phone VARCHAR(30) DEFAULT '',
            created_at DATETIME NOT NULL) ENGINE=InnoDB""")
        cur.execute("""CREATE TABLE IF NOT EXISTS predictions (
            id INT AUTO_INCREMENT PRIMARY KEY, user_id INT NOT NULL,
            prediction_type VARCHAR(50) NOT NULL, prediction VARCHAR(120) NOT NULL,
            risk_percentage DECIMAL(6,2) NULL, inputs_json LONGTEXT NOT NULL,
            recommendations_json LONGTEXT NOT NULL, explanation TEXT,
            created_at DATETIME NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE,
            INDEX idx_predictions_user(user_id)) ENGINE=InnoDB""")
        cur.execute("""CREATE TABLE IF NOT EXISTS reports (
            id INT AUTO_INCREMENT PRIMARY KEY, user_id INT NOT NULL,
            prediction_id INT NOT NULL, file_name VARCHAR(255) NOT NULL,
            file_path TEXT NOT NULL, downloads INT NOT NULL DEFAULT 0,
            created_at DATETIME NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE,
            FOREIGN KEY(prediction_id) REFERENCES predictions(id) ON DELETE CASCADE,
            INDEX idx_reports_user(user_id)) ENGINE=InnoDB""")
        conn.commit(); cur.close(); conn.close()
    except Error as e:
        raise RuntimeError(f"MySQL connection failed. Ensure MySQL Server is running and your DB settings are correct. Details: {e}")

class RegisterIn(BaseModel):
    username: str = Field(min_length=2, max_length=50)
    email: str = Field(min_length=5, max_length=120)
    password: str = Field(min_length=6, max_length=128)
    full_name: str = Field(default="", max_length=100)
    phone: str = Field(default="", max_length=30)

class LoginIn(BaseModel):
    email: str
    password: str

class ProfileIn(BaseModel):
    username: str = Field(min_length=2, max_length=50)
    full_name: str = Field(default="", max_length=100)
    phone: str = Field(default="", max_length=30)

class ManualHistoryIn(BaseModel):
    prediction_type: str
    prediction: str
    risk_percentage: float | None = None
    inputs: dict = {}
    recommendations: list[str] = []
    explanation: str = ""

def hash_password(password, salt=None):
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 210000)
    return base64.b64encode(salt + digest).decode()

def verify_password(password, stored):
    raw = base64.b64decode(stored.encode()); salt, expected = raw[:16], raw[16:]
    actual = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 210000)
    return hmac.compare_digest(actual, expected)

def make_token(user_id):
    payload = {"sub": user_id, "exp": int((datetime.now(timezone.utc)+timedelta(hours=TOKEN_HOURS)).timestamp())}
    body = base64.urlsafe_b64encode(json.dumps(payload,separators=(",",":")).encode()).decode().rstrip("=")
    sig = hmac.new(SECRET.encode(), body.encode(), hashlib.sha256).hexdigest()
    return body + "." + sig

def current_user(authorization: str | None = Header(default=None)):
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(401, "Please log in to continue.")
    try:
        body, sig = authorization.split(" ",1)[1].split(".",1)
        if not hmac.compare_digest(sig, hmac.new(SECRET.encode(),body.encode(),hashlib.sha256).hexdigest()): raise ValueError()
        payload = json.loads(base64.urlsafe_b64decode(body + "=" * (-len(body)%4)))
        if payload["exp"] < int(datetime.now(timezone.utc).timestamp()): raise ValueError()
        conn=mysql_conn(); cur=conn.cursor(dictionary=True); cur.execute("SELECT * FROM users WHERE id=%s",(payload["sub"],)); user=cur.fetchone(); cur.close(); conn.close()
        if not user: raise ValueError()
        return user
    except Exception:
        raise HTTPException(401,"Your session is invalid or expired. Please log in again.")

def create_pdf(user, report_id, prediction_type, prediction, risk, inputs, recommendations, explanation, created_at):
    file_name=f"health_report_{report_id}_{prediction_type.lower().replace(' ','_')}.pdf"
    path=REPORT_DIR/file_name
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name="ReportTitle",parent=styles["Title"],fontSize=21,leading=25,alignment=TA_CENTER,textColor=colors.HexColor("#176b87")))
    styles.add(ParagraphStyle(name="Small",parent=styles["Normal"],fontSize=9,textColor=colors.HexColor("#64748b")))
    doc=SimpleDocTemplate(str(path),pagesize=A4,rightMargin=18*mm,leftMargin=18*mm,topMargin=16*mm,bottomMargin=16*mm)
    story=[Paragraph("AI Health Dashboard",styles["ReportTitle"]),Spacer(1,5*mm),Paragraph("Personal Health Assessment Report",styles["Heading2"]),Spacer(1,3*mm)]
    rows=[["User details","Value"],["Name",user.get("full_name") or user["username"]],["Username",user["username"]],["Email",user["email"]],["Phone",user.get("phone") or "—"]]
    t=Table(rows,colWidths=[45*mm,125*mm]); t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#176b87")),("TEXTCOLOR",(0,0),(-1,0),colors.white),("GRID",(0,0),(-1,-1),.4,colors.HexColor("#cbd5e1")),("BACKGROUND",(0,1),(0,-1),colors.HexColor("#eef8fb")),("PADDING",(0,0),(-1,-1),7)])); story += [t,Spacer(1,7*mm),Paragraph(f"Assessment: {prediction_type}",styles["Heading2"]),Paragraph(f"Date & time: {created_at}",styles["Small"]),Spacer(1,3*mm)]
    summary=[["Result","Risk / Score"],[prediction,f"{risk:.2f}%" if risk is not None else "—"]]
    t=Table(summary,colWidths=[120*mm,50*mm]); t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#176b87")),("TEXTCOLOR",(0,0),(-1,0),colors.white),("GRID",(0,0),(-1,-1),.4,colors.HexColor("#cbd5e1")),("PADDING",(0,0),(-1,-1),7)])); story += [t,Spacer(1,7*mm),Paragraph("Input values",styles["Heading3"])]
    ir=[[str(k),str(v)] for k,v in inputs.items()]
    if ir:
        t=Table([["Field","Value"]]+ir,colWidths=[95*mm,75*mm]); t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#e9f7f7")),("GRID",(0,0),(-1,-1),.35,colors.HexColor("#cbd5e1")),("PADDING",(0,0),(-1,-1),6)])); story.append(t)
    story += [Spacer(1,6*mm),Paragraph("Explanation",styles["Heading3"]),Paragraph(explanation or "This is an AI/ML screening result and not a medical diagnosis.",styles["BodyText"]),Spacer(1,5*mm),Paragraph("Recommendations",styles["Heading3"])]
    for r in recommendations: story.append(Paragraph("• "+r,styles["BodyText"]))
    story.append(Spacer(1,8*mm)); story.append(Paragraph("Important: This report is for informational screening only and does not replace a qualified healthcare professional.",styles["Small"]))
    doc.build(story)
    return file_name,str(path)

def record_prediction(user,prediction_type,prediction,risk,inputs,recommendations,explanation):
    created=datetime.now().astimezone().replace(tzinfo=None); display=created.strftime("%d %b %Y, %I:%M %p")
    conn=mysql_conn(); cur=conn.cursor()
    cur.execute("INSERT INTO predictions(user_id,prediction_type,prediction,risk_percentage,inputs_json,recommendations_json,explanation,created_at) VALUES(%s,%s,%s,%s,%s,%s,%s,%s)",(user["id"],prediction_type,prediction,risk,json.dumps(inputs),json.dumps(recommendations),explanation,created)); pid=cur.lastrowid
    cur.execute("INSERT INTO reports(user_id,prediction_id,file_name,file_path,created_at) VALUES(%s,%s,%s,%s,%s)",(user["id"],pid,"pending","pending",created)); rid=cur.lastrowid
    file_name,path=create_pdf(user,rid,prediction_type,prediction,risk,inputs,recommendations,explanation,display)
    cur.execute("UPDATE reports SET file_name=%s,file_path=%s WHERE id=%s",(file_name,path,rid)); conn.commit(); cur.close(); conn.close()
    return pid,rid,display

app=FastAPI(title="AI Health Dashboard API",version="3.0")
app.add_middleware(CORSMiddleware,allow_origins=["*"],allow_credentials=False,allow_methods=["*"],allow_headers=["*"])

# ML assets
heart_model=joblib.load(MODEL_DIR/"heart_model.pkl"); heart_scaler=joblib.load(MODEL_DIR/"heart_scaler.pkl")
diabetes_model=joblib.load(MODEL_DIR/"diabetes_model.pkl"); diabetes_scaler=joblib.load(MODEL_DIR/"diabetes_scaler.pkl")
mental_model=joblib.load(MODEL_DIR/"mental_model.pkl"); mental_scaler=joblib.load(MODEL_DIR/"mental_scaler.pkl")
gender_encoder=joblib.load(MODEL_DIR/"gender_encoder.pkl"); smoking_encoder=joblib.load(MODEL_DIR/"smoking_encoder.pkl"); mental_label_encoders=joblib.load(MODEL_DIR/"mental_label_encoders.pkl")

init_db()

@app.get("/")
def home(): return {"message":"AI Health Dashboard API Running","database":"MySQL"}
@app.get("/health")
def health():
    try:
        conn=mysql_conn(); cur=conn.cursor(); cur.execute("SELECT 1"); cur.fetchone(); cur.close(); conn.close(); return {"status":"Backend Working Successfully","database":"MySQL connected"}
    except Exception as e: raise HTTPException(503,f"Database unavailable: {e}")

@app.post("/register")
def register(data:RegisterIn):
    email=data.email.strip().lower(); username=data.username.strip()
    if "@" not in email: raise HTTPException(400,"Enter a valid email address.")
    try:
        conn=mysql_conn(); cur=conn.cursor(dictionary=True); cur.execute("SELECT id FROM users WHERE email=%s OR username=%s",(email,username));
        if cur.fetchone(): cur.close(); conn.close(); raise HTTPException(409,"Email or username already registered.")
        cur2=conn.cursor(); cur2.execute("INSERT INTO users(username,email,password_hash,full_name,phone,created_at) VALUES(%s,%s,%s,%s,%s,%s)",(username,email,hash_password(data.password),data.full_name.strip(),data.phone.strip(),datetime.now().astimezone().replace(tzinfo=None))); uid=cur2.lastrowid; conn.commit(); cur.close(); cur2.close(); conn.close()
        return {"message":"Registration successful","token":make_token(uid),"user":{"id":uid,"username":username,"email":email,"full_name":data.full_name.strip(),"phone":data.phone.strip()}}
    except HTTPException: raise
    except Error as e: raise HTTPException(500,f"Could not create account: {e}")

@app.post("/login")
def login(data:LoginIn):
    email=data.email.strip().lower()
    conn=mysql_conn(); cur=conn.cursor(dictionary=True); cur.execute("SELECT * FROM users WHERE LOWER(email)=LOWER(%s)",(email,)); user=cur.fetchone(); cur.close(); conn.close()
    try:
        valid=bool(user) and verify_password(data.password,user["password_hash"])
    except Exception:
        valid=False
    if not valid: raise HTTPException(401,"Incorrect email or password. If you created the account successfully, use the same email address and password you registered with.")
    return {"message":"Login successful","token":make_token(user["id"]),"user":{"id":user["id"],"username":user["username"],"email":user["email"],"full_name":user["full_name"],"phone":user["phone"]}}

@app.get("/profile")
def profile(user=Depends(current_user)): return {k:user[k] for k in ["id","username","email","full_name","phone","created_at"]}

@app.put("/profile")
def update_profile(data:ProfileIn,user=Depends(current_user)):
    conn=mysql_conn(); cur=conn.cursor(); cur.execute("SELECT id FROM users WHERE username=%s AND id<>%s",(data.username.strip(),user["id"]))
    if cur.fetchone(): cur.close(); conn.close(); raise HTTPException(409,"Username already exists.")
    cur.execute("UPDATE users SET username=%s,full_name=%s,phone=%s WHERE id=%s",(data.username.strip(),data.full_name.strip(),data.phone.strip(),user["id"])); conn.commit(); cur.close(); conn.close(); return profile(user=current_user(authorization="Bearer "+make_token(user["id"])))

@app.get("/history")
def history(user=Depends(current_user)):
    conn=mysql_conn(); cur=conn.cursor(dictionary=True); cur.execute("""SELECT p.*,r.id AS report_id FROM predictions p LEFT JOIN reports r ON r.prediction_id=p.id WHERE p.user_id=%s ORDER BY p.id DESC""",(user["id"],)); rows=cur.fetchall(); cur.close(); conn.close()
    return [{"id":r["id"],"prediction_type":r["prediction_type"],"prediction":r["prediction"],"risk_percentage":float(r["risk_percentage"]) if r["risk_percentage"] is not None else None,"inputs":json.loads(r["inputs_json"]),"recommendations":json.loads(r["recommendations_json"]),"explanation":r["explanation"],"created_at":r["created_at"].strftime("%d %b %Y, %I:%M %p"),"report_id":r["report_id"]} for r in rows]

@app.get("/reports")
def reports(user=Depends(current_user)):
    conn=mysql_conn(); cur=conn.cursor(dictionary=True); cur.execute("SELECT * FROM reports WHERE user_id=%s ORDER BY id DESC",(user["id"],)); rows=cur.fetchall(); cur.close(); conn.close()
    return [{"id":r["id"],"prediction_id":r["prediction_id"],"file_name":r["file_name"],"downloads":r["downloads"],"created_at":r["created_at"].strftime("%d %b %Y, %I:%M %p")} for r in rows]

def get_report(report_id,user):
    conn=mysql_conn(); cur=conn.cursor(dictionary=True); cur.execute("SELECT * FROM reports WHERE id=%s AND user_id=%s",(report_id,user["id"])); row=cur.fetchone(); cur.close(); conn.close()
    if not row: raise HTTPException(404,"Report not found or you do not have access to it.")
    path=Path(row["file_path"])
    if not path.exists() or path.stat().st_size==0: raise HTTPException(404,"The PDF file is missing. Generate the report again.")
    return row,path

@app.get("/reports/{report_id}/view")
def view_report(report_id:int,user=Depends(current_user)):
    row,path=get_report(report_id,user); return FileResponse(str(path),media_type="application/pdf",headers={"Content-Disposition":f'inline; filename="{row["file_name"]}"'})

@app.get("/reports/{report_id}/download")
def download_report(report_id:int,user=Depends(current_user)):
    row,path=get_report(report_id,user)
    conn=mysql_conn(); cur=conn.cursor(); cur.execute("UPDATE reports SET downloads=downloads+1 WHERE id=%s",(report_id,)); conn.commit(); cur.close(); conn.close()
    return FileResponse(str(path),media_type="application/pdf",filename=row["file_name"],headers={"Content-Disposition":f'attachment; filename="{row["file_name"]}"','Cache-Control':'no-store'})

@app.post("/history/manual")
def manual_history(data:ManualHistoryIn,user=Depends(current_user)):

    pid,rid,created=record_prediction(user,data.prediction_type,data.prediction,data.risk_percentage,data.inputs,data.recommendations,data.explanation); return {"prediction_id":pid,"report_id":rid,"created_at":created}

@app.post("/predict/heart")
def predict_heart(data:dict,user=Depends(current_user)):
    try:
        features=pd.DataFrame([{k:data[k] for k in ["age","sex","trestbps","chol","fbs","thalach","exang"]}]); scaled=heart_scaler.transform(features); prediction=int(heart_model.predict(scaled)[0]); risk=round(float(heart_model.predict_proba(scaled)[0][1])*100,2)
        result="Heart Disease Detected" if prediction else "No Heart Disease Detected"
        tips=( ["Schedule a medical review and discuss cardiovascular risk factors.","Monitor blood pressure and cholesterol.","Choose vegetables, whole grains and lower-saturated-fat foods.","Stay physically active within your clinician's advice.","Avoid smoking and manage stress."] if prediction else ["Maintain a balanced diet and regular activity.","Keep blood pressure and cholesterol in a healthy range.","Avoid smoking and limit excess alcohol.","Continue routine health checkups."] )
        explanation=f"The model classified this screening input as {'higher' if prediction else 'lower'} heart-disease risk. The probability is a model estimate, not a diagnosis."
        inputs={"Age":data["age"],"Gender":"Male" if data["sex"]==1 else "Female","Blood Pressure":data["trestbps"],"Cholesterol":data["chol"],"High Blood Sugar":"Yes" if data["fbs"] else "No","Heart Rate":data["thalach"],"Chest Pain During Exercise":"Yes" if data["exang"] else "No"}
        pid,rid,created=record_prediction(user,"Heart Disease",result,risk,inputs,tips,explanation); return {"Prediction":result,"RiskPercentage":risk,"Tips":tips,"Explanation":explanation,"PredictionId":pid,"ReportId":rid,"CreatedAt":created}
    except Exception as e: raise HTTPException(400,f"Heart prediction failed: {e}")

@app.post("/predict/diabetes")
def predict_diabetes(data:dict,user=Depends(current_user)):
    try:
        gender=int(gender_encoder.transform([data["gender"]])[0]); smoking=int(smoking_encoder.transform([data["smoking_history"]])[0]); features=pd.DataFrame([{"gender":gender,"age":data["age"],"hypertension":data["hypertension"],"heart_disease":data["heart_disease"],"smoking_history":smoking,"bmi":data["bmi"],"HbA1c_level":data["HbA1c_level"],"blood_glucose_level":data["blood_glucose_level"]}]); scaled=diabetes_scaler.transform(features); prediction=int(diabetes_model.predict(scaled)[0]); risk=round(float(diabetes_model.predict_proba(scaled)[0][1])*100,2); result="Diabetic" if prediction else "Non-Diabetic"
        tips=(["Discuss this result with a healthcare professional.","Monitor blood glucose as advised.","Prioritize fibre-rich foods and balanced portions.","Stay physically active as appropriate.","Follow prescribed treatment and screening plans."] if prediction else ["Maintain a balanced, fibre-rich diet.","Stay physically active and maintain a healthy weight.","Limit excessive added sugar and refined foods.","Monitor glucose during routine checkups."])
        explanation=f"The classification model returned '{result}'. The percentage is the model's estimated probability for the diabetic class and is not a diagnosis."
        inputs={"Gender":data["gender"],"Age":data["age"],"Hypertension":"Yes" if data["hypertension"] else "No","Heart Disease":"Yes" if data["heart_disease"] else "No","Smoking History":data["smoking_history"],"BMI":data["bmi"],"HbA1c":data["HbA1c_level"],"Blood Glucose":data["blood_glucose_level"]}
        pid,rid,created=record_prediction(user,"Diabetes",result,risk,inputs,tips,explanation); return {"Prediction":result,"RiskPercentage":risk,"Tips":tips,"Explanation":explanation,"PredictionId":pid,"ReportId":rid,"CreatedAt":created}
    except Exception as e: raise HTTPException(400,f"Diabetes prediction failed: {e}")

@app.post("/predict/mental")
def predict_mental(data:dict,user=Depends(current_user)):
    try:
        diet=int(mental_label_encoders["diet_quality"].transform([data["diet_quality"]])[0]); weather=int(mental_label_encoders["weather"].transform([data["weather"]])[0]); features=pd.DataFrame([{"sleep_hours":data["sleep_hours"],"screen_time":data["screen_time"],"exercise_minutes":data["exercise_minutes"],"daily_pending_tasks":data.get("daily_pending_tasks",5),"interruptions":data.get("interruptions",3),"fatigue_level":data["fatigue_level"],"social_hours":data["social_hours"],"coffee_cups":data["coffee_cups"],"diet_quality":diet,"weather":weather,"mood_score":data["mood_score"]}]); stress=round(float(mental_model.predict(mental_scaler.transform(features))[0]),2); category="Low Stress" if stress<=3 else "Moderate Stress" if stress<=6 else "High Stress"; risk=min(round(stress*10,2),100); tips=["Aim for a consistent 7–9 hour sleep routine.","Take screen breaks and avoid late-night scrolling.","Try breathing, mindfulness or relaxation exercises.","Keep regular movement and social connection.","If distress persists or interferes with daily life, speak with a mental-health professional."]; explanation=f"The model estimated a stress score of {stress}/10, categorized as {category}. This is a wellness screening estimate, not a diagnosis."; inputs={"Sleep Hours":data["sleep_hours"],"Screen Time":data["screen_time"],"Exercise Minutes":data["exercise_minutes"],"Fatigue Level":data["fatigue_level"],"Mood Score":data["mood_score"],"Social Hours":data["social_hours"],"Coffee Cups":data["coffee_cups"],"Diet Quality":data["diet_quality"],"Weather":data["weather"]}; pid,rid,created=record_prediction(user,"Mental Health",category,risk,inputs,tips,explanation); return {"Prediction":category,"StressLevel":stress,"StressPercentage":risk,"Tips":tips,"Explanation":explanation,"PredictionId":pid,"ReportId":rid,"CreatedAt":created}
    except Exception as e: raise HTTPException(400,f"Mental health prediction failed: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app",host="0.0.0.0",port=int(os.getenv("PORT",8000)),reload=True)
