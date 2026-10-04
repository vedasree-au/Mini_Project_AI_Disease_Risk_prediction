# AI Health Dashboard — Final Full-Stack Version

A healthcare screening dashboard using the existing Heart Disease, Diabetes and Mental Health ML models, with a real MySQL database, authentication, prediction history, PDF reports, BMI/symptom tools, and an interactive health assistant.

## 1. Important: MySQL Workbench vs MySQL Server

MySQL Workbench is the graphical tool used to manage MySQL. The project itself connects to **MySQL Server**. Make sure MySQL Server is installed and running on your computer. Workbench alone is not the database server.

## 2. Create the database

Open MySQL Workbench, connect to your local MySQL Server, and either:

- open `backend/schema.sql` and run it, or
- let the FastAPI application create the database/tables automatically when the configured MySQL account has permission to create databases.

The database name is:

`ai_health_dashboard`

Tables:

- `users` — account/profile data and password hashes
- `predictions` — every screening result linked to a user
- `reports` — PDF metadata and download counts linked to a user/prediction

## 3. Configure MySQL credentials

Copy:

`backend/.env.example` → `backend/.env`

Set your own values:

```text
DB_HOST=127.0.0.1
DB_PORT=3306
DB_USER=root
DB_PASSWORD=YOUR_MYSQL_PASSWORD
DB_NAME=ai_health_dashboard
HEALTH_APP_SECRET=use-a-long-random-secret
```

Do not commit `.env` to GitHub.

## 4. Install Python packages

From the `backend` folder:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## 5. Start the backend

```bash
uvicorn app:app --reload
```

Backend:

`http://127.0.0.1:8000`

Health check:

`http://127.0.0.1:8000/health`

Swagger API documentation:

`http://127.0.0.1:8000/docs`

The health endpoint should report `MySQL connected`.

## 6. Start the frontend

Open a second terminal:

```bash
cd frontend
python -m http.server 5500
```

Open:

`http://127.0.0.1:5500`

The frontend defaults to the local FastAPI server. You can override the API URL from the browser console with:

```js
localStorage.setItem('apiBaseUrl', 'https://YOUR-BACKEND.example.com')
```

## 7. How data is saved

Registration → `users` table.

Prediction → `predictions` table with the authenticated `user_id`.

PDF generation → PDF is stored in `backend/reports/`, while its metadata is stored in `reports`.

History → retrieved from MySQL for the currently authenticated user.

Reports → retrieved from MySQL and authorized by `user_id` before viewing/downloading.

## 8. PDF reports

Every saved prediction receives a report record and a generated PDF. The Reports page provides **View** and **PDF** actions. The backend has separate `/view` and `/download` endpoints and checks that the report belongs to the logged-in user.

For local college-demo use, generated PDFs are kept in `backend/reports/`. For production hosting, use persistent object/file storage because some hosting platforms have ephemeral filesystems.

## 9. Security notes

- Passwords are stored as salted PBKDF2-SHA256 hashes, not plaintext.
- Authentication uses signed expiring tokens.
- Every history/report query is restricted to the authenticated user's ID.
- Never put your MySQL password or `HEALTH_APP_SECRET` directly into frontend JavaScript.
- This application provides screening information and is not a medical diagnosis.

## 10. Existing ML functionality

The existing model files and visible prediction inputs are preserved. The backend continues to use the project's trained Heart Disease, Diabetes and Mental Health models.
