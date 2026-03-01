# Deployment Guide: Thermal Anomaly Detection

This project uses a hybrid architecture:
- **Frontend**: Next.js (Deployed to Vercel)
- **Backend**: Flask + PyTorch (Deployed to Render or Railway)

---

## Step 1: Deploy the Backend (Flask API)

The backend needs a persistent Python environment with ~2GB RAM (due to PyTorch).

### Option A: Render (Recommended)
1.  Sign in to [Render](https://render.com).
2.  Click **New +** > **Web Service**.
3.  Connect your GitHub repository: `Zorrojurro/Thermal-Anomaly-Detection-Engine`.
4.  Configure the service:
    - **Name**: `thermal-backend`
    - **Environment**: `Python 3`
    - **Build Command**: `pip install -r requirements.txt`
    - **Start Command**: `python web_app.py`
    - **Plan**: Starter (or any plan with ≥ 2GB RAM).
5.  Wait for deployment. Once live, copy your **service URL** (e.g., `https://thermal-backend.onrender.com`).

---

## Step 2: Deploy the Frontend (Next.js)

1.  Sign in to [Vercel](https://vercel.com).
2.  Click **Add New** > **Project**.
3.  Import `Zorrojurro/Thermal-Anomaly-Detection-Engine`.
4.  In the **Configure Project** screen:
    - **Root Directory**: Select `frontend`.
    - **Framework Preset**: Next.js.
5.  Click **Deploy**.
6.  Once deployed, copy your **Vercel URL** (e.g., `https://thermal-frontend.vercel.app`).

---

## Step 3: Connect Frontend to Backend

To make the "Analyze" button work, the frontend needs to know where the backend is.

1.  Go to your project in the **Vercel Dashboard**.
2.  Go to **Settings** > **Environment Variables**.
3.  Add a new variable:
    - **Key**: `NEXT_PUBLIC_API_URL`
    - **Value**: Your Render Backend URL (from Step 1).
4.  **Redeploy** the frontend for the changes to take effect.

---

## Step 4: Final Verification

1.  Open your Vercel URL.
2.  The Warp shader background should appear.
3.  Click a **Sample Image** chip.
4.  The "Analyzing..." spinner should appear, followed by the results from your Render backend.

---

## Troubleshooting

- **CORS Error**: The backend is already configured to allow all origins via `flask-cors`. If you see errors, ensure `CORS(app)` is present in `web_app.py`.
- **Memory Limit**: If the backend crashes during "Loading model...", your hosting plan's RAM is too low. PyTorch + ResNet-18 requires at least 1.5GB to be safe.
- **Port**: The backend automatically uses the `$PORT` environment variable provided by Render/Railway.
