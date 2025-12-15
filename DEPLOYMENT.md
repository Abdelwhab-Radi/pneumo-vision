# 🚀 Deployment Guide - Pneumonia Detection API

This guide covers deploying the Pneumonia Detection API to **Streamlit Cloud**, **Render**, **Azure**, and **Koyeb**.

---

## 📋 Prerequisites

- Docker installed locally
- Your trained model in `results/models/model_final.keras`
- Account on your chosen platform

---

## 🎈 Deploy to Streamlit Cloud (Easiest!)

Streamlit Cloud is the **simplest option** - no Docker needed, just push to GitHub!

### Step 1: Prepare Your Repository

Your repo should have these files:
```
├── streamlit_app.py          # ✅ Created
├── requirements_streamlit.txt # ✅ Created (rename to requirements.txt)
├── .streamlit/
│   └── config.toml           # ✅ Created
└── results/
    └── models/
        └── model_final.keras  # Your trained model
```

### Step 2: Rename Requirements File
```bash
# For Streamlit Cloud, use the streamlit requirements
cp requirements_streamlit.txt requirements.txt
```

### Step 3: Push to GitHub
```bash
git add .
git commit -m "Add Streamlit app"
git push origin main
```

### Step 4: Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click **"New app"**
3. Connect your GitHub repository
4. Configure:
   | Setting | Value |
   |---------|-------|
   | **Repository** | your-username/your-repo |
   | **Branch** | main |
   | **Main file** | streamlit_app.py |
5. Click **"Deploy!"**

### Step 5: Wait for Deployment
- First build takes **5-10 minutes** (TensorFlow installation)
- Subsequent deploys are faster (cached dependencies)

### Your App URL
```
https://your-app-name.streamlit.app
```

### ⚠️ Streamlit Cloud Limits

| Limit | Value |
|-------|-------|
| **RAM** | 1GB (may be tight for TensorFlow) |
| **Storage** | 1GB for app + model |
| **Compute** | Shared CPU |
| **Sleep** | After 7 days inactivity (free tier) |

### 💡 Tips for Streamlit Cloud

1. **Use `tensorflow-cpu`** instead of `tensorflow` (smaller)
2. **Keep model under 100MB** if possible (use LFS for larger)
3. **Add secrets** in Streamlit Cloud dashboard (not in code)

### Run Locally First
```bash
# Test before deploying
pip install -r requirements_streamlit.txt
streamlit run streamlit_app.py

# Opens at http://localhost:8501
```

---

## ☁️ Deploy to Koyeb

### Option 1: Via Koyeb CLI
```bash
# Install Koyeb CLI
npm install -g koyeb-cli

# Login
koyeb login

# Deploy using the config file
koyeb deploy --config koyeb.yaml
```

### Option 2: Via Koyeb Dashboard (Recommended)
1. Go to [Koyeb Dashboard](https://app.koyeb.com)
2. Click **Create App** → **Docker**
3. Connect your GitHub repository OR use a container registry
4. Set the following:
   - **Port**: `8000`
   - **Health Check Path**: `/health`
   - **Instance Type**: `small` (minimum for TensorFlow)
5. Add environment variables:
   ```
   HOST=0.0.0.0
   PORT=8000
   LOG_LEVEL=INFO
   ALLOWED_ORIGINS=*
   ```
6. Deploy!

### Option 3: Deploy with Docker Registry
```bash
# Push to Docker Hub or any registry
docker tag pneumonia-detection-api:latest your-registry/pneumonia-api:latest
docker push your-registry/pneumonia-api:latest

# Then use the pushed image in Koyeb
```

---

## 🟣 Deploy to Render

### Option 1: Via Dashboard (Recommended)

1. **Push your code to GitHub**
   ```bash
   git add .
   git commit -m "Add deployment configuration"
   git push origin main
   ```

2. **Create a new Web Service on Render**
   - Go to [Render Dashboard](https://dashboard.render.com)
   - Click **New** → **Web Service**
   - Connect your GitHub repository
   - Render will auto-detect the `render.yaml` file

3. **Configure the service** (if not using render.yaml):
   | Setting | Value |
   |---------|-------|
   | **Environment** | Docker |
   | **Plan** | Starter ($7/mo) or higher |
   | **Health Check Path** | `/health` |
   | **Port** | `8000` |

4. **Add Environment Variables**:
   ```
   HOST=0.0.0.0
   PORT=8000
   LOG_LEVEL=INFO
   ALLOWED_ORIGINS=*
   ```

5. **Deploy!** - Render will build and deploy automatically

### Option 2: Manual Docker Deploy

```bash
# Build locally
docker build -t pneumonia-api .

# Push to Docker Hub
docker tag pneumonia-api:latest yourusername/pneumonia-api:latest
docker push yourusername/pneumonia-api:latest

# Then in Render:
# 1. New → Web Service
# 2. Select "Deploy an existing image from a registry"
# 3. Enter: docker.io/yourusername/pneumonia-api:latest
```

### Render-Specific Notes

| ⚠️ Important | Details |
|--------------|---------|
| **Free Tier** | Not recommended - 512MB RAM is insufficient for TensorFlow |
| **Starter Plan** | Minimum recommended ($7/month, 512MB-2GB RAM) |
| **Cold Starts** | Starter plan may sleep after inactivity, causing slow first request |
| **Build Time** | First build takes 10-15 minutes (TensorFlow compilation) |
| **Disk Space** | Ensure your model file is under 500MB for smoother builds |

### Accessing Your API on Render

Once deployed, your API will be available at:
```
https://pneumonia-detection-api.onrender.com
```

Test it:
```bash
# Health check
curl https://your-app.onrender.com/health

# API docs
open https://your-app.onrender.com/docs

# Make prediction
curl -X POST https://your-app.onrender.com/predict \
  -F "file=@chest_xray.jpg"
```

---

## 🔷 Deploy to Azure

### Option 1: Azure Container Instances (Quick)

```bash
# Login to Azure
az login

# Create a resource group
az group create --name pneumonia-rg --location eastus

# Create Azure Container Registry (ACR)
az acr create --resource-group pneumonia-rg --name pneumoniaacr --sku Basic

# Login to ACR
az acr login --name pneumoniaacr

# Tag and push image
docker tag pneumonia-detection-api:latest pneumoniaacr.azurecr.io/pneumonia-api:latest
docker push pneumoniaacr.azurecr.io/pneumonia-api:latest

# Deploy using ARM template
az deployment group create \
  --resource-group pneumonia-rg \
  --template-file azure-container.json \
  --parameters imageRegistry=pneumoniaacr.azurecr.io imageName=pneumonia-api:latest
```

### Option 2: Azure App Service

```bash
# Create App Service plan
az appservice plan create \
  --name pneumonia-plan \
  --resource-group pneumonia-rg \
  --is-linux \
  --sku B2

# Create web app from Docker image
az webapp create \
  --resource-group pneumonia-rg \
  --plan pneumonia-plan \
  --name pneumonia-detection-api \
  --deployment-container-image-name pneumoniaacr.azurecr.io/pneumonia-api:latest

# Configure environment variables
az webapp config appsettings set \
  --resource-group pneumonia-rg \
  --name pneumonia-detection-api \
  --settings \
    WEBSITES_PORT=8000 \
    LOG_LEVEL=INFO \
    ALLOWED_ORIGINS=*
```

### Option 3: Azure Container Apps (Serverless)

```bash
# Create Container Apps environment
az containerapp env create \
  --name pneumonia-env \
  --resource-group pneumonia-rg \
  --location eastus

# Deploy the container app
az containerapp create \
  --name pneumonia-api \
  --resource-group pneumonia-rg \
  --environment pneumonia-env \
  --image pneumoniaacr.azurecr.io/pneumonia-api:latest \
  --target-port 8000 \
  --ingress external \
  --cpu 2 \
  --memory 4.0Gi \
  --min-replicas 1 \
  --max-replicas 3
```

---

## 🔧 Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `HOST` | `0.0.0.0` | Server host |
| `PORT` | `8000` | Server port |
| `LOG_LEVEL` | `INFO` | Logging level |
| `ALLOWED_ORIGINS` | `*` | CORS origins (comma-separated) |
| `MODEL_PATH` | `/app/results/models/model_final.keras` | Path to model |
| `CONFIG_PATH` | `/app/results/training_config.json` | Path to config |

---

## 📡 API Endpoints

Once deployed, your API provides these endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API information |
| `/health` | GET | Health check |
| `/predict` | POST | Single image prediction |
| `/predict/batch` | POST | Batch prediction (max 10) |
| `/model/info` | GET | Model information |
| `/docs` | GET | Swagger UI documentation |
| `/redoc` | GET | ReDoc documentation |

### Test your deployment:
```bash
# Health check
curl https://your-app-url/health

# Make a prediction
curl -X POST https://your-app-url/predict \
  -F "file=@chest_xray.jpg"
```

---

## 🎨 Frontend Access

The frontend is included in the Docker image at `/app/frontend/`. 

To serve it:
1. The API serves the frontend files automatically
2. Access at `https://your-app-url/` (root path)
3. Or configure a CDN/static hosting for the `/frontend` folder

---

## ⚠️ Important Notes

1. **Memory**: TensorFlow models require at least **2GB RAM**. Use `small` instances or higher.
2. **Startup Time**: Model loading takes 30-60 seconds. Set health check grace period accordingly.
3. **CPU-Only**: The Dockerfile uses CPU inference for cost efficiency. GPU instances are optional.
4. **CORS**: Update `ALLOWED_ORIGINS` for production to restrict to your frontend domain.

---

## 🔒 Security Recommendations

1. Set specific `ALLOWED_ORIGINS` instead of `*`
2. Use HTTPS (both platforms provide it automatically)
3. Consider adding API key authentication for production
4. Monitor logs for unusual activity

---

## 📊 Monitoring

- **Koyeb**: Built-in metrics and logs in dashboard
- **Azure**: Use Application Insights or Container Insights
- API provides `/health` endpoint for monitoring tools
