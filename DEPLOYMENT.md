# 🚀 Deployment Guide - Pneumonia Detection API

This guide covers deploying the Pneumonia Detection API to **Azure** and **Koyeb**.

---

## 📋 Prerequisites

- Docker installed locally
- Your trained model in `results/models/model_final.keras`
- Account on [Azure](https://azure.microsoft.com) or [Koyeb](https://koyeb.com)

---

## 🐳 Building the Docker Image

### Build locally:
```bash
# Build the image
docker build -t pneumonia-detection-api:latest .

# Test locally
docker run -p 8000:8000 pneumonia-detection-api:latest

# Test the API
curl http://localhost:8000/health
```

### Using Docker Compose:
```bash
# Build and run
docker compose up --build

# Run in background
docker compose up -d --build
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
