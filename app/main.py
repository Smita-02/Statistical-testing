# app/main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.upload import router as upload_router
from app.governance import router as governance_router

# FastAPI App Initialization

app = FastAPI(
    title="AI Governance System",
    description="AI Governance platform for ML model inspection, fairness analysis, and deterministic metric evaluation.",
    version="1.0.0"
)

# CORS Configuration
# Allows frontend apps (React/Vue/etc.) to access API

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API Routers

app.include_router(
    upload_router,
    prefix="/api",
    tags=["Model Upload"]
)

app.include_router(
    governance_router,
    prefix="/api",
    tags=["Governance Engine"]
)

# Root Health Check Endpoint

@app.get("/")
def home():
    return {
        "message": "AI Governance System Running",
        "status": "healthy",
        "version": "1.0.0"
    }

# Readiness Endpoint
# Useful for Kubernetes / Docker health checks

@app.get("/health")
def health_check():
    return {
        "service": "AI Governance System",
        "status": "UP"
    }