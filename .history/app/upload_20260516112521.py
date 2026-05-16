from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

import os
import shutil
import uuid
import zipfile

from app.inspector import find_model_files, inspect_model, parse_mlmodel
from app.utils import load_model


router = APIRouter()

UPLOAD_DIR = "uploaded_models"
os.makedirs(UPLOAD_DIR, exist_ok=True)


def is_zip_file(filename: str) -> bool:
    return filename.endswith(".zip")


@router.post("/upload-model")
async def upload_model(file: UploadFile = File(...)):
    if not is_zip_file(file.filename):
        raise HTTPException(status_code=400, detail="Only .zip model uploads are supported")

    model_id = str(uuid.uuid4())
    model_upload_dir = os.path.join(UPLOAD_DIR, model_id)
    os.makedirs(model_upload_dir, exist_ok=True)

    zip_path = os.path.join(model_upload_dir, file.filename)

    with open(zip_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    extracted_path = os.path.join(model_upload_dir, "extracted")
    os.makedirs(extracted_path, exist_ok=True)

    try:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extracted_path)
    except zipfile.BadZipFile:
        raise HTTPException(status_code=400, detail="Invalid ZIP file")

    artifacts = find_model_files(extracted_path)

    if not (
        artifacts["mlmodel"]
        or artifacts["pickle_files"]
        or artifacts["joblib_files"]
        or artifacts["onnx_files"]
    ):
        raise HTTPException(
            status_code=400,
            detail="No supported model artifact was found (MLmodel, .pkl, .joblib, .onnx)",
        )

    metadata = parse_mlmodel(artifacts["mlmodel"])

    try:
        loaded = load_model(artifacts)
        model = loaded["model"]
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Model loading failed: {error}")

    try:
        inspection = inspect_model(model, metadata)
        inspection["loaded_model_format"] = loaded["model_format"]
        inspection["loaded_model_path"] = loaded["model_path"]
    except Exception as error:
        inspection = {"inspection_error": str(error)}

    response = {
        "status": "success",
        "message": "Model uploaded successfully",
        "model_id": model_id,
        "upload_location": model_upload_dir,
        "extracted_path": extracted_path,
        "detected_artifacts": {
            "MLmodel": artifacts["mlmodel"],
            "mlmodel_files": artifacts["mlmodel_files"],
            "pickle_files": artifacts["pickle_files"],
            "joblib_files": artifacts["joblib_files"],
            "onnx_files": artifacts["onnx_files"],
            "requirements_file": artifacts["requirements"],
        },
        "model_metadata": metadata,
        "model_inspection": inspection,
    }

    return JSONResponse(content=response)


@router.get("/models")
def list_uploaded_models():
    models = []

    for folder in os.listdir(UPLOAD_DIR):
        folder_path = os.path.join(UPLOAD_DIR, folder)
        if os.path.isdir(folder_path):
            models.append({"model_id": folder, "path": folder_path})

    return {"total_models": len(models), "models": models}


@router.get("/model/{model_id}")
def get_model_details(model_id: str):
    model_path = os.path.join(UPLOAD_DIR, model_id, "extracted")

    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model not found")

    return {
        "model_id": model_id,
        "artifacts": find_model_files(model_path),
    }
