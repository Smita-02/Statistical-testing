# app/upload.py

from fastapi import (
    APIRouter,
    File,
    HTTPException,
    UploadFile
)

from fastapi.responses import (
    JSONResponse
)

import os
import shutil
import uuid
import zipfile

from app.inspector import (
    find_model_files,
    inspect_model,
    parse_mlmodel
)

from app.utils import (
    load_model
)

# ---------------------------------------------------
# Router
# ---------------------------------------------------

router = APIRouter()

# ---------------------------------------------------
# Upload Directory
# ---------------------------------------------------

UPLOAD_DIR = "uploaded_models"

os.makedirs(
    UPLOAD_DIR,
    exist_ok=True
)

# ---------------------------------------------------
# Validate ZIP
# ---------------------------------------------------

def is_zip_file(
    filename: str
):

    return filename.lower().endswith(
        ".zip"
    )

# ---------------------------------------------------
# Upload Model Endpoint
# ---------------------------------------------------

@router.post("/upload-model")
async def upload_model(
    file: UploadFile = File(...)
):

    """
    Upload ML model ZIP,
    extract artifacts,
    inspect model,
    return metadata.
    """

    # ---------------------------------------------------
    # Validate File
    # ---------------------------------------------------

    if not is_zip_file(
        file.filename
    ):

        raise HTTPException(

            status_code=400,

            detail=(
                "Only .zip uploads "
                "are supported"
            )
        )

    # ---------------------------------------------------
    # Create Upload Directory
    # ---------------------------------------------------

    model_id = str(
        uuid.uuid4()
    )

    model_upload_dir = os.path.join(
        UPLOAD_DIR,
        model_id
    )

    os.makedirs(
        model_upload_dir,
        exist_ok=True
    )

    # ---------------------------------------------------
    # Save ZIP File
    # ---------------------------------------------------

    zip_path = os.path.join(
        model_upload_dir,
        file.filename
    )

    try:

        with open(
            zip_path,
            "wb"
        ) as buffer:

            shutil.copyfileobj(
                file.file,
                buffer
            )

    except Exception as error:

        raise HTTPException(

            status_code=500,

            detail=(
                f"Failed to save ZIP: "
                f"{str(error)}"
            )
        )

    # ---------------------------------------------------
    # Extract ZIP
    # ---------------------------------------------------

    extracted_path = os.path.join(
        model_upload_dir,
        "extracted"
    )

    os.makedirs(
        extracted_path,
        exist_ok=True
    )

    try:

        # Validate ZIP
        if not zipfile.is_zipfile(
            zip_path
        ):

            raise HTTPException(

                status_code=400,

                detail=(
                    "Uploaded file "
                    "is not a valid ZIP"
                )
            )

        # Extract ZIP
        with zipfile.ZipFile(
            zip_path,
            "r"
        ) as zip_ref:

            zip_ref.extractall(
                extracted_path
            )

    except zipfile.BadZipFile:

        raise HTTPException(

            status_code=400,

            detail="Corrupted ZIP file"
        )

    except Exception as error:

        raise HTTPException(

            status_code=500,

            detail=(
                f"ZIP extraction failed: "
                f"{str(error)}"
            )
        )

    # ---------------------------------------------------
    # Find Model Artifacts
    # ---------------------------------------------------

    artifacts = find_model_files(
        extracted_path
    )

    # ---------------------------------------------------
    # Validate Artifacts
    # ---------------------------------------------------

    if not (

        artifacts["mlmodel"]

        or len(
            artifacts["pickle_files"]
        ) > 0

        or len(
            artifacts["joblib_files"]
        ) > 0

        or len(
            artifacts["onnx_files"]
        ) > 0
    ):

        raise HTTPException(

            status_code=400,

            detail=(
                "No supported model "
                "artifact found"
            )
        )

    # ---------------------------------------------------
    # Parse MLmodel Metadata
    # ---------------------------------------------------

    metadata = parse_mlmodel(
        artifacts["mlmodel"]
    )

    # ---------------------------------------------------
    # Load Model
    # ---------------------------------------------------

    model = None

    loaded = None

    try:

        loaded = load_model(
            artifacts
        )

        model = loaded["model"]

    except Exception as error:

        raise HTTPException(

            status_code=500,

            detail=(
                f"Model loading failed: "
                f"{str(error)}"
            )
        )

    # ---------------------------------------------------
    # Inspect Model
    # ---------------------------------------------------

    inspection = {}

    try:

        if model is not None:

            inspection = inspect_model(
                model,
                metadata
            )

            inspection[
                "loaded_model_format"
            ] = loaded[
                "model_format"
            ]

            inspection[
                "loaded_model_path"
            ] = loaded[
                "model_path"
            ]

    except Exception as error:

        inspection = {

            "inspection_error":
                str(error)
        }

    # ---------------------------------------------------
    # Final Response
    # ---------------------------------------------------

    response = {

        "status": "success",

        "message":
            "Model uploaded successfully",

        "model_id":
            model_id,

        "upload_location":
            model_upload_dir,

        "extracted_path":
            extracted_path,

        "detected_artifacts": {

            "MLmodel":
                artifacts["mlmodel"],

            "mlmodel_files":
                artifacts[
                    "mlmodel_files"
                ],

            "pickle_files":
                artifacts[
                    "pickle_files"
                ],

            "joblib_files":
                artifacts[
                    "joblib_files"
                ],

            "onnx_files":
                artifacts[
                    "onnx_files"
                ],

            "requirements_file":
                artifacts[
                    "requirements"
                ]
        },

        "model_metadata":
            metadata,

        "model_inspection":
            inspection
    }

    return JSONResponse(
        content=response
    )

# ---------------------------------------------------
# List Uploaded Models
# ---------------------------------------------------

@router.get("/models")
def list_uploaded_models():

    """
    Returns uploaded models.
    """

    models = []

    for folder in os.listdir(
        UPLOAD_DIR
    ):

        folder_path = os.path.join(
            UPLOAD_DIR,
            folder
        )

        if os.path.isdir(
            folder_path
        ):

            models.append({

                "model_id":
                    folder,

                "path":
                    folder_path
            })

    return {

        "total_models":
            len(models),

        "models":
            models
    }

# ---------------------------------------------------
# Get Model Details
# ---------------------------------------------------

@router.get("/model/{model_id}")
def get_model_details(
    model_id: str
):

    """
    Returns model artifact details.
    """

    model_path = os.path.join(

        UPLOAD_DIR,

        model_id,

        "extracted"
    )

    if not os.path.exists(
        model_path
    ):

        raise HTTPException(

            status_code=404,

            detail="Model not found"
        )

    artifacts = find_model_files(
        model_path
    )

    return {

        "model_id":
            model_id,

        "artifacts":
            artifacts
    }
