from contextlib import asynccontextmanager
import logging

import io
import os
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile, status
from fastapi.responses import StreamingResponse
import pandas as pd

from src.api.dependencies import get_predictor
from src.api.schemas import (
    AstroObservationBatchInput,
    AstroObservationInput,
    BatchPredictionOutput,
    HealthOutput,
    PredictionOutput,
)
from src.predict.predictor import AstroPredict
from src.utils.logger import restore_logging_after_mlflow

logger = logging.getLogger(__name__)


# =====================================================================
# 1. LIFESPAN 
# =====================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages the startup and shutdown of the API. 
    Loads the ML model into memory once at startup.
    """
    logger.info("API startup: loading the AstroPredict model into memory...")
    model_name = os.getenv("MODEL_NAME", "Classification_Astro_Model")
    # Instantiate the model and save it to the app's global state
    app.state.predictor = AstroPredict(model_name=model_name, verbose=False)
    restore_logging_after_mlflow()
    logger.info("Model successfully loaded. Server ready for predictions!")
    
    yield 
    
    logger.info("API shutdown: releasing resources...")
    app.state.predictor = None


# =====================================================================
# 2. FASTAPI APPLICATION INITIALIZATION
# =====================================================================
app = FastAPI(
    title="Astronomical Object Classification API",
    description=(
        "Production-grade RESTful API for the photometric classification of celestial objects "
        "(Galaxies, Stars, Quasars) based on SDSS observations."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# =====================================================================
# 3. ENDPOINT: GET /health (Health Check-up)
# =====================================================================
@app.get(
    "/health",
    response_model=HealthOutput,
    tags=["Health"],
    summary="Service and model health check",
    description="Verify that the API is operational and that the MLflow model is correctly loaded into memory.",
)
def health_check(predictor: AstroPredict = Depends(get_predictor)) -> HealthOutput:
    model_loaded = predictor is not None
    model_name = getattr(predictor, "model_name", "Classification_Astro_Model")
    model_version = getattr(predictor, "model_version", "latest")

    return HealthOutput(
        status="healthy" if model_loaded else "unhealthy",
        model_loaded=model_loaded,
        model_name=model_name,
        model_version=str(model_version),
    )


# =====================================================================
# 4. ENDPOINT:POST /predict (Single Prediction)
# =====================================================================
@app.post(
    "/predict/single_record",
    response_model=PredictionOutput,
    tags=["Predictions"],
    summary="Classification of a single astronomical observation",
    description="It predicts the celestial class (GALAXY, STAR, QSO) and the associated confidence for a single observation.",
)
def predict_single(
    observation: AstroObservationInput,
    predictor: AstroPredict = Depends(get_predictor),
) -> PredictionOutput:
    try:
        input_df = pd.DataFrame([observation.model_dump(by_alias=True)])

        result_df = predictor.predict(input_df)

        prediction_class = str(result_df["prediction"].iloc[0])
        proba = float(result_df["proba_prediction"].iloc[0])
        model_name = getattr(predictor, "model_name", "Classification_Astro_Model")
        model_version = str(getattr(predictor, "model_version", "latest"))

        return PredictionOutput(
            prediction=prediction_class,
            proba_prediction=proba,
            actual_class=observation.target_class,
            model_name=model_name,
            model_version=model_version,
        )

    except Exception as e:
        logger.error(f"Error during single inference: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error during prediction: {str(e)}",
        ) from e


# =====================================================================
# 5. ENDPOINT: POST /predict/batch (Batch / Vectorized Prediction)
# =====================================================================
@app.post(
    "/predict/multi_record",
    response_model=BatchPredictionOutput,
    tags=["Predictions"],
    summary="Vectorized classification of a batch of observations",
    description="Process a list of observations (from 1 to 1,000 elements) in a single, efficient matrix operation.",
)
def predict_batch(
    batch: AstroObservationBatchInput,
    predictor: AstroPredict = Depends(get_predictor),
) -> BatchPredictionOutput:
    try:
        batch_data = [obs.model_dump(by_alias=True) for obs in batch.batch_list]
        input_df = pd.DataFrame(batch_data)

        result_df = predictor.predict(input_df)

        model_version = str(getattr(predictor, "model_version", "latest"))
        model_name = getattr(predictor, "model_name", "Classification_Astro_Model")
        predictions: list[PredictionOutput] = []

        for idx, obs in enumerate(batch.batch_list):
            predictions.append(
                PredictionOutput(
                    prediction=str(result_df["prediction"].iloc[idx]),
                    proba_prediction=float(result_df["proba_prediction"].iloc[idx]),
                    actual_class=obs.target_class,
                    model_name=model_name,
                    model_version=model_version,
                )
            )

        return BatchPredictionOutput(
            total_processed=len(predictions),
            predictions=predictions,
        )

    except Exception as e:
        logger.error(f"Error during batch inference: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error during batch prediction: {str(e)}",
        ) from e

# =====================================================================
# 6. ENDPOINT: POST /predict/csv (CSV File Upload and Download)
# =====================================================================
@app.post(
    "/predict/csv",
    tags=["Predictions"],
    summary="Prediction from CSV file with result download",
    description="It allows you to upload a CSV file of astronomical observations and returns " \
        "the CSV file enriched with predictions.",
)
async def predict_csv(
    file: UploadFile = File(..., description="CSV file containing astronomical observations."),
    predictor: AstroPredict = Depends(get_predictor),
) -> StreamingResponse:
    # 1. Controllo difensivo sull'estensione del file
    if not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file format: only files with the .csv extension may be uploaded.",
        )

    try:
        input_df = pd.read_csv(file.file)

        result_df = predictor.predict(input_df)

        stream = io.StringIO()
        result_df.to_csv(stream, index=False)
        stream.seek(0)

        base_name = file.filename.rsplit(".", 1)[0]
        output_filename = f"{base_name}_predicted.csv"

        return StreamingResponse(
            iter([stream.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="{output_filename}"'},
        )

    except Exception as e:
        logger.error(f"Error processing the CSV file: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error while processing the file: {str(e)}",
        ) from e
    finally:
        await file.close()