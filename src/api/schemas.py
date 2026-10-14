from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime, timezone

class AstroObservationInput(BaseModel):
    """
    Validation scheme for a single astronomical photometric observation.
    """

    # 1. Magnitudini fotometriche nelle 5 bande ottiche (SDSS)
    u: float = Field(..., ge=-10.0, le=60.0, description="Ultraviolet band magnitude (u)")
    g: float = Field(..., ge=-10.0, le=60.0, description="Green band magnitude (g)")
    r: float = Field(..., ge=-10.0, le=60.0, description="Red band (r) magnitude")
    i: float = Field(..., ge=-10.0, le=60.0, description="Near-infrared band magnitude (i)")
    z: float = Field(..., ge=-10.0, le=60.0, description="Infrared band magnitude (z)")

    # 2. Spostamento spettrale verso il rosso
    redshift: float = Field(..., ge=-0.1, le=15.0, description="Redshift spectrum (redshift z)")

    # Target Column
    target_class: str | None = Field(
        default=None,
        alias="class",
        description="Actual label of the celestial object, if known (e.g., GALAXY, STAR, QSO)",
    )

    # Metadati del catalogo e del telescopio (Opzionali con default)
    obj_ID: float = Field(default=0.0, description="Identificativo univoco dell'oggetto")
    alpha: float = Field(default=0.0, description="Ascensione retta (coordinate celesti)")
    delta: float = Field(default=0.0, description="Declinazione (coordinate celesti)")
    run_ID: int = Field(default=0, description="Numero di run della scansione fotometrica")
    rerun_ID: int = Field(default=0, description="Numero di re-run di rielaborazione")
    cam_col: int = Field(default=0, description="Colonna della camera ottica (1-6)")
    field_ID: int = Field(default=0, description="Identificativo del campo di osservazione")
    spec_obj_ID: float = Field(default=0.0, description="Identificativo univoco dello spettro ottico")
    plate: int = Field(default=0, description="Numero della piastra spettrografica")
    MJD: int = Field(default=0, description="Modified Julian Date (data di osservazione)")
    fiber_ID: int = Field(default=0, description="Identificativo della fibra ottica")

    model_config = ConfigDict(
        populate_by_name=True,
        extra="ignore",
        json_schema_extra={
            "example": {
                "u": 22.67751,
                "g": 24.06153,
                "r": 22.13861,
                "i": 20.92139,
                "z": 20.1441,
                "redshift": 0.5875528,
            }
        },
    )

class AstroObservationBatchInput(BaseModel):
    batch_list: list[AstroObservationInput] = Field(
        ..., 
        min_length=1,
        max_length=1000, 
        description="List of astronomical observations to be classified (from 1 to 1,000 items)"
    )

class PredictionOutput(BaseModel):
    proba_prediction: float = Field(..., le=1.0, ge=0.0, description="Probability associated with the predicted class for the provided celestial object.")
    prediction: str = Field(..., description="Predicted class for the provided celestial object (e.g., GALAXY, STAR, QSO).")
    prediction_timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp UTC della predizione."
    )
    actual_class: str | None = Field(
        default=None,
        description="Actual class of the celestial object, if known (e.g., GALAXY, STAR, QSO).",
    )
    model_name: str = Field(default="Classification_Astro_Model", description="Name of the model used for prediction.")
    model_version: str = Field(..., description="Version of the model used for the prediction.")

class BatchPredictionOutput(BaseModel):
    total_processed: int = Field(..., description="Numero totale di osservazioni elaborate.")
    predictions: list[PredictionOutput] = Field(
        ..., 
        min_length=1,
        max_length=1000, 
        description="Lista di predizioni per le osservazioni astronomiche fornite (da 1 a 1000 elementi)"
    )

class HealthOutput(BaseModel):
    status: str = Field(..., description="Stato di salute dell'API (es. 'healthy', 'unhealthy').")
    model_loaded: bool = Field(..., description="Indica se il modello ML è caricato in memoria ed è pronto all'inferenza.")
    model_name: str = Field(..., description="Nome del modello registrato.")
    model_version: str = Field(..., description="Versione del modello in uso.")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp UTC del controllo di salute."
    )