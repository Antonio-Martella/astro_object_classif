from fastapi import Request
from src.predict.predictor import AstroPredict

def get_predictor(request: Request) -> AstroPredict:
    """Recupera l'istanza di AstroPredict salvata nello stato dell'applicazione."""
    return request.app.state.predictor