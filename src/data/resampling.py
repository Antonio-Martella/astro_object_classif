from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

from configs.random_seed_loader import load_random_seed_config


class ResamplerFactory:
    _registry = {
        "smote": SMOTE,
        "undersampling": RandomUnderSampler,
        "class_weight": None,  # <-- Metto 'None' poiché i modelli di default hanno class_weight = 'balanced'
    }

    @classmethod
    def get_resampler(cls, strategy_name: str, random_state: int | None = None):
        strategy = strategy_name

        if strategy not in cls._registry:
            raise ValueError(f"Strategia applicata per il resempling '{strategy}' non valida!")

        resampler_class = cls._registry[strategy]

        if resampler_class is None:
            return None

        if random_state is None:
            random_state = load_random_seed_config().random_seed_resempling

        return resampler_class(random_state=random_state)
