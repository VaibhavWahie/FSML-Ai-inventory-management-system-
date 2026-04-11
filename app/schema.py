from pydantic import BaseModel

class InputData(BaseModel):
    id: int
    store_nbr: int
    family: int
    onpromotion: int
    day_of_week: int
    month: int
    week: int
    is_weekend: int
    lag_1: float
    lag_2: float
    lag_3: float
    lag_7: float
    lag_14: float
    lag_21: float
    lag_28: float
    rolling_mean_7: float
    rolling_mean_14: float
    rolling_std_7: float