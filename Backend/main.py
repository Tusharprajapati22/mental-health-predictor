from typing import Literal
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
from fastapi.responses import FileResponse  # FileResponse import karein

from fastapi.middleware.cors import CORSMiddleware  # 1. Import karein







# Load model
Model = joblib.load('Mental_health_Predicator.pkl')
top_countries = [
    'Other',
    'India',
    'USA',
    'Canada',
    'Australia',
    'UK',
    'Germany',
    'Mexico',
    'Turkey',
    'France',
]

app = FastAPI()

#  2. CORS Middleware Add Karein
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Sabhi connections allow karne ke liye
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class StudentData(BaseModel):
  Age: int = Field(..., gt=10, le=100)
  Gender: Literal['Male', 'Female']
  Country: str
  Academic_Level: Literal['Undergraduate', 'Graduate', 'High School']
  Most_Used_Platform: Literal[
      'Facebook',
      'LinkedIn',
      'Instagram',
      'Snapchat',
      'Twitter',
      'Youtube',
      'TikTok',
      'LINE',
      'KakaoTalk',
      'VKontakte',
      'WhatsApp',
      'WeChat',
  ]
  Purpose_Of_Use: Literal['Networking', 'Education', 'Entertainment', 'News']
  Avg_Daily_Usage_Hours: float = Field(..., ge=0, le=24)
  Daily_Unlocks: int = Field(..., ge=0)
  Study_Hours: float = Field(..., ge=0, le=24)
  Physical_Activity_Hours: float = Field(..., ge=0, le=24)
  Sleep_Hours_Per_Night: float = Field(..., ge=0, le=24)
  Stress_Level: Literal['Medium', 'Low', 'Very High', 'High']


class PredictionResponse(BaseModel):
  predicted_mental_health_score: float


@app.get('/')
def serve_ui():
    return FileResponse('frontend/index.html')

# /Predict endpoint baki bilkul same rahega...


@app.post('/Predict', response_model=PredictionResponse)
def predict(data: StudentData):
  country_group = data.Country if data.Country in top_countries else 'Other'

  input_row = pd.DataFrame([{
      'Age': data.Age,
      'Gender': data.Gender,
      'Country': data.Country, 
      'Grouped_country': country_group, # Using grouped country for model consistency
      'Academic_Level': data.Academic_Level,
      'Most_Used_Platform': data.Most_Used_Platform,
      'Purpose_Of_Use': data.Purpose_Of_Use,
      'Avg_Daily_Usage_Hours': data.Avg_Daily_Usage_Hours,
      'Daily_Unlocks': data.Daily_Unlocks,
      'Study_Hours': data.Study_Hours,
      'Physical_Activity_Hours': data.Physical_Activity_Hours,
      'Sleep_Hours_Per_Night': data.Sleep_Hours_Per_Night,
      'Stress_Level': data.Stress_Level,
  }])

  prediction = Model.predict(input_row)[0]

  return PredictionResponse(
      predicted_mental_health_score=float(round(prediction, 2))
  )