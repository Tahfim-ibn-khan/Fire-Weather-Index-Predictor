from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import numpy as np
import pickle
# Load scaler and model
scaler = pickle.load(open('../models/scaler.pkl', 'rb'))
model = pickle.load(open('../models/ridge.pkl', 'rb'))

# Initialize app and templates
app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Input fields (excluding DC, BUI)
FIELDS = ['Temperature', 'RH', 'Ws', 'Rain', 'FFMC', 'DMC', 'ISI', 'Classes', 'Region']

@app.get("/", response_class=HTMLResponse)
def form_page(request: Request):
    return templates.TemplateResponse("form.html", {"request": request, "fields": FIELDS, "result": None})

@app.post("/predict", response_class=HTMLResponse)
async def predict_fwi(request: Request,
                      Temperature: float = Form(...), RH: float = Form(...), Ws: float = Form(...),
                      Rain: float = Form(...), FFMC: float = Form(...), DMC: float = Form(...),
                      ISI: float = Form(...), Classes: int = Form(...), Region: int = Form(...)):

    # Prepare input
    input_data = np.array([[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]])
    scaled_input = scaler.transform(input_data)
    prediction = model.predict(scaled_input)[0]

    return templates.TemplateResponse("form.html", {
        "request": request,
        "fields": FIELDS,
        "result": round(float(prediction), 3)
    })