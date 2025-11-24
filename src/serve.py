
import os
from typing import List 
from typing import Optional, Union
from datetime import datetime

from fastapi import FastAPI, HTTPException, Request, Form
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse

import torch

import model as libInsulin


app = FastAPI()

port: int = 7777
origins = [
    f"http://localhost:{port}",
    f"localhost:{port}",
    f"0.0.0.0:{port}",
    
    # frontend 
    f"http://192.168.1.149:{port}",
    f"192.168.1.149:{port}",
]

app.add_middleware(GZipMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

models: list[tuple[libInsulin.InsulinModule, str]] = []

def getInsulinPrediction(model: libInsulin.InsulinModule, time: datetime, carbs:float, bloodSugar:float, targetBloodSugar: float) -> float:

    tensors = libInsulin.dataToTensorMap(time, carbs, bloodSugar, targetBloodSugar)

    data = { key: value.to(model.device) for key, value in tensors.items() }

    with torch.no_grad():

        prediction: torch.Tensor = model(data)

        print(prediction)

        return prediction.detach().cpu().item()

@app.post("/submit", response_class=HTMLResponse)
async def submit_form(
    meal_datetime: str = Form(...),
    blood_sugar: float = Form(...),
    net_carbs: float = Form(...),
    target_blood_sugar: float = Form(...),
):

    parsed_datetime = datetime.fromisoformat(meal_datetime)

    results: list[str] = []

    for (model, name) in models:

        guess : float = getInsulinPrediction(model, parsed_datetime, net_carbs, blood_sugar, target_blood_sugar)
        results.append(f"<tr><td>{name}</td><td>{guess}</td></tr>")


    result = "\n".join(results)

    return f"""
    <html>
        <body>
            <h2>Submitted Data</h2>
            <table border="1" cellpadding="6" cellspacing="0">
                <thead>
                    <tr>
                        <th>Name</th>
                        <th>Guess</th>
                    </tr>
                </thead>
                <tbody>
                    {result}
                </tbody>
            </table>
        </body>
    </html>
    """

@app.get("/", response_class=HTMLResponse)
async def home_papge(request : Request):
    return """
    <html>
        <head>
            <title>Health Input Form</title>
        </head>
        <body>
            <h1>Enter Health Data</h1>
            <form id="insulinForm" method="post" action="/submit">
                
                <label for="meal_datetime">Date & Time:</label>
                <input type="datetime-local" id="meal_datetime" name="meal_datetime" required>
                <br><br>
                
                <label for="blood_sugar">Blood Sugar (mg/dL):</label>
                <input type="number" id="blood_sugar" name="blood_sugar" step="0.1" required>
                <br><br>
                
                <label for="net_carbs">Net Carbs (g):</label>
                <input type="number" id="net_carbs" name="net_carbs" step="0.1" required>
                <br><br>

                <label for="target_blood_sugar">Target Blood Sugar:</label>
                <input type="number" id="target_blood_sugar" name="target_blood_sugar" value="6.7" required>
                <br><br>
                
                <button type="submit">Submit</button>
            </form>
            <div id="responseContainer"></div>

            <script>
            document.getElementById("insulinForm").addEventListener("submit", async function(event) {
                event.preventDefault(); // Prevent default form submission

                const form = event.target;
                const formData = new FormData(form);

                try {
                    const response = await fetch(form.action, {
                        method: form.method,
                        body: formData
                    });

                    if (!response.ok) {
                        throw new Error("Network response was not ok");
                    }

                    const html = await response.text();

                    // Insert response HTML into the div
                    document.getElementById("responseContainer").innerHTML = html;

                } catch (error) {
                    console.error("Error submitting form:", error);
                    document.getElementById("responseContainer").innerHTML = `<p style="color:red">Error submitting form: ${error.message}</p>`;
                }
            });
            </script>
        </body>
    </html>
    """

if __name__ == "__main__":

    import uvicorn

    toLoad = [
        ("./models/model_insulin_v0.ckpt", "v0", libInsulin.InsulinModule),
        ("./models/model_insulin_v1.ckpt", "v1", libInsulin.InsulinModule),
        ("./models/model_insulin_v2.ckpt", "v2", libInsulin.InsulinModule2),
    ]

    for (path, name, modelInst) in toLoad:

        print(f"loading model: {path} {name}")

        uma = modelInst.load_from_checkpoint(path)
        uma.eval()

        models.append((uma, name))

    uvicorn.run(app, host="0.0.0.0", port=port)


