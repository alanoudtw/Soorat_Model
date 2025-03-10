from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.encoders import jsonable_encoder
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from io import BytesIO
import joblib
import logging
import os

app = FastAPI()

logger = logging.getLogger('uvicorn.error')
logger.setLevel(logging.DEBUG)


def calories_from_macro(protein, carbs, fat):
    return protein * 4 + carbs * 4 + fat * 9


def direct_prediction(img):
    
    img = tf.keras.utils.load_img(img)
    img = img_to_array(img)
    img = tf.image.resize(img, (320, 320))
    img = tf.expand_dims(img, axis=0)  # Add batch dimension
    
    # Load the model
    model = tf.keras.models.load_model('models/direct_regression.h5')
    # model = tf.keras.models.model_from_json(open('models/direct_regression.json').read())

    # Load the scalers
    mass_scaler = joblib.load('./scalers/mass_scaler.save')
    fat_scaler = joblib.load('./scalers/fat_scaler.save')
    carb_scaler = joblib.load('./scalers/carb_scaler.save')
    protein_scaler = joblib.load('./scalers/protein_scaler.save')


    predictions = model.predict(img, verbose=0)
    protein = protein_scaler.inverse_transform(predictions['protein'][0][0].reshape(1, -1))
    fat = fat_scaler.inverse_transform(predictions['fat'][0][0].reshape(1, -1))
    carbs = carb_scaler.inverse_transform(predictions['carbs'][0][0].reshape(1, -1))
    mass = mass_scaler.inverse_transform(predictions['mass'][0][0].reshape(1, -1))
    protein_val = protein[0][0]
    fat_val = fat[0][0]
    carbs_val = carbs[0][0]
    mass_val = mass[0][0]
    calories = calories_from_macro(
        protein=protein_val,
        carbs=carbs_val,
        fat=fat_val,
    )
    results = {
        # 'predictions': predictions,
        'protein': protein_val.item(),
        'fat': fat_val.item(),
        'carbs': carbs_val.item(),
        'calories': calories.item(),
        'mass': mass_val.item(),
    }
    logger.info(results)
    # log the types for all results items
    logger.info({k: type(v) for k, v in results.items()})
    return results


def independent_prediction(img, total_mass):
    model = tf.keras.models.load_model('models/portion_independent.h5')
    predictions = model.predict(img, verbose=0)
    protein = predictions['protein'][0][0] * total_mass
    fat = predictions['fat'][0][0] * total_mass
    carbs = predictions['carbs'][0][0] * total_mass
    calories = calories_from_macro(
        protein=protein,
        carbs=carbs,
        fat=fat,
    )
    results = {
        # 'predictions': predictions,
        'protein': protein,
        'fat': fat,
        'carbs': carbs,
        'calories': calories,
        'mass': total_mass,
    }
    logger.info(results)
    return results

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/models/independent")
async def independent(file: UploadFile = File(...), total_mass: float = Form(...)):
    try:
        img = BytesIO(await file.read())
        result = independent_prediction(img, total_mass)
        return result
    except Exception as e:
        logger.error(str(e))
        return {"Error"}
        

@app.post("/models/direct")
async def direct(file: UploadFile = File(...)):
    try:
        img = BytesIO(await file.read())
        logger.info("Image received!!")
        result = direct_prediction(img)
        return jsonable_encoder(result)
    except Exception as e:
        return {"error": str(e)}
    

@app.post("/models/direct-demo")
async def direct_demo():
    try:
        img = 'demo_images/pizza.jpg'
        logger.info("Image received!!")
        result = direct_prediction(img)
        return jsonable_encoder(result)
    except Exception as e:
        return {"error": str(e)}



# def test():
#     # load model
#     model = tf.keras.models.load_model('models/direct_regression.h5')
#     # save as json 
#     model_json = model.to_json()
#     with open("models/direct_regression.json", "w") as json_file:
#         json_file.write(model_json)
# how to run the app (fastapi command)
# fastapi main:app --reload