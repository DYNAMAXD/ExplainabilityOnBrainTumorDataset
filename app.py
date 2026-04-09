import os
import base64
import numpy as np
import cv2
import tensorflow as tf
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import io

app = FastAPI(title="Brain Tumor XAI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model globally
model_path = "resnet_finetuned_model_v1.keras"
if os.path.exists(model_path):
    print("Loading model...")
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully.")
else:
    print(f"Warning: Model not found at {model_path}.")
    model = None

class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

def img_to_base64(img_array):
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode('.jpg', img_bgr)
    return base64.b64encode(buffer).decode('utf-8')

def superimpose_heatmap(original_img, heatmap_arr):
    # Resize heatmap to match original image if necessary, but original_img here is resized? 
    # original_img usually is kept its own size, but for UI, we will resize both to e.g. 256x256 or just the original size
    heatmap_resized = cv2.resize(heatmap_arr, (original_img.shape[1], original_img.shape[0]))
    # Convert heatmap to RGB 
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    superimposed_img = heatmap_rgb * 0.5 + original_img * 0.5
    return np.uint8(superimposed_img)

def compute_gradcam(img_array, model, layer_name="conv5_block3_out"):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(layer_name).output, model.output]
    )
    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor)
        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap) + 1e-8
    return heatmap.numpy()

def compute_gradcam_plus_plus(img_array, model, layer_name="conv5_block3_out"):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(layer_name).output, model.output]
    )
    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor)
        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]
    grads = tape.gradient(loss, conv_outputs)
    conv_outputs = conv_outputs[0]
    grads = grads[0]
    grads_power_2 = grads ** 2
    grads_power_3 = grads ** 3
    sum_activations = tf.reduce_sum(conv_outputs, axis=(0,1))
    alpha_num = grads_power_2
    alpha_denom = 2*grads_power_2 + grads_power_3 * sum_activations
    alpha_denom = tf.where(alpha_denom != 0.0, alpha_denom, tf.ones_like(alpha_denom))
    alphas = alpha_num / alpha_denom
    weights = tf.reduce_sum(alphas * tf.nn.relu(grads), axis=(0,1))
    heatmap = tf.reduce_sum(weights * conv_outputs, axis=-1)
    heatmap = tf.maximum(heatmap,0)
    heatmap /= tf.reduce_max(heatmap) + 1e-8
    return heatmap.numpy()

def compute_scorecam(img_array, model, layer_name="conv5_block3_out"):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(layer_name).output]
    )
    conv_outputs = grad_model(img_array)
    conv_outputs = conv_outputs[0].numpy()
    heatmap = np.zeros(conv_outputs.shape[:2])
    
    # Restrict channel iteration to speed up server response if it's too slow:
    # We will use sub-sampling of channels to speed it up to ~50 channels
    num_channels = min(50, conv_outputs.shape[-1])
    channels_to_process = np.linspace(0, conv_outputs.shape[-1]-1, num_channels, dtype=int)
    
    for i in channels_to_process:
        activation = conv_outputs[:,:,i]
        activation_norm = (activation - activation.min()) / (activation.max() - activation.min() + 1e-8)
        activation_resized = cv2.resize(activation_norm, (224,224))
        masked_img = img_array * activation_resized[...,np.newaxis]
        preds = model.predict(masked_img, verbose=0)
        score = np.max(preds)
        heatmap += activation_norm * score

    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) + 1e-8
    return heatmap

def compute_occlusion(img_array, model, patch_size=20, stride=20):
    img = img_array.copy()
    h = img.shape[1]
    w = img.shape[2]
    heatmap = np.zeros((h, w))
    counts = np.zeros((h, w))
    
    preds = model.predict(img_array, verbose=0)
    class_idx = np.argmax(preds)
    baseline_score = preds[0][class_idx]

    # Stride of 20 to speed up API response
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            occluded = img.copy()
            occluded[:, y:y+patch_size, x:x+patch_size, :] = 0
            preds_occ = model.predict(occluded, verbose=0)
            score = preds_occ[0][class_idx]
            importance = baseline_score - score
            heatmap[y:y+patch_size, x:x+patch_size] += importance
            counts[y:y+patch_size, x:x+patch_size] += 1

    heatmap = heatmap / (counts + 1e-8)
    heatmap = np.maximum(heatmap, 0)
    heatmap = heatmap / (np.max(heatmap) + 1e-8)
    heatmap = cv2.GaussianBlur(heatmap, (11,11), 0)
    return heatmap

def compute_eigencam(img_array, model, layer_name="conv5_block3_out"):
    feature_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(layer_name).output]
    )
    feature_maps = feature_model.predict(img_array, verbose=0)
    feature_maps = feature_maps[0]
    h, w, c = feature_maps.shape
    reshaped = feature_maps.reshape((h*w, c))
    reshaped = reshaped - np.mean(reshaped, axis=0)
    U, S, Vt = np.linalg.svd(reshaped, full_matrices=False)
    principal_component = Vt[0]
    heatmap = np.dot(reshaped, principal_component)
    heatmap = heatmap.reshape(h, w)
    heatmap = np.maximum(heatmap, 0)
    heatmap = heatmap / (np.max(heatmap) + 1e-8)
    return heatmap

def compute_layercam(img_array, model, layer_name="conv5_block3_out"):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(layer_name).output, model.output]
    )
    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor)
        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]
    grads = tape.gradient(loss, conv_outputs)
    conv_outputs = conv_outputs[0]
    grads = grads[0]
    positive_grads = tf.nn.relu(grads)
    layercam = positive_grads * conv_outputs
    heatmap = tf.reduce_sum(layercam, axis=-1)
    heatmap = tf.maximum(heatmap,0)
    heatmap /= tf.reduce_max(heatmap) + 1e-8
    return heatmap.numpy()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        return JSONResponse(status_code=500, content={"error": "Model not loaded"})
    
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    original_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Process for ResNet
    img_resized = cv2.resize(original_img, (224, 224))
    img_array = np.array(img_resized, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.resnet50.preprocess_input(img_array)
    
    preds = model.predict(img_array)
    pred_index = int(np.argmax(preds))
    pred_class = class_names[pred_index]
    confidence = float(preds[0][pred_index])
    
    # Compute CAMs
    # Note: to ensure high performance locally, we limit original size base64 to scaled versions if too large,
    # but 224x224 to 500x500 is fine. Here we scale original_img for visualization to 300x300
    viz_img = cv2.resize(original_img, (300, 300))
    
    orig_b64 = img_to_base64(viz_img)
    
    hm_gradcam = compute_gradcam(img_array, model)
    hm_gradcam_pp = compute_gradcam_plus_plus(img_array, model)
    hm_scorecam = compute_scorecam(img_array, model)
    hm_occlusion = compute_occlusion(img_array, model)
    hm_eigencam = compute_eigencam(img_array, model)
    hm_layercam = compute_layercam(img_array, model)
    
    cams = {
        "gradcam": img_to_base64(superimpose_heatmap(viz_img, hm_gradcam)),
        "gradcam_pp": img_to_base64(superimpose_heatmap(viz_img, hm_gradcam_pp)),
        "scorecam": img_to_base64(superimpose_heatmap(viz_img, hm_scorecam)),
        "occlusion": img_to_base64(superimpose_heatmap(viz_img, hm_occlusion)),
        "eigencam": img_to_base64(superimpose_heatmap(viz_img, hm_eigencam)),
        "layercam": img_to_base64(superimpose_heatmap(viz_img, hm_layercam))
    }
    
    return {
        "pred_class": pred_class,
        "confidence": confidence,
        "original_image": orig_b64,
        "cams": cams
    }

print("Mounting static files...")
os.makedirs("static", exist_ok=True)
app.mount("/", StaticFiles(directory="static", html=True), name="static")

