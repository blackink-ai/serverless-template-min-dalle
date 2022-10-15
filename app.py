import torch
from min_dalle import MinDalle
import base64
from io import BytesIO
from PIL import Image

# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"
def init():
  global model

  device = "cuda" if torch.cuda.is_available() else "cpu"
  model = MinDalle(
    models_root='./pretrained',
    dtype=torch.float32,
    device=device,
    is_mega=True,
    is_reusable=True
  )

# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs:dict) -> dict:
  global model

  # Parse out your arguments
  prompt = model_inputs.get('prompt', None)
  if prompt == None:
    return {'message': "No prompt provided"}

  seed = model_inputs.get('seed', -1);
  grid_size = model_inputs.get('grid_size', 3);
  top_k = model_inputs.get('top_k', 128);
  supercondition_factor = model_inputs.get('supercondition_factor', 16);
  temperature = model_inputs.get('temperature', 1);

  # Run the model
  images = model.generate_images(
    text=prompt,
    seed=seed,
    grid_size=grid_size,
    is_seamless=False,
    temperature=temperature,
    top_k=top_k,
    supercondition_factor=supercondition_factor,
    is_verbose=False
  )

  result = {
    "images_base64": []
  }

  for raw_image in images.to('cpu').numpy():
    buffered = BytesIO()
    image = Image.fromarray((raw_image * 255).astype(np.uint8))
    image.save(buffered, format="PNG")
    image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    result["images_base64"].append(image_base64)

  # Return the results as a dictionary
  return result
