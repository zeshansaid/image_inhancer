import runpod
from runpod.serverless.utils.rp_validator import validate
from runpod.serverless.utils.rp_upload import upload_file_to_bucket
from runpod.serverless.utils import rp_download, rp_cleanup

from rp_schema import INPUT_SCHEMA

import os
import time
import torch
import base64
import gdown
from PIL import Image, ImageFile
from enhancer_engine import RealESRGAN
import io

ImageFile.LOAD_TRUNCATED_IMAGES = True


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


def upload_or_base64_encode(file_name, img_path):
    """
    Uploads image to S3 bucket if it is available, otherwise returns base64 encoded image.
    """
    if os.environ.get('BUCKET_ENDPOINT_URL', False):
        return upload_file_to_bucket(file_name, img_path)

    with open(img_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
        return encoded_string.decode("utf-8")



# Take in base64 string and return PIL image
def stringToImage(base64_string):
    imgdata = base64.b64decode(base64_string)
    return Image.open(io.BytesIO(imgdata))

Enhance_model = RealESRGAN(device, scale=4)
checkpoint_path = "weights"

checkpoint_path = os.path.join(checkpoint_path, "RealESRGAN_x4.pth")
# check weigts file if its empty then download file else pass it as it is 
if len(os.listdir('weights')) == 0:
    # download  the model
    print("Model file not found. Downloading...")
    link = "https://drive.google.com/file/d/1K0csgiub_sUbxASV1lvE7YKcmBgsgPAq/view?usp=drive_link"  # Replace with your Google Drive link
    gdown.download(link, checkpoint_path, quiet=False,fuzzy=True)
    print("Download complete.")
    pass
else:    
    pass
Enhance_model.load_weights('weights/RealESRGAN_x4.pth', download=False)

def run(job):
    # Get the uploaded image from the request
    print("/upscaler new request coming")
    job_input = job['input']
    base64Image = job_input['image']
    
    #print(base64Image[:20])
    base64_data = ""
    #extracting the base64 string part 
    comma_index = base64Image.find(',')
    if comma_index != -1:
       base64_string = base64Image[comma_index + 1:]
       base64_data = base64_string
    else:
       print("Comma not found in the data URI.")  
       base64_data = base64Image
    img =  stringToImage(base64_data)
    # img.save("upscaler_requsted_image.png")
    # new_image = img#Image.open("upscaler_requsted_image.png")
    # Perform image enhancement using the model
    stime = time.perf_counter()
    sr_image = Enhance_model.predict(img)
    etime = time.perf_counter()
    print(f'Interence time : {str(etime-stime)} sec')
    # sr_image.save("upscaler_result_image.png")
    bio = io.BytesIO()
    sr_image.save(bio, "PNG")
    bio.seek(0)
    im_b64 = base64.b64encode(bio.getvalue()).decode()
    #print(f"sr_image string = {im_b64[20:]}")
    return im_b64




 
if __name__ == '__main__':
        runpod.serverless.start({"handler": run})