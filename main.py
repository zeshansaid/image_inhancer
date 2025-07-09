import runpod
from runpod.serverless.utils.rp_validator import validate
from runpod.serverless.utils.rp_upload import upload_file_to_bucket
from runpod.serverless.utils import rp_download, rp_cleanup
from rp_schema import INPUT_SCHEMA

import logging
import requests
import os
import time
import torch
import base64
import gdown
from PIL import Image, ImageFile
from enhancer_engine import RealESRGAN
import io

import cloudinary.uploader
import cloudinary.api

target_size_kb=100
# Cloudinary configuration (as shown above)
cloudinary.config(
    cloud_name='dhut1eqjs',
    api_key='951228446286662',
    api_secret='nW4itNhexxcQMAF-75hseenbIS8',
)

ImageFile.LOAD_TRUNCATED_IMAGES = True

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
logging.info(f"Device detected: {device}")


def upload_or_base64_encode(file_name, img_path):
    """
    Uploads image to S3 bucket if it is available, otherwise returns base64 encoded image.
    """
    logging.info("Uploading or encoding image.")
    if os.environ.get('BUCKET_ENDPOINT_URL', False):
        logging.info("Uploading to S3 bucket.")
        return upload_file_to_bucket(file_name, img_path)
    
    logging.debug("Encoding image to base64.")
    with open(img_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
        return encoded_string.decode("utf-8")

# Take in base64 string and return PIL image
def stringToImage(base64_string):
    logging.debug("Converting base64 string to image.")
    imgdata = base64.b64decode(base64_string)
    return Image.open(io.BytesIO(imgdata))

# Path for the weights directory and model file
weights_dir = "weights"
checkpoint_path = os.path.join(weights_dir, "RealESRGAN_x4.pth")

# Ensure the weights directory exists
if not os.path.exists(weights_dir):
    os.makedirs(weights_dir)
    logging.info(f"Created directory: {weights_dir}")

# Check if the model file exists
if not os.path.exists(checkpoint_path):
    print("Model file not found. Downloading...")
    logging.info("Model file not found. Downloading...")
    
    # Google Drive link (use the correct direct download link or fuzzy=True for ID)
    link = "https://drive.google.com/uc?id=1K0csgiub_sUbxASV1lvE7YKcmBgsgPAq"  # Updated for direct download
    
    # Download the model file
    gdown.download(link, checkpoint_path, quiet=False, fuzzy=True)
    
    if os.path.exists(checkpoint_path):
        print("Download complete.")
        logging.info("Download complete.")
    else:
        print("Error: Download failed.")
        logging.error("Error: Download failed.")
else:
    logging.info("Model file found. Skipping download.")
    print("Model file found. Skipping download.")


Enhance_model = RealESRGAN(device, scale=4)
Enhance_model.load_weights('weights/RealESRGAN_x4.pth', download=False)

def run(job):
    # Get the uploaded image from the request
    logging.info("New upscaler request received.")
    print("New upscaler request received.")
    job_input = job['input']
    print(f"new Request recive {time.time()} : {job_input}")

    try:
        response = requests.get(job_input['image'], stream=True)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        logging.error(f"Error: {e}")
        return {"error": "Error: Failed to download image."}
    
    try:
        print("Processing Start ....")
        # Use BytesIO to treat the raw response content as a file-like object
        image = Image.open(io.BytesIO(response.content))
        # Check current size
        current_size_bytes = os.path.getsize(image)
        current_size_kb = current_size_bytes / 1024

        img = image
        print(f"Original image size: {current_size_kb:.2f} KB")
        if current_size_kb > target_size_kb:
            print(f"Image is larger than {target_size_kb} KB. Reducing size...")

            # Option 1: Reduce quality (for JPEG)
            if img.format == "JPEG":
                output_buffer = io.BytesIO()
                img.save(output_buffer, format="JPEG", quality=85) # Adjust quality as needed
                reduced_size_kb = output_buffer.tell() / 1024
                print(f"Size after quality reduction: {reduced_size_kb:.2f} KB")
                if reduced_size_kb > target_size_kb:
                    # If still too large, try resizing
                    print("Quality reduction not enough, attempting resize...")
                    width, height = img.size
                    new_width = int(width * 0.8) # Reduce by 20%
                    new_height = int(height * 0.8)
                    img = img.resize((new_width, new_height), Image.LANCZOS) # Use LANCZOS for better quality
                    #img.save(image_path, format="JPEG", quality=85)
                    #print(f"Image resized and saved to {image_path}")
                #else:
                    #with open(image_path, 'wb') as f:
                        #f.write(output_buffer.getvalue())
                    #print(f"Image quality reduced and saved to {image_path}")

            # Option 2: Resize (for other formats or if quality reduction isn't enough)
            else:
                width, height = img.size
                new_width = int(width * 0.8) # Reduce by 20%
                new_height = int(height * 0.8)
                img = img.resize((new_width, new_height), Image.LANCZOS)
                #img.save(image_path) # Save with default quality for other formats
                #print(f"Image resized and saved to {image_path}")
        else:
            print("Image size is within the target limit.")


        stime = time.perf_counter()
        image = Enhance_model.predict(img)
        etime = time.perf_counter()
        print(f'Interence time : {str(etime-stime)} sec')
        logging.info(f"Inference completed in {etime - stime:.2f} seconds.")

        # convert Image to byte io to send it back to the client
        output = io.BytesIO()
        image.save(output, format="PNG")
        upload_result = cloudinary.uploader.upload(output.getvalue())
        
        #image.save("output.png")
        # adding cloudinary in to save the image and return to mobile app 
        return {"image": upload_result['secure_url']}

        
    except Exception as e:
        logging.error(f"Error: {e}")
        return {"error": f"Error: Failed to process image. {e}"}
    
    #base64Image = job_input['image']
    
    # #print(base64Image[:20])
    # base64_data = ""
    # #extracting the base64 string part
    # logging.debug("Extracting base64 string from input.")
    # comma_index = base64Image.find(',')
    # if comma_index != -1:
    #    base64_string = base64Image[comma_index + 1:]
    #    base64_data = base64_string
    # else:
    #    print("Comma not found in the data URI.")  
    #    base64_data = base64Image
    


    # if not base64_data:
    #     logging.error("Invalid base64 data.")
    #     return {"error": "Invalid base64 data."}

    # try:
    #     img = stringToImage(base64_data)
    # except Exception as e:
    #     logging.exception("Failed to decode base64 image.")
    #     return {"error": f"Image decoding error: {e}"}

    # try:
    #     logging.info("Enhancing image using RealESRGAN.")
    #     stime = time.perf_counter()
    #     sr_image = Enhance_model.predict(img)
    #     etime = time.perf_counter()
    #     print(f'Interence time : {str(etime-stime)} sec')
    #     logging.info(f"Inference completed in {etime - stime:.2f} seconds.")

    #     bio = io.BytesIO()
    #     sr_image.save(bio, "PNG")
    #     bio.seek(0)
    #     im_b64 = base64.b64encode(bio.getvalue()).decode()

    #     return im_b64
    # except Exception as e:
    #     logging.exception("Image enhancement failed.")
    #     return {"error": f"Enhancement error: {e}"}


 
if __name__ == '__main__':
        logging.info("Starting RunPod serverless handler.")
        runpod.serverless.start({"handler": run})