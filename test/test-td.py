import sys
import numpy as np

import torch
from diffusers import AutoencoderTiny, StableDiffusionPipeline
from diffusers.utils import load_image

from streamdiffusion import StreamDiffusion
from streamdiffusion.image_utils import postprocess_image

from config import config, Args
from img2img import Pipeline

from PIL import Image

from server import Server
import asyncio

init_image = load_image("../assets/img2img_example.png")


# # Run the stream infinitely
# while True:
#     x_output = stream(init_image)
#     postprocess_image(x_output, output_type="pil")[0].show()
#     input_response = input("Press Enter to continue or type 'stop' to exit: ")
#     if input_response == "stop":
#         break

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch_dtype = torch.float16
pipeline = Pipeline(config, device, torch_dtype)

 
# create handler for each connection


async def main():
    # warmup pipeline ?
    for _ in range(pipeline.stream.batch_size - 1):
        pipeline.predict_test(init_image)
    
    # out = pipeline.predict_test(init_image)
    # init_image.save("hello_init.png")
    # out.save("hello_out.png")

    # # while False:
    # while True:
    #     ### convert frame to pil Image
    #     image = Image.fromarray(v.data, mode="RGBA")
    #     out = pipeline.predict_test(image)

    #     out.save("out-rgb.png")
    #     continue

    server = Server(pipeline)
    await server.Setup()
    # async with serve(echo, "localhost", 8765):
    #     await asyncio.Future()  # run forever
    return 0


if __name__ == "__main__":
    asyncio.run(main())