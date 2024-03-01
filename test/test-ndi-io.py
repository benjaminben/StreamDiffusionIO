import sys
import numpy as np
import NDIlib as ndi

import torch
from diffusers import AutoencoderTiny, StableDiffusionPipeline
from diffusers.utils import load_image

from streamdiffusion import StreamDiffusion
from streamdiffusion.image_utils import postprocess_image

from config import config, Args
from img2img import Pipeline

from PIL import Image

print(dir(ndi))

# # You can load any models using diffuser's StableDiffusionPipeline
# pipe = StableDiffusionPipeline.from_pretrained("KBlueLeaf/kohaku-v2.1").to(
#     device=torch.device("cuda"),
#     dtype=torch.float16,
# )

# # Wrap the pipeline in StreamDiffusion
# stream = StreamDiffusion(
#     pipe,
#     t_index_list=[32, 45],
#     torch_dtype=torch.float16,
# )

# # If the loaded model is not LCM, merge LCM
# stream.load_lcm_lora()
# stream.fuse_lora()
# # Use Tiny VAE for further acceleration
# stream.vae = AutoencoderTiny.from_pretrained("madebyollin/taesd").to(device=pipe.device, dtype=pipe.dtype)
# # Enable acceleration
# pipe.enable_xformers_memory_efficient_attention()


# prompt = "1girl with dog hair, thick frame glasses"
# # Prepare the stream
# stream.prepare(prompt)

# # Prepare image
init_image = load_image("../assets/img2img_example.png").resize((512, 512))

# # Warmup >= len(t_index_list) x frame_buffer_size
# for _ in range(2):
#     stream(init_image)



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

def main():
    # ndi setup
    if not ndi.initialize():
        return 0

    ndi_find = ndi.find_create_v2()

    if ndi_find is None:
        return 0

    sources = []
    while not len(sources) > 0:
        print('Looking for sources ...')
        ndi.find_wait_for_sources(ndi_find, 1000)
        sources = ndi.find_get_current_sources(ndi_find)

    # ndi receive
    ndi_recv_create = ndi.RecvCreateV3()
    ndi_recv_create.color_format = ndi.RECV_COLOR_FORMAT_E_RGBX_RGBA

    ndi_recv = ndi.recv_create_v3(ndi_recv_create)

    if ndi_recv is None:
        return 0

    ndi.recv_connect(ndi_recv, sources[0])

    ndi.find_destroy(ndi_find)

    # ndi send
    send_settings = ndi.SendCreate()
    send_settings.ndi_name = 'ndi-python'
    ndi_send = ndi.send_create(send_settings)

    if ndi_send is None:
        return 0

    for _ in range(2):
        pipeline.predict_test(init_image)
    
    out = pipeline.predict_test(init_image)
    init_image.save("hello_init.png")
    out.save("hello_out.png")

    # while False:
    while True:
        t, v, a, _ = ndi.recv_capture_v2(ndi_recv, 5000)

        if t == ndi.FRAME_TYPE_NONE:
            print('No data received.')
            continue

        if t == ndi.FRAME_TYPE_VIDEO:
            #print('Video data received (%dx%d).' % (v.xres, v.yres))

            # postprocess_image(x_output, output_type="pil")[0].show()
            # image = pipeline.predict()

            ### convert frame to pil Image
            # print(v.data)
            # image = Image.fromarray(v.data.astype('uint8'), mode="RGB")
            image = Image.fromarray(v.data, mode="RGBA")
            #image.save("test-pil-ndi.png")
            # image.save("test.png")
            # image.convert('RGB')
            # print("RECV IMAGE:", image)
            # out = stream(image)
            # out = stream(image)
            # postprocess_image(out, output_type="pil")[0].show()
            # input_response = input("Press Enter to continue or type 'stop' to exit: ")
            
            out = pipeline.predict_test(image)
            # out = image
            out.save("out-rgb.png")
            fin = out.convert("RGBA")
            fin.save("out-rgba.png")
            
            # video_frame = ndi.VideoFrameV2()
            # video_frame.data = v.data
            # video_frame.data = postprocess_image(out, output_type="pil")[0]
            # video_frame.data = out
            # video_frame.data = v.data
            # video_frame.data = postprocess_image(out, output_type="np")
            # print(image)
            # video_frame.data = Image.fromarray(v.data, mode="RGB")
            # video_frame.data = image
            # video_frame.FourCC = ndi.FOURCC_VIDEO_TYPE_RGBA

            # ndi.send_send_video_v2(ndi_send, video_frame)
            # ndi.recv_free_video_v2(ndi_recv, v)
            continue

        if t == ndi.FRAME_TYPE_AUDIO:
            print('Audio data received (%d samples).' % a.no_samples)
            ndi.recv_free_audio_v2(ndi_recv, a)
            continue

    # ndi.recv_destroy(ndi_recv)
    ndi.send_destroy(ndi_send)

    ndi.destroy()

    return 0


if __name__ == "__main__":
    sys.exit(main())