import sys
from PIL import Image
from diffusers.utils import load_image
import NDIlib as ndi

print(dir(ndi))

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

    # init_image = load_image("../assets/img2img_example.png").resize((512, 512))

    # ndi send
    send_settings = ndi.SendCreate()
    send_settings.ndi_name = 'ndi-python'
    ndi_send = ndi.send_create(send_settings)

    while True:
        t, v, a, _ = ndi.recv_capture_v2(ndi_recv, 5000)

        if t == ndi.FRAME_TYPE_NONE:
            print('No data received.')
            continue

        if t == ndi.FRAME_TYPE_VIDEO:
            image = Image.fromarray(v.data, mode="RGBA")
            # image.convert("RGB").resize((512, 512))
            #image.save("test-pil-ndi.png")
            video_frame = ndi.VideoFrameV2()
            video_frame.data = image
            video_frame.FourCC = ndi.FOURCC_VIDEO_TYPE_RGBA

            ndi.send_send_video_v2(ndi_send, video_frame)
            ndi.recv_free_video_v2(ndi_recv, v)
            continue
#            break

        if t == ndi.FRAME_TYPE_AUDIO:
            print('Audio data received (%d samples).' % a.no_samples)
            ndi.recv_free_audio_v2(ndi_recv, a)
            continue

if __name__ == "__main__":
    sys.exit(main())