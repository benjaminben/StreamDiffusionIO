import asyncio
from websockets.server import serve
from diffusers.utils import load_image
import io
from PIL import Image

# delete l8r
init_image = load_image("../assets/img2img_example.png").resize((512, 512))

class Server:
    def __init__(self, pipeline):
        # self.setup()
        self.pipeline = pipeline
        return
    async def Setup(self):
        print("starting server...")
        async with serve(self.echo, "localhost", 8765):
            print("here we go")
            await asyncio.Future()  # run forever
    async def handler(self, websocket, path):
        print("got sum")
        data = await websocket.recv()
        print("data:", data)
        reply = f"Data recieved as:  {data}!"
        await websocket.send(reply)
    async def echo(self, websocket):
        print("serving")
        myPrompt = "1girl"
        async for message in websocket:
            if type(message) == str:
                myPrompt = message
            else:
                buff = io.BytesIO(message)
                image = Image.open(buff)
                out = self.pipeline.predict_test(image, myPrompt)
                img_byte_arr = io.BytesIO()
                out.save(img_byte_arr, format='PNG')
                await websocket.send(img_byte_arr.getvalue())