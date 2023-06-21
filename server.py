import argparse
import asyncio
import json
import logging
import os
import ssl
import uuid

from aiohttp import web
from av import VideoFrame

from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaBlackhole, MediaPlayer, MediaRecorder, MediaRelay

from time import time
import threading

from methods.classification import Classification
from methods.detection import ObjectDetection
from methods.instance_segmentation import InstanceSegmentation
from methods.semantic_segmentation import SemanticSegmentation
from methods.detection_yolo import ObjectDetectionYOLO

ROOT = os.path.dirname(__file__)

logger = logging.getLogger("pc")
pcs = set()
relay = MediaRelay()

classification = Classification()
object_detection = ObjectDetection()
object_detection_yolo = ObjectDetectionYOLO()
semantic_segmentation = SemanticSegmentation()
instance_segmentation = InstanceSegmentation()

class VideoTransformTrack(MediaStreamTrack):
    """
    A video stream track that transforms frames from an another track.
    """

    kind = "video"

    def __init__(self, track, transform):
        super().__init__()
        self.track = track
        self.transform = transform
        self.nof_frames = 0
        self.round_trip_time = []
        self.jitter = []
        self.time = time()
        self.times = []

    async def recv(self):
        time_start = time()
        #print(time_start)
        frame = await self.track.recv()
        stats = await list(pcs)[0].getStats()
        stats_list = list(stats.values())
        logging.info(f"WebRTC stats: {stats_list}")
        # For latency statistics
        inbound_stats = stats_list[2]
        
        difference = int((time_start - self.time))
        if (difference <= 60 and hasattr(inbound_stats, 'roundTripTime') and hasattr(inbound_stats, 'jitter')):
            self.times.append(difference)
            self.round_trip_time.append(getattr(stats_list[2], 'roundTripTime'))
            self.jitter.append(getattr(stats_list[2], 'jitter'))
        else:
            print(self.times)
            print("\n")
            print(self.round_trip_time)
            print("\n")
            print(self.jitter)

        self.nof_frames += 1

        if self.transform == "classification":
            img = frame.to_ndarray(format="rgb24")
            processed_img = classification.classify_objects_in_frame(img)
            new_frame = VideoFrame.from_ndarray(processed_img, format="rgb24")
            new_frame.pts = frame.pts
            new_frame.time_base = frame.time_base
            return new_frame
        elif self.transform == "detection":
            img = frame.to_ndarray(format="rgb24")
            processed_img = object_detection.detect_objects_in_frame(img)
            new_frame = VideoFrame.from_ndarray(processed_img, format="rgb24")
            new_frame.pts = frame.pts
            new_frame.time_base = frame.time_base
            return new_frame
        elif self.transform == "detection_yolo":
            #print("Time transmission start to processing: " + str(time() - time_start))
            img = frame.to_ndarray(format="rgb24")
            processed_img = object_detection_yolo.detect_objects_in_frame(img)
            new_frame = VideoFrame.from_ndarray(processed_img, format="rgb24")
            new_frame.pts = frame.pts
            new_frame.time_base = frame.time_base
            return new_frame
        elif self.transform == "semantic_segmentation":
            img = frame.to_ndarray(format="rgb24")
            processed_img = semantic_segmentation.recognize_objects_in_frame(img)
            new_frame = VideoFrame.from_ndarray(processed_img, format="rgb24")
            new_frame.pts = frame.pts
            new_frame.time_base = frame.time_base
            return new_frame
        elif self.transform == "instance_segmentation":
            img = frame.to_ndarray(format="rgb24")
            processed_img = instance_segmentation.recognize_objects_in_frame(img)
            new_frame = VideoFrame.from_ndarray(processed_img, format="rgb24")
            new_frame.pts = frame.pts
            new_frame.time_base = frame.time_base
            return new_frame
        else:
            return frame
    #print("Time end: " + str(time()))

async def index(request):
    content = open(os.path.join(ROOT, "index.html"), "r").read()
    return web.Response(content_type="text/html", text=content)


async def javascript(request):
    content = open(os.path.join(ROOT, "client.js"), "r").read()
    return web.Response(content_type="application/javascript", text=content)


async def offer(request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    pc_id = "PeerConnection(%s)" % uuid.uuid4()
    pcs.add(pc)

    def log_info(msg, *args):
        logger.info(pc_id + " " + msg, *args)

    log_info("Created for %s", request.remote)

    # prepare local media
    #player = MediaPlayer(os.path.join(ROOT, "test.mp4"))

    if args.record_to:
        recorder = MediaRecorder(args.record_to)
    else:
        recorder = MediaBlackhole()

    @pc.on("datachannel")
    def on_datachannel(channel):
        @channel.on("message")
        def on_message(message):
            if isinstance(message, str) and message.startswith("ping"):
                channel.send("pong" + message[4:])

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        log_info("Connection state is %s", pc.connectionState)
        if pc.connectionState == "failed":
            await pc.close()
            pcs.discard(pc)

    @pc.on("track")
    def on_track(track):
        log_info("Track %s received", track.kind)

        if track.kind == "audio":
            #pc.addTrack(player.audio)
            recorder.addTrack(track)
        elif track.kind == "video":
            pc.addTrack(
                VideoTransformTrack(
                    relay.subscribe(track), transform=params["video_transform"]
                )
            )
            if args.record_to:
                recorder.addTrack(relay.subscribe(track))

        @track.on("ended")
        async def on_ended():
            log_info("Track %s ended", track.kind)
            await recorder.stop()

    # handle offer
    await pc.setRemoteDescription(offer)
    await recorder.start()

    # send answer
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
        ),
    )

async def on_shutdown(app):
    # close peer connections
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="WebRTC audio / video / data-channels demo"
    )
    parser.add_argument("--cert-file", help="SSL certificate file (for HTTPS)")
    parser.add_argument("--key-file", help="SSL key file (for HTTPS)")
    parser.add_argument(
        "--host", default="0.0.0.0", help="Host for HTTP server (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", type=int, default=8080, help="Port for HTTP server (default: 8080)"
    )
    parser.add_argument("--record-to", help="Write received media to a file."),
    parser.add_argument("--verbose", "-v", action="count")

    parser.add_argument(
        "--cert_file", type=str, default="cert.pem", help="Path for SSL cert file"
    )
    parser.add_argument(
        "--key_file", type=str, default="key.pem", help="Path for SSL key file"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    if args.cert_file:
        ssl_context = ssl.SSLContext()
        ssl_context.load_cert_chain(args.cert_file, args.key_file)
    else:
        ssl_context = None

    app = web.Application()
    app.on_shutdown.append(on_shutdown)
    app.router.add_get("/", index)
    app.router.add_get("/client.js", javascript)
    app.router.add_post("/offer", offer)
    web.run_app(
        app, access_log=None, host=args.host, port=args.port, ssl_context=ssl_context
    )