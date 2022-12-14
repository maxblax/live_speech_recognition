import logging
import logging.handlers
import queue
import threading
from pprint import pformat

import time
import traceback
import urllib.request
import wave
from collections import deque
from io import BytesIO
from pathlib import Path
from typing import List

import av
import numpy as np
import pydub
import streamlit as st

from streamlit_webrtc import WebRtcMode, webrtc_streamer
import whisper
import struct
import av.audio.frame
import streamlit.components.v1 as components  # Import Streamlit
from streamlit_webrtc.config import MediaTrackConstraintSet, MediaStreamConstraints

from utils import load_audio, load_html_component_from_file

HERE = Path(__file__).parent

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(funcName)2s : %(message)s", level=logging.DEBUG)
logger.setLevel(level=logging.DEBUG)

CLOSURE_STRING = [".", "!", "?"]
ONGOING_STRING = ["..."]

def main():
    st.header("Interactive Multilingual Speech-Recognition")
    #load_html_component_from_file('audio_motion_analyzer.html')
    try:
        app_sst()
    except :
        logger.error(traceback.format_exc())
        pass



def whisper_stt(buf: BytesIO, model):
    audio = load_audio(buf.read())
    audio = whisper.pad_or_trim(audio)

    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # detect the spoken language
    _, probs = model.detect_language(mel)
    logger.info(f"Detected language: {max(probs, key=probs.get)}")
    # decode the audio
    options = whisper.DecodingOptions(fp16=False)

    result = whisper.decode(model, mel, options)
    return result.text, probs.get

@st.cache
def load_ai_model(type="base"):
    model = whisper.load_model("base")
    return model

def app_sst():
    status_indicator = st.empty()
    model = load_ai_model()
    # This ain't working for some reason.... instantiation is OK but still get audio frame in 48KHz stereo
    constraints = {"audio": {"sampleRate": 16000,
                            "channelCount": 1,
                            "echoCancellation": True,
                            "noiseSuppression": True},
                   "video": False}
    webrtc_ctx = webrtc_streamer(
        key="speech-to-text",
        mode=WebRtcMode.SENDONLY,
        audio_receiver_size=1024,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": False, "audio": True},
        # media_stream_constraints=constraints,
    )
    logger.debug("@Heere webrtc_ctx loaded")

    if not webrtc_ctx.state.playing:
        logger.debug('State is not playing, I m done in app_sst')
        return

    text_output = st.empty()
    agg_audio_segments = pydub.AudioSegment.empty()
    sample_width = None
    text = ""
    b1_audio_segments = []
    last_nbr_stable_audio_segments=0
    last_stable_text = ""
    while True:
        logger.debug("-----------------Hello")
        if webrtc_ctx.audio_receiver:
            try:
                audio_frames = webrtc_ctx.audio_receiver.get_frames(timeout=1)
            except queue.Empty:
                time.sleep(0.1)
                status_indicator.write("No frame arrived.")
                continue
            status_indicator.write("Running. Say something!")

            # If audio params are not set, init them from the first audio frame received.
            if len(audio_frames) > 0 and sample_width == None:
                audio_frame = audio_frames[0]
                sample_width = audio_frame.format.bytes
                sample_rate = audio_frame.sample_rate
                channels = len(audio_frame.layout.channels)
                logger.debug("sample_width: %d | sample_rate: %d | channels: %d "
                             "" % (sample_width, sample_rate, channels))
                logger.debug("type of audio_frames[0]: %s"% type(audio_frame))
                logger.debug("type audio_frame.to_ndarray().tobytes(): %s"% type(audio_frame.to_ndarray().tobytes()))
                logger.debug("audio_frame.samples: %s"% audio_frame.samples)
                logger.debug("audio_frame.layout: %s" % audio_frame.layout)
                logger.debug("audio_frame.to_ndarray(): %s" % audio_frame.to_ndarray())

            # For each frame of the array
            for audio_frame in audio_frames:
                # Get raw bytes and store in buffer
                # buffer_audio += audio_frame.to_ndarray().tobytes()
                audio_segment = pydub.AudioSegment(
                    data=audio_frame.to_ndarray().tobytes(),
                    sample_width=audio_frame.format.bytes,
                    frame_rate=audio_frame.sample_rate,
                    channels=len(audio_frame.layout.channels),
                )
                agg_audio_segments += audio_segment
                b1_audio_segments.append(audio_segment)

            logger.info("============================= duration: %d"
                        "" % (agg_audio_segments. duration_seconds))
            if agg_audio_segments.duration_seconds >= 1:
                f = BytesIO()
                agg_audio_segments.export(f, format='wav')
                f.seek(0)
                c_text, language = whisper_stt(f, model)

                if not c_text.endswith(ONGOING_STRING) \
                    and c_text.endswith(CLOSURE_STRING):
                    # Stacking the last stable text
                    text += last_stable_text
                    # Removing the first (heading) part of this stable string
                    c_text = c_text.replace(last_stable_text, "", 1)
                    # Loosing the head of the audio Segment array, proportionally to the last stable occurrence
                    b1_audio_segments = b1_audio_segments[last_nbr_stable_audio_segments:]
                    # Now the current buffer is the new stable version
                    last_nbr_stable_audio_segments = len(b1_audio_segments)
                    # And the current text is the 'new' last stable one.
                    last_stable_text = c_text

                    # Now we just need to rebuild the agg_audio_segment
                    # Cause pydub.AudioSegment doesn't implement __sub__
                    agg_audio_segments = pydub.AudioSegment.empty()
                    for audio_segment in b1_audio_segments:
                        agg_audio_segments += audio_segment

                # We display the text
                text_output.markdown(f"**Text:** {text} \n\n {c_text}")

                # In the event of no closing string being detected, we have to flush before reaching 30sec
                if b1_audio_segment.duration_seconds > 25:
                    b1_audio_segment = pydub.AudioSegment.empty()
                    agg_audio_segments = pydub.AudioSegment.empty()
                    text += c_text
        else:
            status_indicator.write("AudioReceiver is not set. Abort.")
            break
        time.sleep(0.1)


def app_sst_with_video(
    model_path: str, lm_path: str, lm_alpha: float, lm_beta: float, beam: int
):
    frames_deque_lock = threading.Lock()
    frames_deque: deque = deque([])

    async def queued_audio_frames_callback(
        frames: List[av.AudioFrame],
    ) -> av.AudioFrame:
        with frames_deque_lock:
            frames_deque.extend(frames)

        # Return empty frames to be silent.
        new_frames = []
        for frame in frames:
            input_array = frame.to_ndarray()
            new_frame = av.AudioFrame.from_ndarray(
                np.zeros(input_array.shape, dtype=input_array.dtype),
                layout=frame.layout.name,
            )
            new_frame.sample_rate = frame.sample_rate
            new_frames.append(new_frame)

        return new_frames

    webrtc_ctx = webrtc_streamer(
        key="speech-to-text-w-video",
        mode=WebRtcMode.SENDRECV,
        queued_audio_frames_callback=queued_audio_frames_callback,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": True},
    )

    status_indicator = st.empty()

    if not webrtc_ctx.state.playing:
        return

    status_indicator.write("Loading...")
    text_output = st.empty()
    stream = None

    while True:
        if webrtc_ctx.state.playing:
            if stream is None:
                from deepspeech import Model

                model = Model(model_path)
                model.enableExternalScorer(lm_path)
                model.setScorerAlphaBeta(lm_alpha, lm_beta)
                model.setBeamWidth(beam)

                stream = model.createStream()

                status_indicator.write("Model loaded.")

            sound_chunk = pydub.AudioSegment.empty()

            audio_frames = []
            with frames_deque_lock:
                while len(frames_deque) > 0:
                    frame = frames_deque.popleft()
                    audio_frames.append(frame)

            if len(audio_frames) == 0:
                time.sleep(0.1)
                status_indicator.write("No frame arrived.")
                continue

            status_indicator.write("Running. Say something!")

            for audio_frame in audio_frames:
                sound = pydub.AudioSegment(
                    data=audio_frame.to_ndarray().tobytes(),
                    sample_width=audio_frame.format.bytes,
                    frame_rate=audio_frame.sample_rate,
                    channels=len(audio_frame.layout.channels),
                )
                sound_chunk += sound

            if len(sound_chunk) > 0:
                sound_chunk = sound_chunk.set_channels(1).set_frame_rate(
                    model.sampleRate()
                )
                buffer = np.array(sound_chunk.get_array_of_samples())
                stream.feedAudioContent(buffer)
                text = stream.intermediateDecode()
                text_output.markdown(f"**Text:** {text}")
        else:
            status_indicator.write("Stopped.")
            break


if __name__ == "__main__":
    import os

    # DEBUG = os.environ.get("DEBUG", "false").lower() not in ["false", "no", "0"]
    #
    # logging.basicConfig(
    #     format="[%(asctime)s] %(levelname)7s from %(name)s in %(pathname)s:%(lineno)d: "
    #     "%(message)s",
    #     force=True,
    # )

    # logger.setLevel(level=logging.DEBUG if DEBUG else logging.INFO)

    st_webrtc_logger = logging.getLogger("streamlit_webrtc")
    st_webrtc_logger.setLevel(logging.DEBUG)

    fsevents_logger = logging.getLogger("fsevents")
    fsevents_logger.setLevel(logging.WARNING)
    main()
