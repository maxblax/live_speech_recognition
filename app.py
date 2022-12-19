import queue
from datetime import datetime
from pprint import pformat

import time
from io import BytesIO

import pydub
import streamlit as st

from streamlit_webrtc import WebRtcMode, webrtc_streamer
import whisper

from utils import load_audio, load_html_component_from_file

import logging

logger = logging.getLogger(__name__)
# not working properly with streamlit..
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(funcName)2s : %(message)s",
                    level=logging.DEBUG)
logger.setLevel(level=logging.DEBUG)

CLOSING_PUNCTUATION = [",", ".", "!", "?", ";", ":"]

def main():
    st.set_page_config(layout="wide")

    st.header("Multilingual Speech Recognition - Xmas 22!")
    col1, col2 = st.columns(2)
    with col1:
        load_html_component_from_file('audio_motion_analyzer.html')
    with col2:
        app_sst()


@st.cache
def load_ai_model(type="base"):
    model = whisper.load_model(type)
    return model

def app_sst():
    status_indicator = st.empty()
    model = load_ai_model(type="base")
    text_output = st.empty()
    webrtc_ctx = webrtc_streamer(
        key="speech-to-text",
        mode=WebRtcMode.SENDONLY,
        audio_receiver_size=1024,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": False, "audio": True},
        # media_stream_constraints=constraints,
    )
    if not webrtc_ctx.state.playing:
        logger.debug('State is not playing, I m done in app_sst')
        return

    agg_audio_segments = pydub.AudioSegment.empty()
    text = ""
    b1_audio_segments = []
    sample_width = None
    result = None
    while True:
        if webrtc_ctx.audio_receiver:
            try:
                audio_frames = webrtc_ctx.audio_receiver.get_frames(timeout=1)
            except queue.Empty:
                time.sleep(0.1)
                status_indicator.write("No frame arrived.")
                continue
            status_indicator.write("C'est parti, dit quelque chose: ")

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
                audio_segment = pydub.AudioSegment(
                    data=audio_frame.to_ndarray().tobytes(),
                    sample_width=audio_frame.format.bytes,
                    frame_rate=audio_frame.sample_rate,
                    channels=len(audio_frame.layout.channels),
                )
                agg_audio_segments += audio_segment
                b1_audio_segments.append(audio_segment)

            if agg_audio_segments.duration_seconds >= 1:
                logger.info("---------------------------------- %s "% agg_audio_segments.duration_seconds)
                f = BytesIO()
                agg_audio_segments.export(f, format='wav')
                f.seek(0)
                start = datetime.now()
                b2 = load_audio(f.read())
                logger.debug("load_audio in %s"% (datetime.now() - start).total_seconds())
                # logger.debug("audio.shape: %s " % str(b2.shape))
                # logger.debug("audio: %s " % str(b2))
                start = datetime.now()
                result = whisper.transcribe(model=model, audio=b2, fp16=False)
                logger.info("whisper transcribe in %s"% (datetime.now() - start).total_seconds())

                logger.debug(pformat(result, compact=True))
                # logger.debug(result)
                c_text = result['text']
                # We display the text
                text_output.markdown(f"**Text:** {text} \n\n {c_text}")

            if agg_audio_segments.duration_seconds >= 25:
                logger.debug("result: %s " % pformat(result))
                # Selecting the first closing segment in last results, that end position will diminish the overall buffer
                # size under 22 seconds
                i = -1
                duration_removed = 0
                while len(result['segments']) > i + 1 \
                    and agg_audio_segments.duration_seconds - duration_removed > 20 \
                    and result['segments'][i]['text'].endswith(tuple(CLOSING_PUNCTUATION)):
                    i += 1
                    duration_removed = result['segments'][i]['end']
                    text += result['segments'][i]['text']

                duration_to_pop = max(result['segments'][i]['end'], agg_audio_segments.duration_seconds - 22)\
                                        if i > -1 else agg_audio_segments.duration_seconds - 22
                logger.debug("Duration to remove from original buffer: %s"% duration_to_pop)
                acc = 0
                while len(b1_audio_segments) > 0 and acc < duration_to_pop:
                    audio_segment = b1_audio_segments.pop(0)
                    acc += audio_segment.duration_seconds

                logger.debug("Duration removed: %s"%acc)
                # Now rebuilding the aggregated segment since __pydub__ does not implement -
                agg_audio_segments = pydub.AudioSegment.empty()
                for audio_segment in b1_audio_segments:
                    agg_audio_segments += audio_segment
        else:
            status_indicator.write("AudioReceiver is not set. Abort.")
            break
        # Little nap to cool me down
        time.sleep(0.1)


if __name__ == "__main__":

    st_webrtc_logger = logging.getLogger("streamlit_webrtc")
    st_webrtc_logger.setLevel(logging.DEBUG)
    fsevents_logger = logging.getLogger("fsevents")
    fsevents_logger.setLevel(logging.WARNING)
    main()
