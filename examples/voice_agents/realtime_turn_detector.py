import logging

from dotenv import load_dotenv
from google.genai import types  # noqa: F401

from livekit.agents import Agent, AgentSession, JobContext, WorkerOptions, cli
from livekit.plugins import openai
from livekit.plugins.openai.realtime.realtime_model import RealtimeCustomVoice  # noqa: F401

logger = logging.getLogger("realtime-turn-detector")
logger.setLevel(logging.INFO)

load_dotenv()

## This example demonstrates how to use LiveKit's turn detection model with a realtime LLM.
## Since the current turn detection model runs in text space, it will need to be combined
## with a STT model, even though the audio is going directly to the Realtime API.
## In this example, speech is being processed in parallel by both the STT and the realtime API


async def entrypoint(ctx: JobContext):
    session = AgentSession(
        allow_interruptions=True,
        # To use OpenAI Realtime API
        llm=openai.realtime.RealtimeModel(
            voice=RealtimeCustomVoice(id="voice_68efbce1405c8191be643acc225dec05"),
            # it's necessary to turn off turn detection in the OpenAI Realtime API in order to use
            # LiveKit's turn detection model
            # turn_detection=None,
            # input_audio_transcription=None,  # we use Deepgram STT instead
        ),
        # To use Gemini Live API
        # llm=google.realtime.RealtimeModel(
        #     realtime_input_config=types.RealtimeInputConfig(
        #         automatic_activity_detection=types.AutomaticActivityDetection(
        #             disabled=True,
        #         ),
        #     ),
        #     input_audio_transcription=None,
        # ),
    )
    await session.start(agent=Agent(instructions="You are a helpful assistant."), room=ctx.room)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
