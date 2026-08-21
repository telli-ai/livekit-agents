# Copyright 2023 LiveKit, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import asyncio
import base64
import contextlib
import dataclasses
import json
import os
import time
import weakref
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, replace
from functools import cached_property
from typing import Any, Literal

import aiohttp

from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIError,
    APIStatusError,
    APITimeoutError,
    LanguageCode,
    tokenize,
    tts,
    utils,
)
from livekit.agents.tokenize.basic import split_words
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given
from livekit.agents.voice.io import TimedString

from ._utils import trace_id_from_headers
from .log import logger
from .models import TTSEncoding, TTSModels

# by default, use 22.05kHz sample rate at 32kbps
# in our testing,  reduce TTFB by about ~110ms
_DefaultEncoding: TTSEncoding = "mp3_22050_32"


def _sample_rate_from_format(output_format: TTSEncoding) -> int:
    split = output_format.split("_")  # e.g: mp3_44100
    return int(split[1])


def _encoding_to_mimetype(encoding: TTSEncoding) -> str:
    if encoding.startswith("mp3"):
        return "audio/mp3"
    elif encoding.startswith("opus"):
        return "audio/opus"
    elif encoding.startswith("pcm"):
        return "audio/pcm"
    else:
        raise ValueError(f"Unsupported encoding: {encoding}")


@dataclass
class VoiceSettings:
    stability: float  # [0.0 - 1.0]
    similarity_boost: float  # [0.0 - 1.0]
    style: NotGivenOr[float] = NOT_GIVEN  # [0.0 - 1.0]
    speed: NotGivenOr[float] = NOT_GIVEN  # [0.8 - 1.2]
    use_speaker_boost: NotGivenOr[bool] = NOT_GIVEN


@dataclass
class Voice:
    id: str
    name: str
    category: str


@dataclass
class PronunciationDictionaryLocator:
    pronunciation_dictionary_id: str
    version_id: str


DEFAULT_VOICE_ID = "hpp4J3VqNfWAUOO0d1Us"
API_BASE_URL_V1 = "https://api.elevenlabs.io/v1"
AUTHORIZATION_HEADER = "xi-api-key"
WS_INACTIVITY_TIMEOUT = 180
# the dialogue websocket closes after ~20s without client messages; keep_alive resets the timer
_DIALOGUE_KEEP_ALIVE_INTERVAL = 10


def _is_dialogue_model(model: TTSModels | str) -> bool:
    # eleven_v3 models are rejected by the multi-stream endpoint (403 at handshake)
    # and only stream over the text-to-dialogue websocket
    return str(model).startswith("eleven_v3")


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        voice_id: str = DEFAULT_VOICE_ID,
        voice_settings: NotGivenOr[VoiceSettings] = NOT_GIVEN,
        model: TTSModels | str = "eleven_turbo_v2_5",
        encoding: NotGivenOr[TTSEncoding] = NOT_GIVEN,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        base_url: NotGivenOr[str] = NOT_GIVEN,
        streaming_latency: NotGivenOr[int] = NOT_GIVEN,
        inactivity_timeout: int = WS_INACTIVITY_TIMEOUT,
        auto_mode: NotGivenOr[bool] = NOT_GIVEN,
        apply_text_normalization: Literal["auto", "off", "on"] = "auto",
        apply_language_text_normalization: NotGivenOr[bool] = NOT_GIVEN,
        word_tokenizer: NotGivenOr[tokenize.WordTokenizer | tokenize.SentenceTokenizer] = NOT_GIVEN,
        enable_ssml_parsing: bool = False,
        enable_logging: bool = True,
        chunk_length_schedule: NotGivenOr[list[int]] = NOT_GIVEN,  # range is [50, 500]
        http_session: aiohttp.ClientSession | None = None,
        language: NotGivenOr[str] = NOT_GIVEN,
        sync_alignment: bool = True,
        preferred_alignment: NotGivenOr[Literal["normalized", "original"]] = NOT_GIVEN,
        pronunciation_dictionary_locators: NotGivenOr[
            list[PronunciationDictionaryLocator]
        ] = NOT_GIVEN,
    ) -> None:
        """
        Create a new instance of ElevenLabs TTS.

        Args:
            voice_id (str): Voice ID. Defaults to `DEFAULT_VOICE_ID`.
            voice_settings (NotGivenOr[VoiceSettings]): Voice settings.
            model (TTSModels | str): TTS model to use. Defaults to "eleven_turbo_v2_5".
            api_key (NotGivenOr[str]): ElevenLabs API key. Can be set via argument or `ELEVEN_API_KEY` environment variable.
            base_url (NotGivenOr[str]): Custom base URL for the API. Optional.
            streaming_latency (NotGivenOr[int]): Optimize for streaming latency, defaults to 0 - disabled. 4 for max latency optimizations. deprecated
            inactivity_timeout (int): Inactivity timeout in seconds for the websocket connection. Defaults to 300.
            auto_mode (bool): Reduces latency by disabling chunk schedule and buffers.
                Sentence tokenizer will be used to synthesize one sentence at a time.
                Defaults to True unless ``chunk_length_schedule`` is provided.
            apply_text_normalization (Literal["auto", "off", "on"]): This parameter controls text normalization with three modes: ‘auto’, ‘on’, and ‘off’. When set to ‘auto’, the system will automatically decide whether to apply text normalization (e.g., spelling out numbers). With ‘on’, text normalization will always be applied, while with ‘off’, it will be skipped.
            apply_language_text_normalization (bool): This parameter controls language text normalization. This helps with proper pronunciation of text in some supported languages.
            word_tokenizer (NotGivenOr[tokenize.WordTokenizer | tokenize.SentenceTokenizer]): Tokenizer for processing text. Defaults to basic WordTokenizer when auto_mode=False, `livekit.agents.tokenize.blingfire.SentenceTokenizer` otherwise.
            enable_ssml_parsing (bool): Enable SSML parsing for input text. Defaults to False.
            enable_logging (bool): Enable logging of the request. When set to false, zero retention mode will be used. Defaults to True.
            chunk_length_schedule (NotGivenOr[list[int]]): Schedule for chunk lengths, ranging from 50 to 500. Defaults are [120, 160, 250, 290].
            http_session (aiohttp.ClientSession | None): Custom HTTP session for API requests. Optional.
            language (NotGivenOr[str]): Language code used to enforce a language for the model and text normalization. If the model does not support language overrides, it will be ignored.
            sync_alignment (bool): Enable sync alignment for the TTS model. Defaults to True.
            preferred_alignment (Literal["normalized", "original"]): Use normalized or original alignment. Defaults to "normalized", or "original" for CJK (ja, ko, zh) languages.
            pronunciation_dictionary_locators (NotGivenOr[list[PronunciationDictionaryLocator]]): List of pronunciation dictionary locators to use for pronunciation control.
        """  # noqa: E501

        if not is_given(encoding):
            encoding = _DefaultEncoding

        super().__init__(
            capabilities=tts.TTSCapabilities(
                streaming=True,
                aligned_transcript=sync_alignment,
            ),
            sample_rate=_sample_rate_from_format(encoding),
            num_channels=1,
        )

        elevenlabs_api_key = api_key if is_given(api_key) else os.environ.get("ELEVEN_API_KEY")
        if not elevenlabs_api_key:
            raise ValueError(
                "ElevenLabs API key is required, either as argument or set ELEVEN_API_KEY environmental variable"  # noqa: E501
            )

        if not is_given(auto_mode):
            auto_mode = not is_given(chunk_length_schedule)

        if not is_given(word_tokenizer):
            word_tokenizer = (
                tokenize.basic.WordTokenizer(ignore_punctuation=False)
                if not auto_mode
                else tokenize.blingfire.SentenceTokenizer()
            )
        elif auto_mode and not isinstance(word_tokenizer, tokenize.SentenceTokenizer):
            logger.warning(
                "auto_mode is enabled, it expects full sentences or phrases, "
                "please provide a SentenceTokenizer instead of a WordTokenizer."
            )

        self._opts = _TTSOptions(
            voice_id=voice_id,
            voice_settings=voice_settings,
            model=model,
            api_key=elevenlabs_api_key,
            base_url=base_url if is_given(base_url) else API_BASE_URL_V1,
            encoding=encoding,
            sample_rate=self.sample_rate,
            streaming_latency=streaming_latency,
            word_tokenizer=word_tokenizer,
            chunk_length_schedule=chunk_length_schedule,
            enable_ssml_parsing=enable_ssml_parsing,
            enable_logging=enable_logging,
            language=LanguageCode(language) if is_given(language) else NOT_GIVEN,
            inactivity_timeout=inactivity_timeout,
            sync_alignment=sync_alignment,
            auto_mode=auto_mode,
            apply_text_normalization=apply_text_normalization,
            apply_language_text_normalization=apply_language_text_normalization,
            preferred_alignment=preferred_alignment,
            pronunciation_dictionary_locators=pronunciation_dictionary_locators,
        )
        self._session = http_session
        self._streams = weakref.WeakSet[SynthesizeStream]()

        self.__current_connection: _Connection | _DialogueConnection | None = None
        self._connection_lock = asyncio.Lock()
        self._background_tasks: set[asyncio.Task[None]] = set()
        self._prewarm_task: asyncio.Task[None] | None = None
        self._closing = False

    @property
    def model(self) -> str:
        return self._opts.model

    @property
    def provider(self) -> str:
        return "ElevenLabs"

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()
        return self._session

    async def list_voices(self) -> list[Voice]:
        async with self._ensure_session().get(
            f"{self._opts.base_url}/voices",
            headers={AUTHORIZATION_HEADER: self._opts.api_key},
        ) as resp:
            return _dict_to_voices_list(await resp.json())

    def update_options(
        self,
        *,
        voice_id: NotGivenOr[str] = NOT_GIVEN,
        voice_settings: NotGivenOr[VoiceSettings] = NOT_GIVEN,
        model: NotGivenOr[TTSModels | str] = NOT_GIVEN,
        language: NotGivenOr[str] = NOT_GIVEN,
        pronunciation_dictionary_locators: NotGivenOr[
            list[PronunciationDictionaryLocator]
        ] = NOT_GIVEN,
    ) -> None:
        """
        Args:
            voice_id (NotGivenOr[str]): Voice ID.
            voice_settings (NotGivenOr[VoiceSettings]): Voice settings.
            model (NotGivenOr[TTSModels | str]): TTS model to use.
            language (NotGivenOr[str]): Language code for the TTS model.
            pronunciation_dictionary_locators (NotGivenOr[list[PronunciationDictionaryLocator]]): List of pronunciation dictionary locators.
        """
        changed = False

        if is_given(model) and model != self._opts.model:
            self._opts.model = model
            changed = True

        if is_given(voice_id) and voice_id != self._opts.voice_id:
            self._opts.voice_id = voice_id
            changed = True

        if is_given(voice_settings):
            self._opts.voice_settings = voice_settings
            changed = True

        if is_given(language):
            language = LanguageCode(language)
            if language != self._opts.language:
                self._opts.language = language
                changed = True

        if is_given(pronunciation_dictionary_locators):
            self._opts.pronunciation_dictionary_locators = pronunciation_dictionary_locators
            changed = True

        if changed and self.__current_connection:
            self.__current_connection.mark_non_current()
            self.__current_connection = None

    async def _current_connection(self) -> tuple[_Connection | _DialogueConnection, float, bool]:
        """Get the current connection, creating one if needed.

        Returns:
            Tuple of (connection, acquire_time, connection_reused)
        """
        async with self._connection_lock:
            if self._closing:
                raise APIConnectionError("TTS instance is closed", retryable=False)

            if (
                self.__current_connection
                and self.__current_connection.is_current
                and not self.__current_connection._closed
            ):
                return self.__current_connection, 0.0, True

            session = self._ensure_session()
            conn: _Connection | _DialogueConnection
            if _is_dialogue_model(self._opts.model):
                conn = _DialogueConnection(self._opts, session, spawn=self._spawn_background)
            else:
                conn = _Connection(self._opts, session)
            t0 = time.perf_counter()
            await conn.connect()
            acquire_time = time.perf_counter() - t0

            if self._closing:
                # aclose() may have run while the handshake was in flight; close the
                # fresh connection instead of publishing it on a closed instance
                await conn.aclose()
                raise APIConnectionError("TTS instance is closed", retryable=False)

            self.__current_connection = conn
            return conn, acquire_time, False

    def _discard_dialogue_connection(self, connection: _DialogueConnection) -> None:
        """Close a dialogue connection and warm a replacement in the background.

        Closing the socket is the only way to stop in-flight v3 synthesis, so this is
        the interruption path; the background reconnect keeps the next turn warm.
        """
        if self.__current_connection is connection:
            self.__current_connection = None
        connection.mark_non_current()  # spawns the close now that the turn is released
        if self._closing:
            # a stream woken by shutdown must not schedule a replacement connection
            return
        if self._prewarm_task is not None and not self._prewarm_task.done():
            self._prewarm_task.cancel()
        self._prewarm_task = asyncio.create_task(self._prewarm_dialogue())

    async def _prewarm_dialogue(self) -> None:
        with contextlib.suppress(Exception):
            await self._current_connection()

    def _spawn_background(self, coro: Coroutine[Any, Any, Any]) -> asyncio.Task[None]:
        task = asyncio.create_task(coro)
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        return task

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> ChunkedStream:
        return ChunkedStream(tts=self, input_text=text, conn_options=conn_options)

    def stream(
        self, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> SynthesizeStream:
        stream = SynthesizeStream(tts=self, conn_options=conn_options)
        self._streams.add(stream)
        return stream

    async def aclose(self) -> None:
        self._closing = True

        for stream in list(self._streams):
            await stream.aclose()
        self._streams.clear()

        if self._prewarm_task is not None:
            self._prewarm_task.cancel()

        # connection close tasks must complete, not be cancelled: a close interrupted
        # after it marks the connection closed would leave the websocket and its
        # recv/keep-alive tasks alive past shutdown
        if self._background_tasks:
            await asyncio.gather(*list(self._background_tasks), return_exceptions=True)

        if self.__current_connection:
            await self.__current_connection.aclose()
            self.__current_connection = None


class ChunkedStream(tts.ChunkedStream):
    """Synthesize using the chunked api endpoint"""

    def __init__(self, *, tts: TTS, input_text: str, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        # route on the snapshotted model so one request never mixes protocols when
        # update_options() switches the model family concurrently
        if _is_dialogue_model(self._opts.model):
            await self._run_dialogue(output_emitter)
            return

        voice_settings = (
            _strip_nones(dataclasses.asdict(self._opts.voice_settings))
            if is_given(self._opts.voice_settings)
            else None
        )
        extra_params: dict[str, str | bool] = {}
        if is_given(self._opts.language):
            extra_params["language_code"] = self._opts.language.language
        if is_given(self._opts.apply_language_text_normalization):
            extra_params["apply_language_text_normalization"] = (
                self._opts.apply_language_text_normalization
            )
        try:
            async with self._tts._ensure_session().post(
                _synthesize_url(self._opts),
                headers={AUTHORIZATION_HEADER: self._opts.api_key},
                json={
                    "text": self._input_text,
                    "model_id": self._opts.model,
                    "voice_settings": voice_settings,
                    "apply_text_normalization": self._opts.apply_text_normalization,
                    **extra_params,
                },
                timeout=aiohttp.ClientTimeout(
                    total=30,
                    sock_connect=self._conn_options.timeout,
                ),
            ) as resp:
                resp.raise_for_status()

                if not resp.content_type.startswith("audio/"):
                    content = await resp.text()
                    raise APIError(message="11labs returned non-audio data", body=content)

                output_emitter.initialize(
                    request_id=utils.shortuuid(),
                    sample_rate=self._opts.sample_rate,
                    num_channels=1,
                    mime_type=_encoding_to_mimetype(self._opts.encoding),
                )

                async for data, _ in resp.content.iter_chunks():
                    output_emitter.push(data)

                output_emitter.flush()

        except asyncio.TimeoutError as e:
            raise APITimeoutError() from e
        except aiohttp.ClientResponseError as e:
            raise APIStatusError(
                message=e.message,
                status_code=e.status,
                request_id=trace_id_from_headers(e.headers),
                body=None,
            ) from e
        except Exception as e:
            raise APIConnectionError() from e

    async def _run_dialogue(self, output_emitter: tts.AudioEmitter) -> None:
        """Synthesize via the dialogue websocket as a single one-shot turn"""
        connection, _, _ = await _acquire_dialogue_connection(self._tts, self._conn_options)

        output_emitter.initialize(
            request_id=utils.shortuuid(),
            sample_rate=self._opts.sample_rate,
            num_channels=1,
            mime_type=_encoding_to_mimetype(self._opts.encoding),
        )

        waiter: asyncio.Future[None] = asyncio.get_event_loop().create_future()
        turn = await connection.start_turn(
            emitter=output_emitter, stream=None, waiter=waiter, timeout=self._conn_options.timeout
        )

        clean = False
        try:
            if self._input_text:
                await connection.send_text(turn, self._input_text)
            await connection.end_turn_input(turn)
            await waiter
            output_emitter.flush()
            clean = True
        except asyncio.TimeoutError as e:
            raise APITimeoutError() from e
        except APIError:
            raise
        except Exception as e:
            raise APIConnectionError("elevenlabs dialogue synthesis failed") from e
        finally:
            connection.finish_turn(turn)
            if not clean:
                self._tts._discard_dialogue_connection(connection)


class SynthesizeStream(tts.SynthesizeStream):
    """Streamed API using websockets

    Uses multi-stream API:
    https://elevenlabs.io/docs/api-reference/text-to-speech/v-1-text-to-speech-voice-id-multi-stream-input
    """

    def __init__(self, *, tts: TTS, conn_options: APIConnectOptions):
        super().__init__(tts=tts, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)
        self._context_id = ""
        self._text_buffer = ""
        self._start_times_ms: list[int] = []
        self._durations_ms: list[int] = []
        self._connection: _Connection | None = None

    async def aclose(self) -> None:
        await super().aclose()

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        # route on the snapshotted model so one request never mixes protocols when
        # update_options() switches the model family concurrently
        if _is_dialogue_model(self._opts.model):
            await self._run_dialogue(output_emitter)
            return

        self._context_id = utils.shortuuid()
        self._text_buffer = ""
        self._start_times_ms = []
        self._durations_ms = []

        sent_tokenizer_stream = self._opts.word_tokenizer.stream()

        output_emitter.initialize(
            request_id=self._context_id,
            sample_rate=self._opts.sample_rate,
            num_channels=1,
            stream=True,
            mime_type=_encoding_to_mimetype(self._opts.encoding),
        )
        output_emitter.start_segment(segment_id=self._context_id)

        try:
            conn, self._acquire_time, self._connection_reused = await asyncio.wait_for(
                self._tts._current_connection(), self._conn_options.timeout
            )
        except asyncio.TimeoutError as e:
            raise APITimeoutError() from e
        except aiohttp.WSServerHandshakeError as e:
            raise APIStatusError(
                message=e.message,
                status_code=e.status,
                request_id=trace_id_from_headers(e.headers),
            ) from e
        except Exception as e:
            raise APIConnectionError("could not connect to ElevenLabs") from e

        if not isinstance(conn, _Connection):
            # update_options() switched the model family while this request was starting; the
            # request keeps its snapshotted model, so retrying can never match - fail fast
            raise APIConnectionError(
                "model family changed while starting synthesis; create a new stream after "
                "switching between eleven_v3 and other models",
                retryable=False,
            )
        connection: _Connection = conn

        waiter: asyncio.Future[None] = asyncio.get_event_loop().create_future()
        connection.register_stream(self, output_emitter, waiter)
        context_closed = False

        async def _input_task() -> None:
            async for data in self._input_ch:
                if isinstance(data, self._FlushSentinel):
                    sent_tokenizer_stream.flush()
                    continue
                sent_tokenizer_stream.push_text(data)
            sent_tokenizer_stream.end_input()

        async def _sentence_stream_task() -> None:
            nonlocal context_closed

            flush_on_chunk = (
                isinstance(self._opts.word_tokenizer, tokenize.SentenceTokenizer)
                and is_given(self._opts.auto_mode)
                and self._opts.auto_mode
            )
            xml_content: list[str] = []
            async for data in sent_tokenizer_stream:
                text = data.token
                # send xml tags fully formed
                xml_start_tokens = ["<phoneme", "<break"]
                xml_end_tokens = ["</phoneme>", "/>"]

                if (
                    self._opts.enable_ssml_parsing
                    and any(text.startswith(start) for start in xml_start_tokens)
                    or xml_content
                ):
                    xml_content.append(text)

                    if any(text.find(end) > -1 for end in xml_end_tokens):
                        text = (
                            self._opts.word_tokenizer.format_words(xml_content)
                            if isinstance(self._opts.word_tokenizer, tokenize.WordTokenizer)
                            else " ".join(xml_content)
                        )
                        xml_content = []
                    else:
                        continue

                formatted_text = f"{text} "  # must always end with a space
                # when using auto_mode, we are flushing for each sentence
                connection.send_content(
                    _SynthesizeContent(self._context_id, formatted_text, flush=flush_on_chunk)
                )
                self._mark_started()

            if xml_content:
                logger.warning("ElevenLabs stream ended with incomplete xml content")

            connection.send_content(_SynthesizeContent(self._context_id, "", flush=True))
            connection.close_context(self._context_id)
            context_closed = True

        input_t = asyncio.create_task(_input_task())
        stream_t = asyncio.create_task(_sentence_stream_task())

        try:
            await waiter
        except asyncio.TimeoutError as e:
            raise APITimeoutError() from e
        except Exception as e:
            if isinstance(e, APIStatusError):
                raise e
            raise APIStatusError("Could not synthesize") from e
        finally:
            output_emitter.end_segment()
            await utils.aio.gracefully_cancel(input_t, stream_t)
            if not context_closed:
                with contextlib.suppress(Exception):
                    connection.close_context(self._context_id)
            await sent_tokenizer_stream.aclose()

    async def _run_dialogue(self, output_emitter: tts.AudioEmitter) -> None:
        """Stream via the dialogue websocket: one synthesis turn on the shared connection"""
        request_id = utils.shortuuid()
        self._text_buffer = ""
        self._start_times_ms = []
        self._durations_ms = []

        sent_tokenizer_stream = self._opts.word_tokenizer.stream()

        output_emitter.initialize(
            request_id=request_id,
            sample_rate=self._opts.sample_rate,
            num_channels=1,
            stream=True,
            mime_type=_encoding_to_mimetype(self._opts.encoding),
        )
        output_emitter.start_segment(segment_id=request_id)

        connection: _DialogueConnection
        (
            connection,
            self._acquire_time,
            self._connection_reused,
        ) = await _acquire_dialogue_connection(self._tts, self._conn_options)

        waiter: asyncio.Future[None] = asyncio.get_event_loop().create_future()
        turn = await connection.start_turn(
            emitter=output_emitter, stream=self, waiter=waiter, timeout=self._conn_options.timeout
        )

        async def _input_task() -> None:
            async for data in self._input_ch:
                if isinstance(data, self._FlushSentinel):
                    sent_tokenizer_stream.flush()
                    continue
                sent_tokenizer_stream.push_text(data)
            sent_tokenizer_stream.end_input()

        async def _sentence_stream_task() -> None:
            # the server streams naturally once ~40 chars are buffered; an explicit
            # flush is only needed at the end of input to force out the short tail
            async for data in sent_tokenizer_stream:
                await connection.send_text(turn, f"{data.token} ")
                self._mark_started()
            await connection.end_turn_input(turn)

        input_t = asyncio.create_task(_input_task())
        stream_t = asyncio.create_task(_sentence_stream_task())

        clean = False
        try:
            await waiter
            clean = True
        except asyncio.TimeoutError as e:
            raise APITimeoutError() from e
        except APIError:
            raise
        except Exception as e:
            raise APIStatusError("Could not synthesize") from e
        finally:
            output_emitter.end_segment()
            await utils.aio.gracefully_cancel(input_t, stream_t)
            await sent_tokenizer_stream.aclose()
            connection.finish_turn(turn)
            if not clean:
                self._tts._discard_dialogue_connection(connection)


@dataclass
class _TTSOptions:
    api_key: str
    voice_id: str
    voice_settings: NotGivenOr[VoiceSettings]
    model: TTSModels | str
    language: NotGivenOr[LanguageCode]
    base_url: str
    encoding: TTSEncoding
    sample_rate: int
    streaming_latency: NotGivenOr[int]
    word_tokenizer: tokenize.WordTokenizer | tokenize.SentenceTokenizer
    chunk_length_schedule: NotGivenOr[list[int]]
    enable_ssml_parsing: bool
    enable_logging: bool
    inactivity_timeout: int
    sync_alignment: bool
    apply_text_normalization: Literal["auto", "on", "off"]
    apply_language_text_normalization: NotGivenOr[bool]
    preferred_alignment: NotGivenOr[Literal["normalized", "original"]]
    auto_mode: NotGivenOr[bool]
    pronunciation_dictionary_locators: NotGivenOr[list[PronunciationDictionaryLocator]]


def _build_context_init_packet(opts: _TTSOptions, *, context_id: str) -> dict[str, Any]:
    voice_settings = (
        _strip_nones(dataclasses.asdict(opts.voice_settings))
        if is_given(opts.voice_settings)
        else {}
    )
    init_pkt: dict[str, Any] = {
        "text": " ",
        "voice_settings": voice_settings,
        "context_id": context_id,
    }
    if is_given(opts.chunk_length_schedule):
        init_pkt["generation_config"] = {
            "chunk_length_schedule": opts.chunk_length_schedule,
        }
    if is_given(opts.pronunciation_dictionary_locators):
        init_pkt["pronunciation_dictionary_locators"] = [
            {
                "pronunciation_dictionary_id": locator.pronunciation_dictionary_id,
                "version_id": locator.version_id,
            }
            for locator in opts.pronunciation_dictionary_locators
        ]
    return init_pkt


@dataclass
class _SynthesizeContent:
    context_id: str
    text: str
    flush: bool = False


@dataclass
class _CloseContext:
    context_id: str


@dataclass
class _StreamData:
    emitter: tts.AudioEmitter
    stream: SynthesizeStream
    waiter: asyncio.Future[None]
    timeout_timer: asyncio.TimerHandle | None = None


class _Connection:
    """Manages a single WebSocket connection with send/recv loops for multi-context TTS"""

    def __init__(self, opts: _TTSOptions, session: aiohttp.ClientSession):
        self._opts = opts
        self._session = session
        self._ws: aiohttp.ClientWebSocketResponse | None = None
        self._is_current = True
        self._active_contexts: set[str] = set()
        self._input_queue = utils.aio.Chan[_SynthesizeContent | _CloseContext]()

        self._context_data: dict[str, _StreamData] = {}

        self._send_task: asyncio.Task | None = None
        self._recv_task: asyncio.Task | None = None
        self._closed = False

    @property
    def voice_id(self) -> str:
        return self._opts.voice_id

    @property
    def is_current(self) -> bool:
        return self._is_current

    @cached_property
    def preferred_alignment(self) -> Literal["normalized", "original"]:
        return _resolve_preferred_alignment(self._opts)

    def mark_non_current(self) -> None:
        """Mark this connection as no longer current - it will shut down when drained"""
        self._is_current = False

    async def connect(self) -> None:
        """Establish WebSocket connection and start send/recv loops"""
        if self._ws or self._closed:
            return

        url = _multi_stream_url(self._opts)
        headers = {AUTHORIZATION_HEADER: self._opts.api_key}
        self._ws = await self._session.ws_connect(url, headers=headers)

        self._send_task = asyncio.create_task(self._send_loop())
        self._recv_task = asyncio.create_task(self._recv_loop())

    def register_stream(
        self, stream: SynthesizeStream, emitter: tts.AudioEmitter, done_fut: asyncio.Future[None]
    ) -> None:
        """Register a new synthesis stream with this connection"""
        context_id = stream._context_id
        self._context_data[context_id] = _StreamData(
            emitter=emitter, stream=stream, waiter=done_fut
        )

    def send_content(self, content: _SynthesizeContent) -> None:
        """Send synthesis content to the connection"""
        if self._closed or not self._ws or self._ws.closed:
            raise APIConnectionError("WebSocket connection is closed")
        self._input_queue.send_nowait(content)

    def close_context(self, context_id: str) -> None:
        """Close a specific context"""
        if self._closed or not self._ws or self._ws.closed:
            raise APIConnectionError("WebSocket connection is closed")
        self._input_queue.send_nowait(_CloseContext(context_id))

    async def _send_loop(self) -> None:
        """Send loop - processes messages from input queue"""
        try:
            while not self._closed:
                try:
                    msg = await self._input_queue.recv()
                except utils.aio.ChanClosed:
                    break

                if not self._ws or self._ws.closed:
                    break

                if isinstance(msg, _SynthesizeContent):
                    is_new_context = msg.context_id not in self._active_contexts

                    if is_new_context:
                        init_pkt = _build_context_init_packet(
                            self._opts,
                            context_id=msg.context_id,
                        )
                        await self._ws.send_json(init_pkt)
                        self._active_contexts.add(msg.context_id)

                    pkt: dict[str, Any] = {
                        "text": msg.text,
                        "context_id": msg.context_id,
                    }
                    if msg.flush:
                        pkt["flush"] = True

                    # start timeout timer for this context
                    self._start_timeout_timer(msg.context_id)

                    await self._ws.send_json(pkt)

                elif isinstance(msg, _CloseContext):
                    if msg.context_id in self._active_contexts:
                        close_pkt = {
                            "context_id": msg.context_id,
                            "close_context": True,
                        }
                        await self._ws.send_json(close_pkt)

        except Exception as e:
            logger.warning("send loop error", exc_info=e)
        finally:
            if not self._closed:
                await self.aclose()

    async def _recv_loop(self) -> None:
        """Receive loop - processes messages from WebSocket"""
        try:
            while not self._closed and self._ws and not self._ws.closed:
                msg = await self._ws.receive()

                if msg.type in (
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    if not self._closed and len(self._context_data) > 0:
                        # websocket will be closed after all contexts are closed
                        raise APIStatusError(
                            "ElevenLabs websocket connection closed unexpectedly",
                            status_code=self._ws.close_code or -1,
                        )
                    break

                if msg.type != aiohttp.WSMsgType.TEXT:
                    logger.warning("unexpected message type %s", msg.type)
                    continue

                data = json.loads(msg.data)
                # ElevenLabs currently sends snake_case context IDs on the websocket API,
                # while older responses and some examples use camelCase.
                context_id = data.get("contextId") or data.get("context_id")
                ctx = self._context_data.get(context_id) if context_id is not None else None

                if error := data.get("error"):
                    logger.error(
                        "elevenlabs tts returned error",
                        extra={"context_id": context_id, "error": error, "data": data},
                    )
                    if context_id is not None:
                        if ctx and not ctx.waiter.done():
                            ctx.waiter.set_exception(APIError(message=error))
                        self._cleanup_context(context_id)
                    continue

                if ctx is None:
                    if data.get("type") == "flush_done":
                        logger.debug(
                            "ignoring elevenlabs flush_done message for inactive context",
                            extra={"context_id": context_id, "data": data},
                        )
                        continue

                    logger.warning(
                        "unexpected message received from elevenlabs tts", extra={"data": data}
                    )
                    continue

                emitter = ctx.emitter
                stream = ctx.stream

                # ensure alignment
                alignment = (
                    data.get("normalizedAlignment")
                    if self.preferred_alignment == "normalized"
                    else data.get("alignment")
                )
                if alignment and stream is not None:
                    chars = alignment["chars"]
                    starts = alignment.get("charStartTimesMs") or alignment.get("charsStartTimesMs")
                    durs = alignment.get("charDurationsMs") or alignment.get("charsDurationsMs")
                    if starts and durs and len(chars) == len(durs) and len(starts) == len(durs):
                        stream._text_buffer += "".join(chars)
                        # in case item in chars has multiple characters
                        for char, start, dur in zip(chars, starts, durs, strict=False):
                            if len(char) > 1:
                                stream._start_times_ms += [start] * (len(char) - 1)
                                stream._durations_ms += [0] * (len(char) - 1)
                            stream._start_times_ms.append(start)
                            stream._durations_ms.append(dur)

                        timed_words, stream._text_buffer = _to_timed_words(
                            stream._text_buffer, stream._start_times_ms, stream._durations_ms
                        )
                        emitter.push_timed_transcript(timed_words)
                        stream._start_times_ms = stream._start_times_ms[-len(stream._text_buffer) :]
                        stream._durations_ms = stream._durations_ms[-len(stream._text_buffer) :]

                if data.get("audio"):
                    b64data = base64.b64decode(data["audio"])
                    emitter.push(b64data)
                    if ctx.timeout_timer:
                        ctx.timeout_timer.cancel()

                if data.get("isFinal"):
                    if stream is not None:
                        timed_words, _ = _to_timed_words(
                            stream._text_buffer,
                            stream._start_times_ms,
                            stream._durations_ms,
                            flush=True,
                        )
                        emitter.push_timed_transcript(timed_words)

                    if not ctx.waiter.done():
                        ctx.waiter.set_result(None)
                    self._cleanup_context(context_id)

                    if not self._is_current and not self._active_contexts:
                        logger.debug("no active contexts, shutting down connection")
                        break
        except Exception as e:
            logger.warning("recv loop error", exc_info=e)
            for ctx in self._context_data.values():
                if not ctx.waiter.done():
                    ctx.waiter.set_exception(e)
                if ctx.timeout_timer:
                    ctx.timeout_timer.cancel()
            self._context_data.clear()
        finally:
            if not self._closed:
                await self.aclose()

    def _cleanup_context(self, context_id: str) -> None:
        """Clean up context state"""
        ctx = self._context_data.pop(context_id, None)
        if ctx and ctx.timeout_timer:
            ctx.timeout_timer.cancel()

        self._active_contexts.discard(context_id)

    def _start_timeout_timer(self, context_id: str) -> None:
        """Start a timeout timer for a context"""
        if not (ctx := self._context_data.get(context_id)) or ctx.timeout_timer:
            return

        timeout = ctx.stream._conn_options.timeout

        def _on_timeout() -> None:
            if not ctx.waiter.done():
                ctx.waiter.set_exception(
                    APITimeoutError(f"11labs tts timed out after {timeout} seconds")
                )
            self._cleanup_context(context_id)

        ctx.timeout_timer = asyncio.get_event_loop().call_later(timeout, _on_timeout)

    async def aclose(self) -> None:
        """Close the connection and clean up"""
        if self._closed:
            return

        self._closed = True
        self._input_queue.close()

        for ctx in self._context_data.values():
            if not ctx.waiter.done():
                # do not cancel the future as it becomes difficult to catch
                # all pending tasks will be aborted with an exception
                ctx.waiter.set_exception(APIStatusError("connection closed"))
            if ctx.timeout_timer:
                ctx.timeout_timer.cancel()
        self._context_data.clear()

        if self._ws:
            await self._ws.close()

        if self._send_task:
            await utils.aio.gracefully_cancel(self._send_task)
        if self._recv_task:
            await utils.aio.gracefully_cancel(self._recv_task)

        self._ws = None


@dataclass
class _DialogueTurn:
    """State for the single active synthesis turn on a dialogue connection"""

    emitter: tts.AudioEmitter
    stream: SynthesizeStream | None
    waiter: asyncio.Future[None]
    voice_id: str
    timeout: float
    flushes_sent: int = 0
    markers_received: int = 0
    dirty: bool = False
    started_input: bool = False
    input_done: bool = False
    timeout_timer: asyncio.TimerHandle | None = None


class _DialogueConnection:
    """Manages a single text-to-dialogue WebSocket for eleven_v3 models.

    The dialogue endpoint has no per-context multiplexing and no cancellation
    message: a new turn never stops in-flight synthesis, only closing the socket
    does. The server emits exactly one ``is_final_audio_for_turn`` per flush that
    had pending text, so a turn is complete when its flush count is matched. The
    connection is reused across consecutive synthesis turns and kept open with
    ``keep_alive`` messages (the server idles out after ~20s otherwise);
    interruption closes the socket (see ``TTS._discard_dialogue_connection``).
    """

    def __init__(
        self,
        opts: _TTSOptions,
        session: aiohttp.ClientSession,
        *,
        spawn: Callable[[Coroutine[Any, Any, Any]], asyncio.Task[None]] | None = None,
    ):
        self._opts = opts
        self._session = session
        # owner-provided task spawner so close tasks can be awaited at TTS shutdown
        self._spawn = spawn
        self._ws: aiohttp.ClientWebSocketResponse | None = None
        self._is_current = True
        self._closed = False
        self._turn: _DialogueTurn | None = None
        self._turn_lock = asyncio.Lock()
        self._send_lock = asyncio.Lock()
        self._recv_task: asyncio.Task[None] | None = None
        self._keepalive_task: asyncio.Task[None] | None = None
        self._close_task: asyncio.Task[None] | None = None
        self._last_activity = 0.0

    @property
    def is_current(self) -> bool:
        return self._is_current

    @cached_property
    def preferred_alignment(self) -> Literal["normalized", "original"]:
        return _resolve_preferred_alignment(self._opts)

    def mark_non_current(self) -> None:
        """Mark this connection as superseded; it closes once idle so sockets don't linger"""
        self._is_current = False
        if self._turn is None:
            self._spawn_close()

    def _spawn_close(self) -> None:
        if self._closed or (self._close_task is not None and not self._close_task.done()):
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return  # no running loop; the keep-alive loop closes it instead
        coro = self.aclose()
        self._close_task = self._spawn(coro) if self._spawn is not None else loop.create_task(coro)

    async def connect(self) -> None:
        """Establish the WebSocket, send the init message, and start recv/keep-alive loops"""
        if self._ws or self._closed:
            return

        url = _dialogue_stream_url(self._opts)
        headers = {AUTHORIZATION_HEADER: self._opts.api_key}
        self._ws = await self._session.ws_connect(url, headers=headers)
        await self._ws.send_json(_build_dialogue_init_packet(self._opts))
        self._last_activity = asyncio.get_event_loop().time()

        self._recv_task = asyncio.create_task(self._recv_loop())
        self._keepalive_task = asyncio.create_task(self._keepalive_loop())

    async def start_turn(
        self,
        *,
        emitter: tts.AudioEmitter,
        stream: SynthesizeStream | None,
        waiter: asyncio.Future[None],
        timeout: float,
    ) -> _DialogueTurn:
        """Acquire the connection for one synthesis turn (turns are serialized)"""
        await self._turn_lock.acquire()
        if self._closed or not self._ws or self._ws.closed:
            self._turn_lock.release()
            raise APIConnectionError("dialogue websocket connection is closed")

        turn = _DialogueTurn(
            emitter=emitter,
            stream=stream,
            waiter=waiter,
            voice_id=self._opts.voice_id,
            timeout=timeout,
        )
        self._turn = turn
        self._last_activity = asyncio.get_event_loop().time()
        return turn

    async def send_text(self, turn: _DialogueTurn, text: str) -> None:
        pkt = {
            "inputs": [
                {"text": text, "voice_id": turn.voice_id, "new_turn": not turn.started_input}
            ]
        }
        turn.started_input = True
        turn.dirty = True
        self._start_timeout_timer(turn)
        await self._send_json(pkt)

    async def flush_turn(self, turn: _DialogueTurn) -> None:
        # a flush with no pending text produces no marker; skip it so the
        # marker count stays exact
        if not turn.dirty:
            return
        turn.dirty = False
        turn.flushes_sent += 1
        await self._send_json({"flush": True})

    async def end_turn_input(self, turn: _DialogueTurn) -> None:
        await self.flush_turn(turn)
        turn.input_done = True
        self._maybe_complete_turn(turn)
        if not turn.waiter.done():
            # from here on audio must keep flowing until the final marker; without this
            # the client keep_alive defeats the server idle close and a stall would
            # hang the turn forever
            self._arm_stall_timer(turn)

    def finish_turn(self, turn: _DialogueTurn) -> None:
        """Release the connection; always called by the turn holder when its run ends"""
        if self._turn is turn:
            self._turn = None
        if turn.timeout_timer:
            turn.timeout_timer.cancel()
        self._last_activity = asyncio.get_event_loop().time()
        if self._turn_lock.locked():
            self._turn_lock.release()
        if not self._is_current:
            self._spawn_close()

    async def _send_json(self, data: dict[str, Any]) -> None:
        if self._closed or not self._ws or self._ws.closed:
            raise APIConnectionError("dialogue websocket connection is closed")
        async with self._send_lock:
            await self._ws.send_json(data)

    def _maybe_complete_turn(self, turn: _DialogueTurn) -> None:
        if not turn.input_done or turn.markers_received < turn.flushes_sent:
            return
        if turn.waiter.done():
            return

        if turn.stream is not None:
            timed_words, _ = _to_timed_words(
                turn.stream._text_buffer,
                turn.stream._start_times_ms,
                turn.stream._durations_ms,
                flush=True,
            )
            turn.emitter.push_timed_transcript(timed_words)

        turn.waiter.set_result(None)
        if turn.timeout_timer:
            turn.timeout_timer.cancel()
        self._turn = None
        self._last_activity = asyncio.get_event_loop().time()

    def _start_timeout_timer(self, turn: _DialogueTurn) -> None:
        """Time out if no audio arrives for the turn; cancelled on first audio"""
        if turn.timeout_timer:
            return

        def _on_timeout() -> None:
            if not turn.waiter.done():
                turn.waiter.set_exception(
                    APITimeoutError(f"11labs tts timed out after {turn.timeout} seconds")
                )

        turn.timeout_timer = asyncio.get_event_loop().call_later(turn.timeout, _on_timeout)

    def _arm_stall_timer(self, turn: _DialogueTurn) -> None:
        """After input ends, time out if audio stops flowing before the final marker"""
        if turn.timeout_timer:
            turn.timeout_timer.cancel()

        def _on_stall() -> None:
            if not turn.waiter.done():
                turn.waiter.set_exception(
                    APITimeoutError(
                        f"11labs tts stalled after input end ({turn.timeout} seconds without audio)"
                    )
                )

        turn.timeout_timer = asyncio.get_event_loop().call_later(turn.timeout, _on_stall)

    async def _recv_loop(self) -> None:
        """Receive loop - routes audio, alignment, and turn markers to the active turn"""
        try:
            while not self._closed and self._ws and not self._ws.closed:
                msg = await self._ws.receive()

                if msg.type in (
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    turn = self._turn
                    if turn is not None and not turn.waiter.done():
                        raise APIStatusError(
                            "ElevenLabs dialogue websocket closed unexpectedly",
                            status_code=self._ws.close_code or -1,
                        )
                    break

                if msg.type != aiohttp.WSMsgType.TEXT:
                    logger.warning("unexpected message type %s", msg.type)
                    continue

                data = json.loads(msg.data)
                turn = self._turn

                if error := (data.get("error") or data.get("message")):
                    if turn is None and data.get("error") == "input_timeout_exceeded":
                        # expected when the connection idles out between turns
                        logger.debug("elevenlabs dialogue connection idled out")
                        break

                    logger.error(
                        "elevenlabs dialogue tts returned error",
                        extra={"error": error, "data": data},
                    )
                    if turn is not None and not turn.waiter.done():
                        turn.waiter.set_exception(
                            APIStatusError(str(error), status_code=data.get("code") or -1)
                        )
                    break

                if data.get("audio"):
                    if turn is None:
                        # with exact flush accounting this should not happen; treat the
                        # connection as poisoned rather than risk emitting stale audio
                        logger.warning(
                            "elevenlabs dialogue tts sent audio with no active turn, "
                            "closing connection"
                        )
                        break
                    turn.emitter.push(base64.b64decode(data["audio"]))
                    if turn.timeout_timer:
                        turn.timeout_timer.cancel()
                    if turn.input_done and not turn.waiter.done():
                        self._arm_stall_timer(turn)

                if turn is not None and turn.stream is not None:
                    alignment = (
                        data.get("normalized_alignment") or data.get("normalizedAlignment")
                        if self.preferred_alignment == "normalized"
                        else data.get("alignment")
                    )
                    if alignment:
                        _push_dialogue_alignment(turn.stream, turn.emitter, alignment)

                if data.get("is_final_audio_for_turn") and turn is not None:
                    turn.markers_received += 1
                    self._maybe_complete_turn(turn)

                if data.get("is_final"):
                    break
        except Exception as e:
            turn = self._turn
            if turn is not None and not turn.waiter.done():
                turn.waiter.set_exception(e)
            self._turn = None
        finally:
            if not self._closed:
                await self.aclose()

    async def _keepalive_loop(self) -> None:
        """Keep the socket alive between turns; drop it after ``inactivity_timeout`` idle"""
        try:
            while not self._closed and self._ws and not self._ws.closed:
                await asyncio.sleep(_DIALOGUE_KEEP_ALIVE_INTERVAL)
                if self._turn is None and not self._is_current:
                    break
                if (
                    self._turn is None
                    and asyncio.get_event_loop().time() - self._last_activity
                    >= self._opts.inactivity_timeout
                ):
                    logger.debug("closing idle elevenlabs dialogue connection")
                    self.mark_non_current()
                    break
                try:
                    await self._send_json({"keep_alive": True})
                except APIConnectionError:
                    break
        finally:
            if not self._closed:
                await self.aclose()

    async def aclose(self) -> None:
        """Close the connection and clean up"""
        if self._closed:
            return

        self._closed = True

        turn = self._turn
        if turn is not None:
            if not turn.waiter.done():
                turn.waiter.set_exception(APIStatusError("connection closed"))
            if turn.timeout_timer:
                turn.timeout_timer.cancel()
        self._turn = None

        if self._ws:
            await self._ws.close()

        current = asyncio.current_task()
        for task in (self._recv_task, self._keepalive_task):
            if task is not None and task is not current:
                await utils.aio.gracefully_cancel(task)

        self._ws = None


async def _acquire_dialogue_connection(
    tts_inst: TTS, conn_options: APIConnectOptions
) -> tuple[_DialogueConnection, float, bool]:
    try:
        connection, acquire_time, reused = await asyncio.wait_for(
            tts_inst._current_connection(), conn_options.timeout
        )
    except asyncio.TimeoutError as e:
        raise APITimeoutError() from e
    except aiohttp.WSServerHandshakeError as e:
        raise APIStatusError(
            message=e.message,
            status_code=e.status,
            request_id=trace_id_from_headers(e.headers),
        ) from e
    except Exception as e:
        raise APIConnectionError("could not connect to ElevenLabs") from e

    if not isinstance(connection, _DialogueConnection):
        # update_options() switched the model family while this request was starting; the
        # request keeps its snapshotted model, so retrying can never match - fail fast
        raise APIConnectionError(
            "model family changed while starting synthesis; create a new stream after "
            "switching between eleven_v3 and other models",
            retryable=False,
        )
    return connection, acquire_time, reused


def _dict_to_voices_list(data: dict[str, Any]) -> list[Voice]:
    voices: list[Voice] = []
    for voice in data["voices"]:
        voices.append(Voice(id=voice["voice_id"], name=voice["name"], category=voice["category"]))

    return voices


def _strip_nones(data: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in data.items() if is_given(v) and v is not None}


def _synthesize_url(opts: _TTSOptions) -> str:
    base_url = opts.base_url
    voice_id = opts.voice_id
    output_format = opts.encoding
    url = (
        f"{base_url}/text-to-speech/{voice_id}/stream?"
        f"output_format={output_format}&enable_logging={str(opts.enable_logging).lower()}"
    )
    if is_given(opts.streaming_latency):
        url += f"&optimize_streaming_latency={opts.streaming_latency}"
    return url


def _multi_stream_url(opts: _TTSOptions) -> str:
    base_url = opts.base_url.replace("https://", "wss://").replace("http://", "ws://")
    voice_id = opts.voice_id
    url = f"{base_url}/text-to-speech/{voice_id}/multi-stream-input?"
    params = []
    params.append(f"model_id={opts.model}")
    params.append(f"output_format={opts.encoding}")
    if is_given(opts.language):
        params.append(f"language_code={opts.language.language}")
    params.append(f"enable_ssml_parsing={str(opts.enable_ssml_parsing).lower()}")
    params.append(f"enable_logging={str(opts.enable_logging).lower()}")
    params.append(f"inactivity_timeout={opts.inactivity_timeout}")
    params.append(f"apply_text_normalization={opts.apply_text_normalization}")
    if is_given(opts.apply_language_text_normalization):
        params.append(
            f"apply_language_text_normalization={str(opts.apply_language_text_normalization).lower()}"
        )
    if opts.sync_alignment:
        params.append("sync_alignment=true")
    if is_given(opts.auto_mode):
        params.append(f"auto_mode={str(opts.auto_mode).lower()}")
    url += "&".join(params)
    return url


def _dialogue_stream_url(opts: _TTSOptions) -> str:
    base_url = opts.base_url.replace("https://", "wss://").replace("http://", "ws://")
    params = [
        f"model_id={opts.model}",
        f"output_format={opts.encoding}",
        f"apply_text_normalization={opts.apply_text_normalization}",
        f"enable_logging={str(opts.enable_logging).lower()}",
    ]
    if is_given(opts.language):
        params.append(f"language_code={opts.language.language}")
    if opts.sync_alignment:
        params.append("sync_alignment=true")
    return f"{base_url}/text-to-dialogue/stream-input?" + "&".join(params)


def _build_dialogue_init_packet(opts: _TTSOptions) -> dict[str, Any]:
    # `voices` must be a list of plain voice-ID strings; voice_settings and
    # pronunciation dictionaries are only accepted in this first message
    init_pkt: dict[str, Any] = {"voices": [opts.voice_id]}
    if is_given(opts.voice_settings):
        init_pkt["voice_settings"] = _strip_nones(dataclasses.asdict(opts.voice_settings))
    if is_given(opts.pronunciation_dictionary_locators):
        init_pkt["pronunciation_dictionary_locators"] = [
            {
                "pronunciation_dictionary_id": locator.pronunciation_dictionary_id,
                "version_id": locator.version_id,
            }
            for locator in opts.pronunciation_dictionary_locators
        ]
    return init_pkt


def _resolve_preferred_alignment(opts: _TTSOptions) -> Literal["normalized", "original"]:
    if is_given(opts.preferred_alignment):
        return opts.preferred_alignment
    if is_given(opts.language) and opts.language.language in {"ja", "ko", "zh"}:
        return "original"
    return "normalized"


def _push_dialogue_alignment(
    stream: SynthesizeStream, emitter: tts.AudioEmitter, alignment: dict[str, Any]
) -> None:
    """Feed one alignment payload (snake_case or camelCase keys) into the stream's timed words"""
    chars = alignment.get("chars")
    starts = (
        alignment.get("char_start_times_ms")
        or alignment.get("charStartTimesMs")
        or alignment.get("charsStartTimesMs")
    )
    durs = (
        alignment.get("char_durations_ms")
        or alignment.get("charDurationsMs")
        or alignment.get("charsDurationsMs")
    )
    if not (chars and starts and durs and len(chars) == len(durs) and len(starts) == len(durs)):
        return

    stream._text_buffer += "".join(chars)
    # in case item in chars has multiple characters
    for char, start, dur in zip(chars, starts, durs, strict=False):
        if len(char) > 1:
            stream._start_times_ms += [start] * (len(char) - 1)
            stream._durations_ms += [0] * (len(char) - 1)
        stream._start_times_ms.append(start)
        stream._durations_ms.append(dur)

    timed_words, stream._text_buffer = _to_timed_words(
        stream._text_buffer, stream._start_times_ms, stream._durations_ms
    )
    emitter.push_timed_transcript(timed_words)
    stream._start_times_ms = stream._start_times_ms[-len(stream._text_buffer) :]
    stream._durations_ms = stream._durations_ms[-len(stream._text_buffer) :]


def _to_timed_words(
    text: str, start_times_ms: list[int], durations_ms: list[int], flush: bool = False
) -> tuple[list[TimedString], str]:
    """Return timed words and the remaining text"""
    if not text:
        return [], ""

    timestamps = start_times_ms + [start_times_ms[-1] + durations_ms[-1]]  # N+1

    words = split_words(text, ignore_punctuation=False, split_character=True)
    if not words:
        return [], text

    timed_words = []
    _, start_indices, _ = zip(*words, strict=False)
    end = 0
    # we don't know if the last word is complete, always leave it as remaining
    for start, end in zip(start_indices[:-1], start_indices[1:], strict=False):
        start_t = timestamps[start] / 1000
        end_t = timestamps[end] / 1000
        timed_words.append(
            TimedString(text=text[start:end], start_time=start_t, end_time=end_t),
        )

    if flush:
        start_t = timestamps[end] / 1000
        end_t = timestamps[-1] / 1000
        timed_words.append(TimedString(text=text[end:], start_time=start_t, end_time=end_t))
        end = len(text)

    return timed_words, text[end:]
