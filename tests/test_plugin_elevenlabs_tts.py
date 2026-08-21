"""Unit tests for ElevenLabs TTS plugin configuration and websocket behavior."""

import asyncio
import base64
import json
from types import SimpleNamespace

import aiohttp
import pytest

from livekit.plugins.elevenlabs import tts as elevenlabs_tts

pytestmark = pytest.mark.plugin("elevenlabs")


class _FakeWebSocket:
    def __init__(self, messages: list[object]) -> None:
        self._messages = messages
        self.closed = False

    async def receive(self) -> object:
        if self._messages:
            return self._messages.pop(0)
        return SimpleNamespace(type=aiohttp.WSMsgType.CLOSE, data="")

    async def close(self) -> None:
        self.closed = True


class _FakeEmitter:
    def __init__(self) -> None:
        self.audio_chunks: list[bytes] = []
        self.timed_transcript_pushes = 0

    def push(self, audio: bytes) -> None:
        self.audio_chunks.append(audio)

    def push_timed_transcript(self, _timed_words: object) -> None:
        self.timed_transcript_pushes += 1


class _FakeStream:
    def __init__(self) -> None:
        self._text_buffer = ""
        self._start_times_ms: list[int] = []
        self._durations_ms: list[int] = []


class _FakeConnection:
    def __init__(self, context_id: str, messages: list[object]) -> None:
        self._closed = False
        self._ws = _FakeWebSocket(messages)
        self._is_current = True
        self._active_contexts = {context_id}
        self.emitter = _FakeEmitter()
        self.waiter: asyncio.Future[None] = asyncio.get_event_loop().create_future()
        self._context_data = {
            context_id: elevenlabs_tts._StreamData(
                emitter=self.emitter,
                stream=_FakeStream(),
                waiter=self.waiter,
            )
        }
        self.preferred_alignment = "normalized"

    def _cleanup_context(self, context_id: str) -> None:
        ctx = self._context_data.pop(context_id, None)
        if ctx and ctx.timeout_timer:
            ctx.timeout_timer.cancel()
        self._active_contexts.discard(context_id)

    async def aclose(self) -> None:
        self._closed = True
        await self._ws.close()


def _websocket_text_message(payload: dict[str, object]) -> object:
    return SimpleNamespace(type=aiohttp.WSMsgType.TEXT, data=json.dumps(payload))


def test_auto_mode_defaults_to_true_without_chunk_length_schedule() -> None:
    tts = elevenlabs_tts.TTS(api_key="test-key")
    assert tts._opts.auto_mode is True


def test_auto_mode_defaults_to_false_with_chunk_length_schedule() -> None:
    tts = elevenlabs_tts.TTS(api_key="test-key", chunk_length_schedule=[120, 160, 250, 290])
    assert tts._opts.auto_mode is False


def test_auto_mode_respects_explicit_value_with_chunk_length_schedule() -> None:
    tts = elevenlabs_tts.TTS(
        api_key="test-key",
        chunk_length_schedule=[120, 160, 250, 290],
        auto_mode=True,
    )
    assert tts._opts.auto_mode is True


def test_build_context_init_packet_includes_generation_config() -> None:
    tts = elevenlabs_tts.TTS(api_key="test-key", chunk_length_schedule=[80, 120], auto_mode=False)
    packet = elevenlabs_tts._build_context_init_packet(  # pyright: ignore[reportPrivateUsage]
        tts._opts, context_id="ctx-1"
    )

    assert packet["text"] == " "
    assert packet["context_id"] == "ctx-1"
    assert packet["generation_config"] == {"chunk_length_schedule": [80, 120]}


def test_build_context_init_packet_omits_generation_config_when_not_set() -> None:
    tts = elevenlabs_tts.TTS(api_key="test-key")
    packet = elevenlabs_tts._build_context_init_packet(  # pyright: ignore[reportPrivateUsage]
        tts._opts, context_id="ctx-2"
    )

    assert "generation_config" not in packet


def test_build_context_init_packet_includes_pronunciation_dictionaries() -> None:
    tts = elevenlabs_tts.TTS(
        api_key="test-key",
        pronunciation_dictionary_locators=[
            elevenlabs_tts.PronunciationDictionaryLocator(
                pronunciation_dictionary_id="dict-1",
                version_id="v1",
            )
        ],
    )
    packet = elevenlabs_tts._build_context_init_packet(  # pyright: ignore[reportPrivateUsage]
        tts._opts, context_id="ctx-3"
    )

    assert packet["pronunciation_dictionary_locators"] == [
        {
            "pronunciation_dictionary_id": "dict-1",
            "version_id": "v1",
        }
    ]


@pytest.mark.asyncio
async def test_recv_loop_accepts_snake_case_context_id() -> None:
    context_id = "ctx_123"
    audio_chunk = b"hello-audio"
    connection = _FakeConnection(
        context_id,
        [
            _websocket_text_message(
                {
                    "context_id": context_id,
                    "audio": base64.b64encode(audio_chunk).decode("ascii"),
                    "isFinal": True,
                }
            ),
        ],
    )

    await elevenlabs_tts._Connection._recv_loop(connection)

    assert connection.emitter.audio_chunks == [audio_chunk]
    assert connection.waiter.done()
    assert connection.waiter.result() is None
    assert connection._context_data == {}


@pytest.mark.asyncio
async def test_recv_loop_still_accepts_camel_case_context_id() -> None:
    context_id = "ctx_123"
    audio_chunk = b"hello-audio"
    connection = _FakeConnection(
        context_id,
        [
            _websocket_text_message(
                {
                    "contextId": context_id,
                    "audio": base64.b64encode(audio_chunk).decode("ascii"),
                    "isFinal": True,
                }
            ),
        ],
    )

    await elevenlabs_tts._Connection._recv_loop(connection)

    assert connection.emitter.audio_chunks == [audio_chunk]
    assert connection.waiter.done()
    assert connection.waiter.result() is None
    assert connection._context_data == {}


@pytest.mark.asyncio
async def test_recv_loop_ignores_flush_done_for_active_context() -> None:
    context_id = "ctx_123"
    audio_chunk = b"hello-audio"
    connection = _FakeConnection(
        context_id,
        [
            _websocket_text_message(
                {
                    "type": "flush_done",
                    "context_id": context_id,
                    "status_code": 206,
                    "done": False,
                    "data": "",
                    "flush_done": True,
                }
            ),
            _websocket_text_message(
                {
                    "context_id": context_id,
                    "audio": base64.b64encode(audio_chunk).decode("ascii"),
                    "isFinal": True,
                }
            ),
        ],
    )

    await elevenlabs_tts._Connection._recv_loop(connection)

    assert connection.emitter.audio_chunks == [audio_chunk]
    assert connection.waiter.done()
    assert connection.waiter.result() is None


@pytest.mark.asyncio
async def test_recv_loop_ignores_flush_done_for_inactive_context() -> None:
    context_id = "ctx_123"
    audio_chunk = b"hello-audio"
    connection = _FakeConnection(
        context_id,
        [
            _websocket_text_message(
                {
                    "type": "flush_done",
                    "context_id": "already_closed_context",
                    "status_code": 206,
                    "done": False,
                    "data": "",
                    "flush_done": True,
                }
            ),
            _websocket_text_message(
                {
                    "context_id": context_id,
                    "audio": base64.b64encode(audio_chunk).decode("ascii"),
                    "isFinal": True,
                }
            ),
        ],
    )

    await elevenlabs_tts._Connection._recv_loop(connection)

    assert connection.emitter.audio_chunks == [audio_chunk]
    assert connection.waiter.done()
    assert connection.waiter.result() is None


class _FakeDialogueWebSocket:
    def __init__(self, messages: list[object] | None = None) -> None:
        self._messages = list(messages or [])
        self.closed = False
        self.close_code: int | None = None
        self.sent: list[dict[str, object]] = []

    async def receive(self) -> object:
        if self._messages:
            return self._messages.pop(0)
        return SimpleNamespace(type=aiohttp.WSMsgType.CLOSE, data="")

    async def send_json(self, data: dict[str, object]) -> None:
        self.sent.append(data)

    async def close(self) -> None:
        self.closed = True


def _make_dialogue_connection(
    messages: list[object] | None = None,
    **tts_kwargs: object,
) -> tuple["elevenlabs_tts._DialogueConnection", _FakeDialogueWebSocket]:
    tts = elevenlabs_tts.TTS(api_key="test-key", model="eleven_v3_conversational", **tts_kwargs)
    connection = elevenlabs_tts._DialogueConnection(  # pyright: ignore[reportPrivateUsage]
        tts._opts, None
    )
    ws = _FakeDialogueWebSocket(messages)
    connection._ws = ws
    return connection, ws


async def _make_dialogue_turn(
    connection: "elevenlabs_tts._DialogueConnection",
    *,
    stream: object | None = None,
) -> "elevenlabs_tts._DialogueTurn":
    emitter = _FakeEmitter()
    waiter: asyncio.Future[None] = asyncio.get_event_loop().create_future()
    return await connection.start_turn(emitter=emitter, stream=stream, waiter=waiter, timeout=5.0)


def test_v3_models_route_to_dialogue() -> None:
    assert elevenlabs_tts._is_dialogue_model("eleven_v3")
    assert elevenlabs_tts._is_dialogue_model("eleven_v3_conversational")
    assert not elevenlabs_tts._is_dialogue_model("eleven_turbo_v2_5")
    assert not elevenlabs_tts._is_dialogue_model("eleven_flash_v2_5")


def test_dialogue_stream_url_includes_expected_params() -> None:
    tts = elevenlabs_tts.TTS(api_key="test-key", model="eleven_v3_conversational", language="de")
    url = elevenlabs_tts._dialogue_stream_url(tts._opts)  # pyright: ignore[reportPrivateUsage]

    assert url.startswith("wss://api.elevenlabs.io/v1/text-to-dialogue/stream-input?")
    assert "model_id=eleven_v3_conversational" in url
    assert "output_format=mp3_22050_32" in url
    assert "language_code=de" in url
    assert "sync_alignment=true" in url


def test_dialogue_stream_url_omits_language_when_not_set() -> None:
    tts = elevenlabs_tts.TTS(api_key="test-key", model="eleven_v3", sync_alignment=False)
    url = elevenlabs_tts._dialogue_stream_url(tts._opts)  # pyright: ignore[reportPrivateUsage]

    assert "language_code" not in url
    assert "sync_alignment" not in url


def test_build_dialogue_init_packet_voices_are_plain_strings() -> None:
    tts = elevenlabs_tts.TTS(api_key="test-key", model="eleven_v3", voice_id="voice-1")
    packet = elevenlabs_tts._build_dialogue_init_packet(  # pyright: ignore[reportPrivateUsage]
        tts._opts
    )

    assert packet["voices"] == ["voice-1"]
    assert "voice_settings" not in packet


def test_build_dialogue_init_packet_includes_voice_settings_when_given() -> None:
    tts = elevenlabs_tts.TTS(
        api_key="test-key",
        model="eleven_v3",
        voice_settings=elevenlabs_tts.VoiceSettings(stability=0.5, similarity_boost=0.8),
    )
    packet = elevenlabs_tts._build_dialogue_init_packet(  # pyright: ignore[reportPrivateUsage]
        tts._opts
    )

    assert packet["voice_settings"] == {"stability": 0.5, "similarity_boost": 0.8}


def test_build_dialogue_init_packet_includes_pronunciation_dictionaries() -> None:
    tts = elevenlabs_tts.TTS(
        api_key="test-key",
        model="eleven_v3",
        pronunciation_dictionary_locators=[
            elevenlabs_tts.PronunciationDictionaryLocator(
                pronunciation_dictionary_id="dict-1",
                version_id="v1",
            )
        ],
    )
    packet = elevenlabs_tts._build_dialogue_init_packet(  # pyright: ignore[reportPrivateUsage]
        tts._opts
    )

    assert packet["pronunciation_dictionary_locators"] == [
        {
            "pronunciation_dictionary_id": "dict-1",
            "version_id": "v1",
        }
    ]


@pytest.mark.asyncio
async def test_dialogue_send_text_flags_new_turn_only_on_first_input() -> None:
    connection, ws = _make_dialogue_connection()
    turn = await _make_dialogue_turn(connection)

    await connection.send_text(turn, "Hello there. ")
    await connection.send_text(turn, "Second sentence. ")

    first, second = ws.sent
    assert first["inputs"][0]["new_turn"] is True
    assert second["inputs"][0]["new_turn"] is False


@pytest.mark.asyncio
async def test_dialogue_empty_flush_is_not_sent() -> None:
    connection, ws = _make_dialogue_connection()
    turn = await _make_dialogue_turn(connection)

    await connection.flush_turn(turn)
    assert ws.sent == []
    assert turn.flushes_sent == 0

    await connection.send_text(turn, "Hello. ")
    await connection.flush_turn(turn)
    assert ws.sent[-1] == {"flush": True}
    assert turn.flushes_sent == 1

    await connection.flush_turn(turn)
    assert turn.flushes_sent == 1


@pytest.mark.asyncio
async def test_dialogue_recv_loop_routes_audio_and_completes_on_marker() -> None:
    audio_chunk = b"dialogue-audio"
    connection, ws = _make_dialogue_connection(
        [
            _websocket_text_message({"audio": base64.b64encode(audio_chunk).decode("ascii")}),
            _websocket_text_message({"is_final_audio_for_turn": True}),
        ]
    )
    turn = await _make_dialogue_turn(connection)
    turn.flushes_sent = 1
    turn.input_done = True

    await connection._recv_loop()

    assert turn.emitter.audio_chunks == [audio_chunk]  # type: ignore[attr-defined]
    assert turn.waiter.done()
    assert turn.waiter.result() is None


@pytest.mark.asyncio
async def test_dialogue_recv_loop_waits_for_all_flush_markers() -> None:
    connection, ws = _make_dialogue_connection(
        [
            _websocket_text_message({"is_final_audio_for_turn": True}),
            _websocket_text_message({"is_final_audio_for_turn": True}),
        ]
    )
    turn = await _make_dialogue_turn(connection)
    turn.flushes_sent = 2
    turn.input_done = True

    async def _recv_one() -> None:
        msg = await ws.receive()
        data = json.loads(msg.data)  # type: ignore[attr-defined]
        if data.get("is_final_audio_for_turn"):
            turn.markers_received += 1
            connection._maybe_complete_turn(turn)

    await _recv_one()
    assert not turn.waiter.done()

    await _recv_one()
    assert turn.waiter.done()
    assert turn.waiter.result() is None


@pytest.mark.asyncio
async def test_dialogue_recv_loop_parses_snake_case_alignment() -> None:
    audio_chunk = b"aligned-audio"
    connection, ws = _make_dialogue_connection(
        [
            _websocket_text_message(
                {
                    "audio": base64.b64encode(audio_chunk).decode("ascii"),
                    "normalized_alignment": {
                        "chars": ["h", "e", "y", " ", "n", "o", "w"],
                        "char_start_times_ms": [0, 10, 20, 30, 40, 50, 60],
                        "char_durations_ms": [10, 10, 10, 10, 10, 10, 10],
                    },
                }
            ),
            _websocket_text_message({"is_final_audio_for_turn": True}),
        ]
    )
    stream = _FakeStream()
    turn = await _make_dialogue_turn(connection, stream=stream)
    turn.flushes_sent = 1
    turn.input_done = True

    await connection._recv_loop()

    assert turn.emitter.audio_chunks == [audio_chunk]  # type: ignore[attr-defined]
    assert turn.emitter.timed_transcript_pushes > 0  # type: ignore[attr-defined]
    assert stream._text_buffer != ""
    assert turn.waiter.done()


@pytest.mark.asyncio
async def test_dialogue_recv_loop_idle_timeout_without_turn_is_benign() -> None:
    connection, ws = _make_dialogue_connection(
        [
            _websocket_text_message(
                {
                    "message": "No message received within 20s.",
                    "error": "input_timeout_exceeded",
                    "code": 1008,
                }
            ),
        ]
    )

    await connection._recv_loop()

    assert connection._closed
    assert ws.closed


@pytest.mark.asyncio
async def test_dialogue_recv_loop_audio_without_turn_closes_connection() -> None:
    connection, ws = _make_dialogue_connection(
        [
            _websocket_text_message({"audio": base64.b64encode(b"stale-audio").decode("ascii")}),
        ]
    )

    await connection._recv_loop()

    assert connection._closed
    assert ws.closed


@pytest.mark.asyncio
async def test_dialogue_recv_loop_close_with_active_turn_sets_error() -> None:
    connection, ws = _make_dialogue_connection([])
    ws.close_code = 1008
    turn = await _make_dialogue_turn(connection)

    await connection._recv_loop()

    assert turn.waiter.done()
    assert isinstance(turn.waiter.exception(), elevenlabs_tts.APIStatusError)


@pytest.mark.asyncio
async def test_dialogue_mark_non_current_closes_idle_connection() -> None:
    connection, ws = _make_dialogue_connection()

    connection.mark_non_current()

    assert connection._close_task is not None
    await connection._close_task
    assert connection._closed
    assert ws.closed


@pytest.mark.asyncio
async def test_dialogue_finish_turn_closes_non_current_connection() -> None:
    connection, ws = _make_dialogue_connection()
    turn = await _make_dialogue_turn(connection)

    connection.mark_non_current()
    assert connection._close_task is None  # active turn keeps the socket open until drained

    connection.finish_turn(turn)
    assert connection._close_task is not None
    await connection._close_task
    assert connection._closed
    assert ws.closed


@pytest.mark.asyncio
async def test_dialogue_stall_timer_armed_until_final_marker() -> None:
    connection, ws = _make_dialogue_connection()
    turn = await _make_dialogue_turn(connection)

    await connection.send_text(turn, "Hello. ")
    await connection.end_turn_input(turn)

    assert not turn.waiter.done()
    assert turn.timeout_timer is not None  # stall timer stays armed until the final marker

    turn.markers_received += 1
    connection._maybe_complete_turn(turn)
    assert turn.waiter.done()
    assert turn.waiter.result() is None
    connection.finish_turn(turn)


@pytest.mark.asyncio
async def test_dialogue_discard_during_shutdown_skips_prewarm() -> None:
    tts = elevenlabs_tts.TTS(api_key="test-key", model="eleven_v3_conversational")
    connection = elevenlabs_tts._DialogueConnection(  # pyright: ignore[reportPrivateUsage]
        tts._opts, None
    )
    ws = _FakeDialogueWebSocket()
    connection._ws = ws

    tts._closing = True
    tts._discard_dialogue_connection(connection)

    assert tts._prewarm_task is None  # shutdown must not schedule a replacement
    assert connection._close_task is not None
    await connection._close_task
    assert connection._closed
    assert ws.closed


@pytest.mark.asyncio
async def test_dialogue_connection_not_published_when_closed_mid_handshake(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tts = elevenlabs_tts.TTS(
        api_key="test-key",
        model="eleven_v3_conversational",
        http_session=SimpleNamespace(),  # type: ignore[arg-type]
    )
    closed: list[bool] = []

    async def _connect(self: object) -> None:
        tts._closing = True  # aclose() begins while the handshake is in flight

    async def _aclose(self: object) -> None:
        closed.append(True)

    monkeypatch.setattr(elevenlabs_tts._DialogueConnection, "connect", _connect)
    monkeypatch.setattr(elevenlabs_tts._DialogueConnection, "aclose", _aclose)

    with pytest.raises(elevenlabs_tts.APIConnectionError):
        await tts._current_connection()

    assert closed == [True]
    assert tts._TTS__current_connection is None  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_dialogue_connection_not_published_when_family_switched_mid_handshake(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tts = elevenlabs_tts.TTS(
        api_key="test-key",
        model="eleven_v3_conversational",
        http_session=SimpleNamespace(),  # type: ignore[arg-type]
    )
    closed: list[bool] = []

    async def _connect(self: object) -> None:
        # update_options() cannot mark the unpublished connection non-current
        tts.update_options(model="eleven_turbo_v2_5")

    async def _aclose(self: object) -> None:
        closed.append(True)

    monkeypatch.setattr(elevenlabs_tts._DialogueConnection, "connect", _connect)
    monkeypatch.setattr(elevenlabs_tts._DialogueConnection, "aclose", _aclose)

    with pytest.raises(elevenlabs_tts.APIConnectionError):
        await tts._current_connection()

    assert closed == [True]
    assert tts._TTS__current_connection is None  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_dialogue_start_turn_times_out_when_turn_is_held() -> None:
    connection, ws = _make_dialogue_connection()
    turn = await _make_dialogue_turn(connection)

    waiter: asyncio.Future[None] = asyncio.get_event_loop().create_future()
    with pytest.raises(elevenlabs_tts.APITimeoutError):
        await connection.start_turn(
            emitter=_FakeEmitter(), stream=None, waiter=waiter, timeout=0.05
        )

    connection.finish_turn(turn)


@pytest.mark.asyncio
async def test_dialogue_connect_closes_socket_when_init_send_fails() -> None:
    tts = elevenlabs_tts.TTS(api_key="test-key", model="eleven_v3_conversational")
    ws = _FakeDialogueWebSocket()

    async def _failing_send_json(data: dict[str, object]) -> None:
        raise RuntimeError("init rejected")

    ws.send_json = _failing_send_json  # type: ignore[method-assign]

    class _FakeSession:
        async def ws_connect(self, url: str, headers: dict[str, str]) -> _FakeDialogueWebSocket:
            return ws

    connection = elevenlabs_tts._DialogueConnection(  # pyright: ignore[reportPrivateUsage]
        tts._opts, _FakeSession()
    )

    with pytest.raises(RuntimeError):
        await connection.connect()

    assert ws.closed
    assert connection._closed
    assert connection._ws is None


@pytest.mark.asyncio
async def test_dialogue_acquire_preserves_non_retryable_closed_error() -> None:
    tts = elevenlabs_tts.TTS(
        api_key="test-key",
        model="eleven_v3_conversational",
        http_session=SimpleNamespace(),  # type: ignore[arg-type]
    )
    tts._closing = True

    from livekit.agents import DEFAULT_API_CONNECT_OPTIONS

    with pytest.raises(elevenlabs_tts.APIConnectionError) as exc_info:
        await elevenlabs_tts._acquire_dialogue_connection(  # pyright: ignore[reportPrivateUsage]
            tts, DEFAULT_API_CONNECT_OPTIONS
        )

    assert exc_info.value.retryable is False


@pytest.mark.asyncio
async def test_dialogue_connection_not_published_when_options_change_mid_handshake(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tts = elevenlabs_tts.TTS(
        api_key="test-key",
        model="eleven_v3",
        http_session=SimpleNamespace(),  # type: ignore[arg-type]
    )
    closed: list[bool] = []

    async def _connect(self: object) -> None:
        # a same-family switch; the URL was already built with the old model
        tts.update_options(model="eleven_v3_conversational")

    async def _aclose(self: object) -> None:
        closed.append(True)

    monkeypatch.setattr(elevenlabs_tts._DialogueConnection, "connect", _connect)
    monkeypatch.setattr(elevenlabs_tts._DialogueConnection, "aclose", _aclose)

    with pytest.raises(elevenlabs_tts.APIConnectionError):
        await tts._current_connection()

    assert closed == [True]
    assert tts._TTS__current_connection is None  # type: ignore[attr-defined]
