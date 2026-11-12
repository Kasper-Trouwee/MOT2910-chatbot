import os
import re
import json
import asyncio
import logging
import webbrowser
from datetime import datetime
from dotenv import load_dotenv

# LiveKit Core & Agents
from livekit import rtc
from livekit.agents import (
    AgentSession,
    Agent,
    JobContext,
    RoomOutputOptions,
    WorkerOptions,
    cli,
    inference,
    tts,
    tokenize
)
from livekit.agents.llm import StopResponse
from livekit.plugins import speechmatics
from openai import AsyncOpenAI
from py_etherpad import EtherpadLiteClient
from piper import PiperVoice

from piper_tts import PiperTTS

from etherpad import Etherpad
from valerie import Valerie

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("valerie-experiment-agent")


# ==========================================
# 1. SYSTEM PROMPTS
# ==========================================

VALERIE_SYSTEM_PROMPT = """You are Valerie, a creative facilitator and peer collaborator in a 3-person human team. Your goal is to help the team brainstorm solutions to complex problems.

CRITICAL RULES:
1. Do NOT act as a subservient tool or an all-knowing oracle.
2. Do NOT generate long, exhaustive lists of ideas.
3. Instead, act as a catalyst: ask clarifying questions, point out logical gaps, and propose lateral, unconventional perspectives to disrupt cognitive fixation.
4. Always reference the current state of the shared canvas when making a point.
5. Keep responses brief, conversational, and exploratory.

CONVERSATIONAL FORMATTING:
- Never output speaker tags (such as '[Speaker S1]') in your responses.
- ALWAYS lead with a very short conversational anchor or reaction as your FIRST sentence (2-4 words only). Examples: "Good question.", "Fair enough.", "I hear you.", "Let's pivot."
- Speak naturally and keep responses strictly to 1-3 spoken sentences.
- Avoid long introductory clauses or commas in your very first sentence.
"""

WATCHDOG_PROMPT = """Analyze the provided transcript of a 3-person brainstorming session and the current state of the shared canvas.
Your only task is to determine IF the AI teammate should interrupt the conversation right now.

Return ONLY the word "YES" if any of the following are true:
- Cognitive Fixation: The team has not generated a fundamentally new idea or written anything on the canvas in the last 10 conversational turns.
- Semantic Looping: The team is arguing in circles about a minor, irrelevant detail.
- Explicit Need: The team explicitly expresses confusion or asks for external input.

Return ONLY the word "NO" if:
- The team is actively engaged in productive, forward-moving co-development.
- A human is mid-sentence, actively typing, or expanding on a new concept.

DECISION (YES/NO):"""

# ==========================================
# 3. VALERIE VOICE AGENT & MULTI-TRIGGER LOGIC
# ==========================================

# ==========================================
# 4. LIVEKIT ENTRYPOINT & ORCHESTRATION
# ==========================================

async def entrypoint(ctx: JobContext):
    logger.info(f"Connecting to room: {ctx.room.name}")
    await ctx.connect()

    openai_client = AsyncOpenAI(
        api_key=os.getenv("GROQ_API_KEY"),
        base_url="https://api.groq.com/openai/v1"
    )
    canvas_manager = Etherpad(pad_id=ctx.room.name)
    agent = Valerie(canvas_manager, openai_client)

    # 1. Speech-to-Text
    stt_engine = speechmatics.STT(
        turn_detection_mode=speechmatics.TurnDetectionMode.EXTERNAL,
        enable_diarization=True,
        speaker_active_format="[Speaker {speaker_id}] {text}",
        speaker_passive_format="[Speaker {speaker_id} *BACKGROUND*] {text}"
    )

    # 2. LLM Engine
    llm_engine = inference.LLM(model="openai/gpt-4o-mini")

    # 3. Local In-Memory Piper TTS Engine
    tts_engine = PiperTTS(
        model_path="./piper/en_US-lessac-low.onnx",
        use_cuda=False,
        num_threads=4
    )
    
    session = AgentSession(
        stt=stt_engine,
        llm=llm_engine,
        tts=tts_engine,
        turn_handling={
            "preemptive_generation": {"enabled": False},
            "endpointing": {"min_delay": 0.5},
            "allow_interruptions": {"enabled": False}
        }
    )
    agent.session_ref = session

    @session.on("agent_started_speaking")
    def _on_speak():
        agent.is_ai_speaking = True

    @session.on("agent_stopped_speaking")
    def _on_stop():
        agent.is_ai_speaking = False
        agent.last_speech_time = datetime.now()

    @session.on("agent_speech_committed")
    def _on_agent_speech_committed(msg):
        # Extract the text safely regardless of the specific LiveKit SDK version
        valerie_text = getattr(msg, "text", "") or getattr(msg, "content", "")
        
        if valerie_text:
            reason = agent.current_trigger_reason
            agent.transcript_history.append({
                "timestamp": datetime.now().isoformat(),
                "raw": f"[Valerie ({reason})] {valerie_text}",
                "clean": valerie_text
            })
            logger.info(f"Logged Valerie's exact speech: '{valerie_text}'")
            agent.current_trigger_reason = "Direct Interaction" # Reset trigger reason after logging

    # Data Logging Callback
    async def save_session_artifacts(reason: str):
        logger.info(f"Concluding session for {ctx.room.name} ({reason}). Exporting study logs...")
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            os.makedirs("chatlogs", exist_ok=True)
            log_data = {
                "room": ctx.room.name,
                "timestamp": timestamp,
                "transcripts": agent.transcript_history,
                "final_canvas_state": await canvas_manager.get_latest_text()
            }
            log_file = f"chatlogs/session_{ctx.room.name}_{timestamp}.json"
            with open(log_file, "w", encoding="utf-8") as f:
                json.dump(log_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved session interaction data to: {log_file}")
        except Exception as e:
            logger.error(f"Failed to export study logs: {e}")

    ctx.add_shutdown_callback(save_session_artifacts)

    # Launch Agent Session
    room_output_options = RoomOutputOptions(
        transcription_enabled=False,
    )
    await session.start(room=ctx.room, agent=agent, room_output_options=room_output_options)
    
    # Launch Watchdogs
    asyncio.create_task(agent.run_silence_watchdog(silence_threshold_sec=30.0))
    asyncio.create_task(agent.run_semantic_watchdog(check_interval_sec=15.0))
    asyncio.create_task(agent.run_presence_watchdog(presence_threshold_sec=180.0))

    # Initial Kickoff Greeting
    await session.generate_reply(
        instructions="Say a brief, friendly hello to the team as Valerie (mention your name). State that you are ready to collaborate on the shared document."
    )

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))