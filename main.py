import os
import json
import asyncio
import logging
import atexit
from datetime import datetime
from dotenv import load_dotenv

# LiveKit Core & Agents
from livekit.agents import (
    AgentSession,
    JobContext,
    JobProcess,
    RoomOutputOptions,
    WorkerOptions,
    cli,
    inference
)
from livekit.plugins import speechmatics
from openai import AsyncOpenAI

from piper_tts import PiperTTS

from etherpad import Etherpad
from valerie import Valerie
from prompts import CLIMATE_CHANGE_MITIGATION_TASK, CHEMICAL_WASTE_MANAGEMENT_TASK, CLOTHING_RECYCLING_POLICY_TASK
from pathlib import Path

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("valerie-experiment-agent")

logging.getLogger("speechmatics.rt.transport").setLevel(logging.WARNING)
logging.getLogger("speechmatics").setLevel(logging.WARNING)

TASK_SEQUENCE = [
    ("climate", CLIMATE_CHANGE_MITIGATION_TASK),
    ("chemical", CHEMICAL_WASTE_MANAGEMENT_TASK),
    ("clothing", CLOTHING_RECYCLING_POLICY_TASK),
]

def get_next_task(group_id: str) -> tuple[str, str]:
    """Read and increment the task index on disk so sequential LiveKit job processes stay in sync."""
    counter_file = Path(f"/tmp/valerie_task_idx_{group_id}.txt")
    idx = 0
    if counter_file.exists():
        try:
            idx = int(counter_file.read_text().strip())
        except ValueError:
            idx = 0

    task = TASK_SEQUENCE[idx % len(TASK_SEQUENCE)]
    counter_file.write_text(str(idx + 1))
    return task
async def entrypoint(ctx: JobContext):
    logger.info(f"Connecting to room: {ctx.room.name}")
    await ctx.connect()

    # Get group_id and next prompt
    group_id = os.getenv("STUDY_GROUP_ID", "default-group")
    prompt_slug, task_prompt = get_next_task(group_id)

    # Create a completely unique, URL-safe pad ID (e.g., "group4-climate")
    pad_id = f"{group_id}-{prompt_slug}"

    logger.info(f"Starting session for pad_id: '{pad_id}'")

    openai_client = AsyncOpenAI(
        api_key=os.getenv("GROQ_API_KEY"),
        base_url="https://api.groq.com/openai/v1"
    )

    canvas_manager = Etherpad(pad_id=pad_id, prompt=task_prompt)
    await canvas_manager.initialize()
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
    tts_engine = ctx.proc.userdata["tts"]
    
    session = AgentSession(
        stt=stt_engine,
        llm=llm_engine,
        tts=tts_engine,
        allow_interruptions=False,
        min_endpointing_delay=0.5,
        turn_handling={
            "preemptive_generation": {"enabled": False},
            "endpointing": {"min_delay": 0.5},
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

    @session.on("conversation_item_added")
    def _on_conversation_item_added(event):
        item = event.item
        if getattr(item, "role", None) != "assistant":
            return

        # In livekit-agents 1.x, ChatMessage exposes .text_content (and .content as a list)
        valerie_text = getattr(item, "text_content", "") or ""
        if not valerie_text and isinstance(getattr(item, "content", None), list):
            valerie_text = " ".join(c for c in item.content if isinstance(c, str))

        valerie_text = valerie_text.strip()
        if valerie_text:
            reason = agent.current_trigger_reason
            agent.transcript_history.append({
                "timestamp": datetime.now().isoformat(),
                "raw": f"[Valerie ({reason})] {valerie_text}",
                "clean": valerie_text,
            })
            logger.info(f"Logged Valerie's exact speech: '{valerie_text}'")
            agent.current_trigger_reason = "Direct Interaction"  # Reset trigger reason after logging
    
    # Data Logging Callback
    async def save_session_artifacts(reason: str):
        logger.info(f"Concluding session for {pad_id} ({reason}). Exporting study logs...")
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            os.makedirs("chatlogs", exist_ok=True)
            log_data = {
                "group_id": group_id,
                "task": prompt_slug,
                "pad_id": pad_id,
                "room": ctx.room.name,
                "timestamp": timestamp,
                "transcripts": agent.transcript_history,
                "final_canvas_state": await canvas_manager.get_latest_text()
            }
            log_file = f"chatlogs/session_{pad_id}_{timestamp}.json"
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
    watchdog_tasks = [
        asyncio.create_task(agent.run_silence_watchdog(silence_threshold_sec=30.0)),
        asyncio.create_task(agent.run_semantic_watchdog(check_interval_sec=15.0)),
        asyncio.create_task(agent.run_presence_watchdog(presence_threshold_sec=180.0)),
    ]
    @session.on("close")
    def _on_session_close(_event):
        for task in watchdog_tasks:
            task.cancel()

    # Initial Kickoff Greeting
    await session.generate_reply(
        instructions="Say a brief, friendly hello to the team as Valerie (mention your name). State that you are ready to collaborate on the shared document."
    )


def cleanup_etherpad():
    # Connects to the port check without needing a specific pad
    manager = Etherpad.__new__(Etherpad)
    manager.url = "http://localhost:9001"
    manager.process = None
    manager.stop_etherpad()

def prewarm(proc: JobProcess):
    proc.userdata["tts"] = PiperTTS(
        model_path="./piper/en_US-lessac-low.onnx",
        use_cuda=False,
        num_threads=4
    )

if __name__ == "__main__":
    group_id = input("Enter Group ID for this session: ").strip()
    if not group_id:
        group_id = f"group_{datetime.now().strftime('%H%M%S')}"
        print(f"No Group ID entered, defaulting to: {group_id}")
    safe_group = "".join(c if c.isalnum() or c in "-_" else "_" for c in group_id)
    # Store in environment variable so worker subprocesses inherit it
    os.environ["STUDY_GROUP_ID"] = group_id

    # Ensure a fresh start at task 0 for this group_id
    counter_file = Path(f"/tmp/valerie_task_idx_{safe_group}.txt")
    if counter_file.exists():
        counter_file.unlink()

    boot_manager = Etherpad(pad_id="boot-check") # start etherpad
    boot_manager.start_etherpad(timeout=45)
    atexit.register(boot_manager.stop_etherpad) # register closing for etherpad

    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))