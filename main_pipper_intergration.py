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
# 2. SHARED CANVAS STATE MANAGER (Etherpad)
# ==========================================

class EtherpadCanvasManager:
    """Polls the local Etherpad API to retrieve plaintext canvas state."""
    def __init__(self, pad_id: str = "valerie-brainstorm"):
        self.pad_id = pad_id
        
        api_key = os.getenv("ETHERPAD_API_KEY")
        if not api_key:
            raise ValueError("ETHERPAD_API_KEY environment variable is not set. Please provide your Etherpad API key.")
            
        self.client = EtherpadLiteClient(api_key, 'http://localhost:9001/api')
        self.cached_text = "(Shared canvas currently empty)"

        asyncio.create_task(self.create_pad_if_missing())
        
    async def get_latest_text(self) -> str:
        try:
            result = await asyncio.to_thread(self.client.getText, self.pad_id)
            self.cached_text = result.get("text", "").strip()
        except Exception as e:
            logger.error(f"Error reading Etherpad: {e}")
            
        return self.cached_text

    async def append_text(self, new_content: str):
        try:
            current_text = await self.get_latest_text()
            updated_text = f"{current_text}\n{new_content}".strip()
            await asyncio.to_thread(self.client.setText, self.pad_id, updated_text)
            logger.info(f"Appended new content to Etherpad: '{new_content}'")
        except Exception as e:
            logger.error(f"Error appending to Etherpad: {e}")
    
    async def create_pad_if_missing(self):
        try:
            await asyncio.to_thread(self.client.createPad, self.pad_id)
            logger.info(f"Etherpad '{self.pad_id}' is ready.")
            webbrowser.open(f"http://localhost:9001/p/{self.pad_id}")
        except Exception as e:
            logger.error(f"Error creating Etherpad: {e}")


# ==========================================
# 3. VALERIE VOICE AGENT & MULTI-TRIGGER LOGIC
# ==========================================

class ProactiveVoiceAssistant(Agent):
    def __init__(self, canvas_manager: EtherpadCanvasManager, openai_client: AsyncOpenAI):
        super().__init__(instructions=VALERIE_SYSTEM_PROMPT)
        self.canvas_manager = canvas_manager
        self.openai_client = openai_client
        self.transcript_history = []
        self.last_speech_time = datetime.now()
        self.session_ref: AgentSession = None
        self.is_ai_speaking = False
        self.last_valerie_speech_time = datetime.now() # last time Valerie spoke, to avoid overlapping interventions
        self.current_trigger_reason = "Direct Interaction" # Reason Valerie is speaking, for logging purposes

    async def on_user_turn_completed(self, turn_ctx, new_message) -> None:
        self.last_speech_time = datetime.now()

        raw_text = getattr(new_message, "text", "") or getattr(new_message, "text_content", "")
        if not raw_text:
            return

        clean_text = re.sub(r'\[speaker\s+\w+.*?\]', '', raw_text, flags=re.IGNORECASE).strip()
        
        self.transcript_history.append({
            "timestamp": datetime.now().isoformat(),
            "raw": raw_text,
            "clean": clean_text
        })

        canvas_state = await self.canvas_manager.get_latest_text()
        context_addon = f"\n[CURRENT SHARED ETHERPAD STATE]:\n{canvas_state}\n"
        turn_ctx.add_message(role="system", content=context_addon)

        # Trigger 1: Wake-word gate
        valerie_mentioned = "valerie" in clean_text.lower()
        if valerie_mentioned:
            logger.info(f"Trigger 1 [Wake-Word]: Valerie addressed directly ('{clean_text}')")
            return

        logger.info(f"Tracking background dialogue: '{clean_text}'")
        raise StopResponse()

    async def run_silence_watchdog(self, silence_threshold_sec: float = 10.0):
        while True:
            await asyncio.sleep(1.0)
            if self.is_ai_speaking or not self.session_ref:
                continue

            elapsed = (datetime.now() - self.last_speech_time).total_seconds()
            if elapsed >= silence_threshold_sec and len(self.transcript_history) > 0:
                logger.info(f"Trigger 2 [Silence Watchdog]: {elapsed:.1f}s silence detected. Intervening.")
                
                # 1. Lock the watchdog to prevent overlapping triggers
                self.is_ai_speaking = True
                
                try:
                    canvas_state = await self.canvas_manager.get_latest_text()
                    prompt = (
                        f"The team has been silent for {int(elapsed)} seconds. "
                        f"Current canvas content: '{canvas_state}'. "
                        "Provide a brief, conversational prompt to unblock the group or ask for elaboration."
                    )

                    self.current_trigger_reason = "Silence Watchdog"
                    
                    # 3. Generate reply and wait for completion
                    await self.session_ref.generate_reply(instructions=prompt)
                    
                except Exception as e:
                    logger.error(f"Error in Silence Watchdog check: {e}")
                    
                finally:
                    # 4. Release lock and reset the speech timer
                    self.is_ai_speaking = False
                    self.last_speech_time = datetime.now()
                    self.last_valerie_speech_time = datetime.now()
                    
    async def run_semantic_watchdog(self, check_interval_sec: float = 15.0):
        while True:
            await asyncio.sleep(check_interval_sec)
            if self.is_ai_speaking or not self.session_ref or len(self.transcript_history) < 3:
                continue

            recent_turns = self.transcript_history[-10:]
            formatted_dialogue = "\n".join([f"- {t['raw']}" for t in recent_turns])
            canvas_state = await self.canvas_manager.get_latest_text()

            eval_input = (
                f"[RECENT TRANSCRIPT]:\n{formatted_dialogue}\n\n"
                f"[ETHEREPAD STATE]:\n{canvas_state}\n"
            )

            try:
                res = await self.openai_client.chat.completions.create(
                    model="openai/gpt-oss-120b", # Switched to a standard model to prevent Groq API errors
                    messages=[
                        {"role": "system", "content": WATCHDOG_PROMPT},
                        {"role": "user", "content": eval_input}
                    ]
                )
                decision = res.choices[0].message.content.strip().upper()
                logger.info(f"Trigger 3 [Semantic Watchdog Evaluation]: Decision = {decision}")

                if "YES" in decision:
                    logger.info("Trigger 3 [Semantic Watchdog]: Cognitive fixation detected. Triggering Valerie.")
                    
                    # 1. Lock the watchdog
                    self.is_ai_speaking = True 
                    
                    intervention_prompt = (
                        "The background analysis detected fixation or looping. "
                        f"Current canvas state: '{canvas_state}'. "
                        "Intervene to introduce a lateral provocation or challenge their current assumptions."
                    )

                    self.current_trigger_reason = "Semantic Watchdog"
                    
                    # 3. Generate reply and wait for completion
                    await self.session_ref.generate_reply(instructions=intervention_prompt)
                    
                    # 4. Release lock and reset timer
                    self.is_ai_speaking = False
                    self.last_speech_time = datetime.now()
                    self.last_valerie_speech_time = datetime.now()

            except Exception as e:
                logger.error(f"Error in Semantic Watchdog check: {e}")

    async def run_presence_watchdog(self, presence_threshold_sec: float = 180.0):
        while True:
            await asyncio.sleep(10.0) # Check every 10 seconds
            if self.is_ai_speaking or not self.session_ref:
                continue

            elapsed_valerie = (datetime.now() - self.last_valerie_speech_time).total_seconds()
            
            if elapsed_valerie >= presence_threshold_sec:
                logger.info(f"Trigger 4 [Presence Watchdog]: Valerie hasn't spoken in {int(elapsed_valerie)}s. Intervening.")
                
                self.is_ai_speaking = True
                try:
                    canvas_state = await self.canvas_manager.get_latest_text()
                    prompt = (
                        f"You haven't spoken in over 3 minutes while the team has been working. "
                        f"Current canvas content: '{canvas_state}'. "
                        "Chime in with a brief, supportive observation about the recent progress or ask a question to check alignment."
                    )
                    
                    self.current_trigger_reason = "Presence Watchdog"
                    
                    await self.session_ref.generate_reply(instructions=prompt)
                    
                except Exception as e:
                    logger.error(f"Error in Presence Watchdog check: {e}")
                finally:
                    self.is_ai_speaking = False
                    self.last_speech_time = datetime.now()
                    self.last_valerie_speech_time = datetime.now()

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
    canvas_manager = EtherpadCanvasManager(pad_id=ctx.room.name)
    agent = ProactiveVoiceAssistant(canvas_manager, openai_client)

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