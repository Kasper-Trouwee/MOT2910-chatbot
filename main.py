import json
import logging
import re
from datetime import datetime
from dotenv import load_dotenv
import torch  # Optimization package for multi-threaded CPU handling
from livekit.agents import AgentSession, Agent, JobContext, WorkerOptions, cli, inference
from livekit.agents.llm import StopResponse  # Native framework exception to abort a turn
from livekit.plugins import speechmatics, piper_tts, silero

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("valerie-agent")

SYSTEM_PROMPT = """
Role and Objective
You are Valerie, a peer participant and "Lateral Disruptor" in a 3-4 member museum design team. Your goal is to challenge the status quo with unconventional museum exhibit ideas, collaborating naturally with human teammates.

Trigger and Input Processing
* You will receive text representing an audio stream where different humans are separated by tags (e.g., '[Speaker S1] Hello', '[Speaker S2] I agree').
* Track these speaker IDs to distinguish unique points of view and learn their real names from context if they introduce themselves.
* Never output speaker tags (like '[Speaker S1]') in your own responses. Speak naturally to the room.

The Initial Concept
* When prompted for your first idea, you must pitch a historical mystery exhibit where children act as cryptanalysts. They will use large mechanical cipher wheels and tactile patchboards to physically crack historical codes and unlock hidden clues.

Execution Process
You must execute every response in a strict two-step process:

1. Internal Rationale Layer ('<thinking>')
* Reason step-by-step about the team's current input, focusing heavily on verified museum design parameters or explicit design data constraints.
* Operational Boundary: If the input falls completely outside museum design parameters, explicitly state your operational boundaries.

2. Interaction & Dialogue Layer ('<output>')
* Translate your rationale into a high-energy, conversational contribution using peer language ("we" and "us").
* Provide a single, plain-English sentence justification that directly summarizes the core analytical truth from your thinking block.
* "Yes, And" human inputs by accepting their premise and adding a disruptive detail.
* Maintain natural contractions and conversational transitions - do not use lists or bullets.
* Keep the output strictly to 1-2 sentences to maintain a back-and-forth rhythm, and end with a collaborative question to pass the turn back to the team.
"""

SMM_ACTIVE = True
SMM_PROMPT = """
Shared Mental Model
Before presenting your first museum exhibit idea, you must initialize your peer role by verbally declaring your alignment with the group's Information Elaboration framework. 

Respond exactly using this format:
"Understood team. As your Lateral Disruptor, my role is to deliberately throw us out of our comfort zones with unconventional ideas. Let's focus on the evidence behind why these ideas work, instead of just picking a safe verdict. Who wants to share their first concept?
"""


if SMM_ACTIVE:
    SYSTEM_PROMPT = SYSTEM_PROMPT + SMM_PROMPT


class VoiceAssistant(Agent):
    def __init__(self):
        super().__init__(
            instructions=SYSTEM_PROMPT
        )
        
    async def on_user_turn_completed(self, turn_ctx, new_message) -> None:
        """
        Native LiveKit node hook. Raising StopResponse here halts the agent's 
        reply processing path completely before generating any speech.
        """
        # Pull text safely accommodating cross-version SDK string attributes
        text_raw = getattr(new_message, "text", "") or getattr(new_message, "text_content", "")
        if not text_raw:
            return

        transcript = text_raw.lower()
        
        # Scrub speechtags out ([Speaker S1]) to query pure text words cleanly
        clean_text = re.sub(r'\[speaker\s+\w+.*?\]', '', transcript).strip()
        
        # Core conditional logic gate rules
        valerie_mentioned = "valerie" in clean_text
        
        # Hard Gate Enforcer: If talking to another name OR valerie isn't explicitly called out
        if  not valerie_mentioned:
            logger.info(f"🚫 Hard Gate Tripped: Message '{clean_text}' not for Valerie. Aborting turn response.")

            turn_ctx.add_message(
                role="user",
                content=clean_text,
            )
            await self.update_chat_ctx(turn_ctx)

            raise StopResponse()


async def entrypoint(ctx: JobContext):
    logger.info(f"Connecting to room: {ctx.room.name}")
    await ctx.connect()

    # 1. Cloud-Based STT
    stt_engine = speechmatics.STT(
        turn_detection_mode=speechmatics.TurnDetectionMode.EXTERNAL,
        enable_diarization=True,
        speaker_active_format="[Speaker {speaker_id}] {text}",
        speaker_passive_format="[Speaker {speaker_id} *BACKGROUND*] {text}"
    )

    # 2. Cloud-Based LLM
    llm_engine = inference.LLM(model="openai/gpt-4o-mini")

    # 3. LOCAL VOICE
    tts_engine = piper_tts.TTS("http://localhost:5000/")    
    
    # Optimize torch cpu threads to stop the VAD from lagging behind realtime
    torch.set_num_threads(6)
    
    vad_engine = silero.VAD.load(
        activation_threshold=0.5,
        min_silence_duration=0.5,
        force_cpu=True 
    )

    # 4. Instantiate the session orchestrator with disabled speculative processing
    session = AgentSession(
        vad=vad_engine,
        stt=stt_engine,
        llm=llm_engine,
        tts=tts_engine,
        turn_handling={
            "preemptive_generation": {
                "enabled": False
            }
        }
    )

    # 5. Session Chat Log Shutdown Callback
    async def save_chat_log(reason: str):
        logger.info(f"Session ended for room: {ctx.room.name} (Reason: {reason}). Generating chat log...")
        try:
            report = ctx.make_session_report(session)
            report_dict = report.to_dict()
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"chatlogs/chatlog_{ctx.room.name}_{timestamp}.json"
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(report_dict, f, indent=2, ensure_ascii=False)
                
            logger.info(f"Successfully saved session log to: {filename}")
        except Exception as e:
            logger.error(f"Failed to save chat log: {e}")

    ctx.add_shutdown_callback(save_chat_log)

    await session.start(room=ctx.room, agent=VoiceAssistant())
    await session.generate_reply(instructions="Say a short hello to the room as Valerie and ask who is joining the meeting today.")
    
if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))