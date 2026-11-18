import asyncio
import re
import logging
from datetime import datetime
from etherpad import Etherpad
from openai import AsyncOpenAI
from livekit.agents import (
    AgentSession,
    Agent
)
from livekit.agents.llm import StopResponse

from prompts import VALERIE_SYSTEM_PROMPT, WATCHDOG_PROMPT

logger = logging.getLogger(__name__)

class Valerie(Agent):
    def __init__(self, canvas_manager: Etherpad, openai_client: AsyncOpenAI, system_prompt: str = VALERIE_SYSTEM_PROMPT, watchdog_prompt: str = WATCHDOG_PROMPT):
        self.watchdog_prompt = watchdog_prompt
        self.system_prompt = system_prompt
        self.canvas_manager = canvas_manager
        self.openai_client = openai_client
        self.transcript_history = []
        self.last_speech_time = datetime.now()
        self.session_ref: AgentSession = None
        self.is_ai_speaking = False
        self.last_valerie_speech_time = datetime.now() # last time Valerie spoke, to avoid overlapping interventions
        self.current_trigger_reason = "Direct Interaction" # Reason Valerie is speaking, for logging purposes
        super().__init__(
            instructions=system_prompt,
            allow_interruptions=False,
        )


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
                    await self.session_ref.generate_reply(instructions=prompt, allow_interruptions=False)
                    
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
                        {"role": "system", "content": self.watchdog_prompt},
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
                    await self.session_ref.generate_reply(instructions=intervention_prompt, allow_interruptions=False)
                    
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
                    
                    await self.session_ref.generate_reply(instructions=prompt, allow_interruptions=False)
                    
                except Exception as e:
                    logger.error(f"Error in Presence Watchdog check: {e}")
                finally:
                    self.is_ai_speaking = False
                    self.last_speech_time = datetime.now()
                    self.last_valerie_speech_time = datetime.now()
