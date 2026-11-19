from py_etherpad import EtherpadLiteClient
import asyncio
import os
import signal
import logging
import webbrowser
import urllib.parse
import socket
import subprocess
import time
from pathlib import Path

# Set up logging
logger = logging.getLogger(__name__)



class Etherpad:
    """Read and update a shared plaintext canvas backed by local Etherpad.

    The Etherpad API key is read from the ``ETHERPAD_API_KEY`` environment
    variable. A pad is created asynchronously when an instance is initialized.

    Attributes:
        pad_id: Etherpad pad identifier to manage.
        client: Client configured for the local Etherpad API.
        cached_text: Most recently retrieved text, or the empty-canvas
            placeholder before text has been retrieved.
    """
    def __init__(self, pad_id: str = "valerie-brainstorm", prompt: str = "", url = "http://localhost:9001"):
        """Initialize the manager and schedule creation of its pad.

        Args:
            pad_id: Etherpad pad identifier. Defaults to
                ``"valerie-brainstorm"``.
            prompt: Initial text to set on the pad.

        Raises:
            ValueError: If ``ETHERPAD_API_KEY`` is not set.
        """
        self.pad_id = pad_id
        self.prompt = prompt
        self.etherpad_dir = Path.home() / "etherpad-lite"

        api_key = os.getenv("ETHERPAD_API_KEY")
        if not api_key:
            raise ValueError("ETHERPAD_API_KEY environment variable is not set. Please provide your Etherpad API key.")

        self.url = url
        self.client = EtherpadLiteClient(api_key, url + '/api')
        self.cached_text = "(Shared canvas currently empty)"

        self.process = None
        running = self.is_etherpad_running()
        if not running:
            self.start_etherpad()

    def is_etherpad_running(self) -> bool:
            """Check if the Etherpad server is listening on the host/port from self.url."""
            parsed = urllib.parse.urlparse(self.url)
            host = parsed.hostname or "127.0.0.1"
            port = parsed.port or 9001

            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(1.0)
                return sock.connect_ex((host, port)) == 0

    def start_etherpad(self, timeout=30):
        """Open the Etherpad in the default web browser."""
        if not self.is_etherpad_running():
            logger.info("Etherpad server is not running. Starting it now...")
            run_script = self.etherpad_dir / "bin" / "run.sh"

            if not run_script.exists():
                logger.error(f"Could not find startup script at {run_script}")
                return

            self.process = subprocess.Popen(
                ["bash", str(run_script)],
                cwd=str(self.etherpad_dir),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )

            # Wait up to 15 seconds for the server to boot
            for _ in range(timeout):
                time.sleep(1)
                if self.is_etherpad_running():
                    logger.info(f"Etherpad server started at {self.url}")
                    break
            else:
                logger.error(
                    f"Timed out waiting for Etherpad server to start at {self.url}."
                )
                return
    async def initialize(self):
        """Ensure server is running, create the pad with initial prompt, and open browser."""
        if not self.is_etherpad_running():
            await asyncio.to_thread(self.start_etherpad)
        await self.create_pad_if_missing()
    
    def stop_etherpad(self, timeout:int = 3):
        """Stop the local Etherpad server process."""
        if not self.is_etherpad_running():
            logger.info("Etherpad server is already stopped.")
            return

        logger.info("Stopping Etherpad server...")

        # 1. If this instance started the process, kill its entire process group
        if self.process and self.process.poll() is None:
            try:
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
                self.process.wait(timeout=timeout)
                self.process = None
                logger.info("Etherpad server stopped cleanly.")
                return
            except Exception as e:
                logger.warning(f"Failed to stop process group cleanly: {e}")

        # 2. Fallback: Find whatever process is listening on the Etherpad port (Linux/macOS)
        parsed = urllib.parse.urlparse(self.url)
        port = parsed.port or 9001

        try:
            # Find PIDs listening on the port
            result = subprocess.run(
                ["lsof", "-t", f"-i:{port}"],
                capture_output=True,
                text=True,
                check=False,
            )
            pids = result.stdout.strip().splitlines()

            for pid_str in pids:
                pid = int(pid_str)
                os.kill(pid, signal.SIGTERM)

            # Wait up to 10 seconds for the port to close
            for _ in range(10):
                time.sleep(1)
                if not self.is_etherpad_running():
                    logger.info("Etherpad server stopped.")
                    return

            # Force kill if it didn't shut down gracefully
            for pid_str in pids:
                os.kill(int(pid_str), signal.SIGKILL)
            logger.info("Etherpad server force-stopped.")

        except FileNotFoundError:
            logger.error("'lsof' command not found; cannot locate Etherpad PID automatically.")
        except Exception as e:
            logger.error(f"Error stopping Etherpad: {e}")

    async def get_latest_text(self) -> str:
        """Fetch and return the latest pad text, retaining the cache on errors."""
        try:
            result = await asyncio.to_thread(self.client.getText, self.pad_id)
            self.cached_text = result.get("text", "").strip()
        except Exception as e:
            logger.error(f"Error reading Etherpad: {e}")
            
        return self.cached_text

    async def append_text(self, new_content: str):
        """Append content to the current pad text.

        Args:
            new_content: Text to add at the end of the pad.
        """
        try:
            current_text = await self.get_latest_text()
            updated_text = f"{current_text}\n{new_content}".strip()
            await asyncio.to_thread(self.client.setText, self.pad_id, updated_text)
            logger.info(f"Appended new content to Etherpad: '{new_content}'")
        except Exception as e:
            logger.error(f"Error appending to Etherpad: {e}")
    
    async def create_pad_if_missing(self):
        """Create the configured pad and open it in a browser.

        Etherpad errors, including the response for an already-existing pad,
        are logged rather than propagated.
        """
        try:
            await asyncio.to_thread(self.client.createPad, self.pad_id)
            logger.info(f"Etherpad '{self.pad_id}' is ready.")
            await asyncio.to_thread(self.client.setText, self.pad_id, self.prompt)

            webbrowser.open(f"{self.url}/p/{self.pad_id}")
        except Exception as e:
            logger.error(f"Error creating Etherpad: {e}")

    async def save_pad_to_file(self, filename: str, type: str = "txt"):
        """Save the current pad text to a local file.

        Args:
            filename: Path to the file where the pad text will be saved.
        """
        try:
            await asyncio.to_thread(self.client.createPad, self.pad_id)
            logger.info(f"Etherpad '{self.pad_id}' is ready.")
            webbrowser.open(f"{self.url}/p/{self.pad_id}/export/{type}")
        except Exception as e:
            logger.error(f"Error creating Etherpad: {e}")