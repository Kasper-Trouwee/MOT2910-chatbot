from py_etherpad import EtherpadLiteClient
import asyncio
import os
import logging
import webbrowser

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
    def __init__(self, pad_id: str = "valerie-brainstorm", prompt: str = ""):
        """Initialize the manager and schedule creation of its pad.

        Args:
            pad_id: Etherpad pad identifier. Defaults to
                ``"valerie-brainstorm"``.
            prompt: Initial text to set on the pad.

        Raises:
            ValueError: If ``ETHERPAD_API_KEY`` is not set.
        """
        self.pad_id = pad_id
        
        api_key = os.getenv("ETHERPAD_API_KEY")
        if not api_key:
            raise ValueError("ETHERPAD_API_KEY environment variable is not set. Please provide your Etherpad API key.")
            
        self.client = EtherpadLiteClient(api_key, 'http://localhost:9001/api')
        self.cached_text = "(Shared canvas currently empty)"
        self.client.setText(prompt);

        asyncio.create_task(self.create_pad_if_missing())
        
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
            webbrowser.open(f"http://localhost:9001/p/{self.pad_id}")
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
            webbrowser.open(f"http://localhost:9001/p/{self.pad_id}/export/{type}")
        except Exception as e:
            logger.error(f"Error creating Etherpad: {e}")


