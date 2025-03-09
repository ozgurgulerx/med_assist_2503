"""
Minimal entry point for Medical Assistant Bot
"""
import asyncio
import logging
from medical_assistant_bot import interactive_conversation

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    try:
        logger.info("Starting Medical Assistant Bot")
        asyncio.run(interactive_conversation())
    except KeyboardInterrupt:
        logger.info("Medical Assistant Bot stopped by user")
    except Exception as e:
        logger.error(f"Error running Medical Assistant Bot: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())