"""
Enhanced logging setup for troubleshooting the medical assistant bot
Save this file as troubleshooting.py in your project root
"""
import os
import logging
import sys
from datetime import datetime
from typing import Optional

def setup_enhanced_logging(log_dir: Optional[str] = None, log_level: str = "DEBUG"):
    """
    Set up enhanced logging with both console and file output
    
    Args:
        log_dir: Directory to store log files (defaults to 'logs' in current directory)
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    # Create logs directory if it doesn't exist
    if log_dir is None:
        log_dir = os.path.join(os.getcwd(), "logs")
    
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Generate timestamp for the log filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"med_assist_{timestamp}.log")
    
    # Set up root logger
    root_logger = logging.getLogger()
    
    # Clear any existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Set log level
    level = getattr(logging, log_level.upper())
    root_logger.setLevel(level)
    
    # Create console handler with formatting
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_format)
    root_logger.addHandler(console_handler)
    
    # Create file handler with detailed formatting
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(pathname)s:%(lineno)d - %(message)s')
    file_handler.setFormatter(file_format)
    root_logger.addHandler(file_handler)
    
    # Log startup message
    root_logger.info(f"Enhanced logging initialized. Log file: {log_file}")
    
    return log_file

def log_system_info():
    """Log basic system information to help with troubleshooting"""
    logger = logging.getLogger(__name__)
    
    # Log Python version
    logger.info(f"Python version: {sys.version}")
    
    # Log environment variables (excluding sensitive ones)
    env_vars = {k: v for k, v in os.environ.items() 
                if not any(s in k.lower() for s in ['key', 'secret', 'password', 'token'])}
    
    logger.info("Environment variables:")
    for key, value in env_vars.items():
        if key.startswith('AZURE') or key.startswith('OPENAI') or key in ['DEBUG_MODE', 'RUN_MODE', 'HOST', 'PORT']:
            # Redact actual values for API keys
            if 'API_KEY' in key:
                logger.info(f"  {key}: [REDACTED]")
            else:
                logger.info(f"  {key}: {value}")
    
    # Check if key environment variables are set
    required_vars = ['AZURE_OPENAI_ENDPOINT', 'AZURE_OPENAI_API_KEY', 'AZURE_OPENAI_DEPLOYMENT_NAME']
    missing_vars = [var for var in required_vars if not os.environ.get(var)]
    
    if missing_vars:
        logger.warning(f"Missing required environment variables: {', '.join(missing_vars)}")
    else:
        logger.info("All required environment variables are set")

if __name__ == "__main__":
    # Test the enhanced logging setup
    log_file = setup_enhanced_logging()
    log_system_info()
    
    logging.info("This is a test info message")
    logging.debug("This is a test debug message")
    logging.warning("This is a test warning message")
    logging.error("This is a test error message")
    
    print(f"Log file created at: {log_file}")