#!/usr/bin/env python3
"""
Test runner script for the medical assistant bot.
Runs all test scripts in the tests directory.
"""
import os
import sys
import glob
import importlib.util
import asyncio
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def import_module_from_file(file_path):
    """
    Import a module from a file path.
    """
    module_name = os.path.basename(file_path).replace('.py', '')
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def run_tests():
    """
    Run all test scripts in the tests directory.
    """
    # Get the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    tests_dir = os.path.join(current_dir, 'tests')
    
    # Find all test scripts
    test_files = glob.glob(os.path.join(tests_dir, 'test_*.py'))
    
    if not test_files:
        logger.error("No test files found in the tests directory.")
        return
    
    logger.info(f"Found {len(test_files)} test files: {[os.path.basename(f) for f in test_files]}")
    
    # Run each test file
    for test_file in test_files:
        logger.info(f"\n{'='*80}")
        logger.info(f"Running test file: {os.path.basename(test_file)}")
        logger.info(f"{'='*80}\n")
        
        try:
            # Import the module
            module = import_module_from_file(test_file)
            
            # Find and run all async test functions
            test_functions = []
            for attr_name in dir(module):
                if attr_name.startswith('test_') and callable(getattr(module, attr_name)):
                    test_functions.append(getattr(module, attr_name))
            
            if not test_functions:
                logger.warning(f"No test functions found in {os.path.basename(test_file)}")
                continue
            
            logger.info(f"Found {len(test_functions)} test functions: {[f.__name__ for f in test_functions]}")
            
            # Run each test function
            for test_function in test_functions:
                logger.info(f"\n{'-'*80}")
                logger.info(f"Running test function: {test_function.__name__}")
                logger.info(f"{'-'*80}\n")
                
                try:
                    # Check if the function is async
                    if asyncio.iscoroutinefunction(test_function):
                        asyncio.run(test_function())
                    else:
                        test_function()
                    
                    logger.info(f"Test function {test_function.__name__} completed successfully")
                except Exception as e:
                    logger.error(f"Error running test function {test_function.__name__}: {str(e)}")
        
        except Exception as e:
            logger.error(f"Error importing test file {os.path.basename(test_file)}: {str(e)}")
    
    logger.info("\nAll tests completed")

if __name__ == "__main__":
    run_tests()
