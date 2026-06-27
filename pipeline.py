import subprocess
import sys
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def run_pipeline():
    """
    Run the full recommendation system pipeline:
    1. Preprocessing - Load raw data, engineer features, temporal split
    2. Training - Train GRU4Rec model with early stopping
    3. Evaluation - Run full evaluation with all production metrics
    """
    
    steps = [
        ("Preprocessing", "preprocessing.py"),
        ("Training", "train.py"),
        ("Evaluation", "main.py"),
    ]
    
    for name, script in steps:
        logger.info("=" * 70)
        logger.info(f"Starting: {name}")
        logger.info("=" * 70)
        
        script_path = Path(__file__).parent / script
        
        if not script_path.exists():
            logger.error(f"Script not found: {script_path}")
            sys.exit(1)
        
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True
        )
        
        # Print stdout in real-time (optional, but helpful)
        if result.stdout:
            logger.info(result.stdout)
        
        if result.stderr:
            logger.warning(result.stderr)
        
        if result.returncode != 0:
            logger.error(f"❌ {name} failed with error code {result.returncode}")
            sys.exit(result.returncode)
        
        logger.info(f"✅ {name} completed successfully")
        logger.info("")
    
    logger.info("=" * 70)
    logger.info("🎯 Pipeline completed successfully!")
    logger.info("=" * 70)


if __name__ == "__main__":
    run_pipeline()