"""
Runs the full pipeline in order: ingest → features → train.
"""

from src.pipeline.ingest   import ingest
from src.pipeline.features import run_features
from src.pipeline.train    import train
from src.utils.logger      import get_logger

logger = get_logger("pipeline")


def main():
    logger.info("Step 1/3 — ingesting data")
    ingest()

    logger.info("Step 2/3 — engineering features")
    run_features()

    logger.info("Step 3/3 — training model")
    train()

    logger.info("Done. Start the API with: python -m uvicorn src.api.main:app --reload")


if __name__ == "__main__":
    main()
