
from moirai.common.database.base import Base

from moirai.common.database import engine

from moirai.common.utils.logging import setup_logger

logger = setup_logger(__name__)

def init_db():

    Base.metadata.create_all(bind=engine)
    
logger.info("Database initialized successfully")

if __name__ == "__main__":

    init_db()
