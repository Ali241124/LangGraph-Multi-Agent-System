
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver


# ==========================================
# SQLITE CONFIGURATION
# ==========================================

CHECKPOINT_DIRECTORY = Path("checkpoints")

CHECKPOINT_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True
)

DATABASE_PATH = CHECKPOINT_DIRECTORY / "workflow.sqlite"


# ==========================================
# CREATE CHECKPOINTER CONTEXT
# ==========================================

def get_checkpointer():

    return SqliteSaver.from_conn_string(
        str(DATABASE_PATH)
    )
