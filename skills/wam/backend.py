"""Preserved WAM import; lifecycle implementation is backend-independent."""
from skills.chunk_backend import CommittedChunkBackend, ExplicitCommitModel

WAMControlBackend = CommittedChunkBackend
