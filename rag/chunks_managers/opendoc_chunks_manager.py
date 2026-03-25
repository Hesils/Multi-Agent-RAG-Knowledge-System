from rag.chunks_managers.base_chunks_manager import BaseChunksManager

class OpenDocsChunksManager(BaseChunksManager):
    def __init__(self):
        super().__init__("odt")