from rag.chunks_managers.base_chunks_manager import BaseChunksManager

class GDocsChunksManager(BaseChunksManager):
    def __init__(self):
        super().__init__("gdoc")