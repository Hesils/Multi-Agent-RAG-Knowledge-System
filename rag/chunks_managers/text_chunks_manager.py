from rag.chunks_managers.base_chunks_manager import BaseChunksManager

class TextChunksManager(BaseChunksManager):
    def __init__(self):
        super().__init__("txt")

class MarkdownChunksManager(BaseChunksManager):
    def __init__(self):
        super().__init__("md")