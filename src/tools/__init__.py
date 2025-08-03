# Tools module - only import what exists
try:
    from .content_extractor import ContentExtractorTool, SearchTool, DeepLinkTool

    CONTENT_EXTRACTOR_AVAILABLE = True
except ImportError:
    CONTENT_EXTRACTOR_AVAILABLE = False
    ContentExtractorTool = None
    SearchTool = None
    DeepLinkTool = None

try:
    from .simple_search import RealSearchTool

    SIMPLE_SEARCH_AVAILABLE = True
except ImportError:
    SIMPLE_SEARCH_AVAILABLE = False
    RealSearchTool = None

__all__ = []

if CONTENT_EXTRACTOR_AVAILABLE:
    __all__.extend(["ContentExtractorTool", "SearchTool", "DeepLinkTool"])

if SIMPLE_SEARCH_AVAILABLE:
    __all__.append("RealSearchTool")
