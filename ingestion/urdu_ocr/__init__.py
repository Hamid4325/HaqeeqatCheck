"""UTRNet + YOLOv8 end-to-end Urdu OCR package (lazy exports).

Heavy imports (torch, ultralytics) happen only when the engine is first used,
so importing this package stays fast.
"""

__all__ = ["UTRNetOCREngine"]


def __getattr__(name):
    if name == "UTRNetOCREngine":
        from ingestion.urdu_ocr.urdu_ocr_engine import UTRNetOCREngine

        return UTRNetOCREngine
    raise AttributeError("module %r has no attribute %r" % (__name__, name))
