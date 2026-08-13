"""Same-query, decoupled style attention for Anima."""

from .adapter import SameQFullRankStyleAdapter, attach_same_q_style_adapter

__all__ = ["SameQFullRankStyleAdapter", "attach_same_q_style_adapter"]
