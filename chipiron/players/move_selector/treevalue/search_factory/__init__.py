"""
This module provides factories for creating search objects and node selectors.

The factories included in this module are:
- SearchFactoryP: A factory for creating search objects with parallel execution.
- SearchFactory: A factory for creating search objects with sequential execution.
- NodeSelectorFactory: A factory for creating node selectors.

To use this module, import the desired factory class from this module and use it to create the desired objects.
"""

from .search_factory import NodeSelectorFactory, SearchFactory, SearchFactoryP

__all__ = ["SearchFactoryP", "SearchFactory", "NodeSelectorFactory"]
