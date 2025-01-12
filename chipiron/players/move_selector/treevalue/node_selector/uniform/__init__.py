"""
This module provides a uniform selection strategy for nodes in a tree.

The Uniform selection strategy selects nodes uniformly at random from the available options.

Example usage:
    from .uniform import Uniform

    uniform_selector = Uniform()
    selected_node = uniform_selector.select_node(nodes)

"""

from .uniform import Uniform

__all__ = ["Uniform"]
