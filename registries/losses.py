"""Compatibility shim for ``neural_trade.registries.losses`` (removed in B18).

This file used to be a stale 924-line copy of the loss module that production never
imported; tests importing it validated code that was no longer running.
"""
from neural_trade._shim import alias_module

alias_module(__name__, "neural_trade.registries.losses")
