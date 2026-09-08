"""Shared utilities with lazy access to the optional communication stack."""

__all__ = ["AgentClient", "AgentServer"]


def __getattr__(name):
    if name == "AgentClient":
        from .comm_utils.client import AgentClient

        return AgentClient
    if name == "AgentServer":
        from .comm_utils.server import AgentServer

        return AgentServer
    raise AttributeError(name)
