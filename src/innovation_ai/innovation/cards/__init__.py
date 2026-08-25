"""Declarative card effect programs, one module per card.

Age packages exist purely for navigation. Each card module owns exactly one card and is
discovered by :mod:`innovation_ai.innovation.effects.registry`; nothing here imports a sibling
card, a registry list, or a mutation primitive.
"""
