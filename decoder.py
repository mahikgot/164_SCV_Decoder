from __future__ import annotations
from abc import ABC, abstractmethod


class Context:

    def __init__(self, SCV) -> None:
        self._state = None
        self.SCV = SCV
        self.setState(Alpha())

    def setState(self, state: State):
        self._state = state
        self._state.context = self

    def get_HL(x):
        if x <= 29:
            return 0, x
        H = x//30
        L = x - H*30
        return H, L


class State(ABC):
    def __init__(self):
        self._mapping = self.prepare_mapping()

    @property
    def context(self) -> Context:
        return self._context
    @context.setter
    def context(self, context: Context) -> None:
        self._context = context

    @property
    def mapping(self):
        pass
    @mapping.setter
    def mapping(self, map_dict: dict) -> None:
        self._mapping = map_dict

    @abstractmethod
    def prepare_mapping(self) -> dict:
        pass

    def switch(self, state: State) -> None:
        self.context.setState(state)
    def latch(self, state: State) -> None:
        self.context.setState(state)

class Alpha(State):
    def prepare_mapping(self):
        mapping = dict(zip(range(26), range(65,91)))
        mapping[26] = 32
        return mapping

class Punctuation(State):
    def prepare_mapping(self):
        ascii_list = [
                59, 60, 62, 64, 91, 92, 93, 95, 96, 126,
                33, 13, 9, 44, 58, 10, 45, 46, 36, 47, 34,
                124, 42, 40, 41, 63, 123, 125, 39
                ]
        mapping = dict(zip(range(29), ascii_list))
        return mapping

class Lower(State):
    def prepare_mapping(self):
        mapping = dict(zip(range(26), range(97, 123)))
        mapping[26] = 32
        return mapping

class Mixed(State):
    def prepare_mapping(self):
        mapping = dict(zip(range(10), range(48, 58)))
        ascii_list = [
                38, 13, 9, 44, 58, 35, 45, 46, 36, 47, 43,
                37, 42, 61, 94
                ]
        mapping.update(dict(zip(range(10, 25), ascii_list)))
        mapping[26] = 32
        return mapping

Punctuation()
Lower()
Alpha()
Mixed()
