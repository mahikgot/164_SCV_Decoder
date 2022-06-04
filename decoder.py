from __future__ import annotations
from abc import ABC, abstractmethod


class Context:
    #State Machine Global Context
    def __init__(self, SCV) -> None:
        self._state = None
        self.SCV = SCV
        self.setState(Alpha())

    def setState(self, state: State):
        self._state = state
        self._state.context = self

    def get_HL(self, x):
        if x <= 29:
            return 0, x
        H = x//30
        if H >= 30:
            H = 29
        L = x - H*30
        return H, L

    def get_data(self, scv_list):
        output = []
        for i in range(scv_list[0] - 1):
            if scv_list[i+1] == 900:
                continue
            output.extend(self.get_HL(scv_list[i+1]))

        if output[-1] == 29:
            output.pop()
        return output

    def decode(self):
        decoded = ''
        converted = self.get_data(self.SCV)
        for element in converted:
            #state.decode method handles state switching and latching
            decoded = self._state.decode(decoded, element)
        return decoded

class State(ABC):
    #Abstract Class named State for the different Submodes
    #Each State have a reference to the Context Class, being able to switch its state
    #Each have their own mapping dictionary
    #Decode methods are different for each submode, this method also handles switching and latching between states
    def __init__(self):
        self.mapping = self.prepare_mapping()
        self.switched = False
        self.lastclass = None

    @property
    def lastclass(self) -> str:
     return self._lastclass
    @lastclass.setter
    def lastclass(self, last: str) -> None:
        self._lastclass = last

    @property
    def switched(self) -> bool:
        return self._switched
    @switched.setter
    def switched(self, from_switch: bool) -> None:
        self._switched = from_switch

    @property
    def context(self) -> Context:
        return self._context
    @context.setter
    def context(self, context: Context) -> None:
        self._context = context

    @property
    def mapping(self) -> dict:
        return self._mapping
    @mapping.setter
    def mapping(self, map_dict: dict) -> None:
        self._mapping = map_dict

    @abstractmethod
    def prepare_mapping(self) -> dict:
        pass

    @abstractmethod
    def decode(self, output: str, num: int) -> None:
        pass

    def switch(self, state: State) -> None:
        state.switched = True
        state.lastclass = self.__class__.__name__
        self.context.setState(state)
    def switch_back(self) -> None:
        self.context.setState(globals()[self.lastclass]())

    def latch(self, state: State) -> None:
        self.context.setState(state)

class Alpha(State):
    def prepare_mapping(self):
        mapping = dict(zip(range(26), range(65,91)))
        mapping[26] = 32
        return mapping

    def decode(self, output, num):
        if num == 27:
            self.latch(Lower())
        elif num == 28:
            self.latch(Mixed())
        elif num == 29:
            self.switch(Punctuation())
        else:
            decoded = self.mapping.get(num)
            output += chr(decoded)

        if self.switched:
            self.switch_back()
        return output


class Punctuation(State):
    def prepare_mapping(self):
        ascii_list = [
                59, 60, 62, 64, 91, 92, 93, 95, 96, 126,
                33, 13, 9, 44, 58, 10, 45, 46, 36, 47, 34,
                124, 42, 40, 41, 63, 123, 125, 39
                ]
        mapping = dict(zip(range(29), ascii_list))
        return mapping

    def decode(self, output, num):
        if num == 29:
            self.Latch(Alpha())
        else:
            decoded = self.mapping.get(num)
            output += chr(decoded)

        if self.switched:
            self.switch_back()
        return output

class Lower(State):
    def prepare_mapping(self):
        mapping = dict(zip(range(26), range(97, 123)))
        mapping[26] = 32
        return mapping

    def decode(self, output, num):
        if num == 27:
            self.switch(Alpha())
        elif num == 28:
            self.latch(Mixed())
        elif num == 29:
            self.switch(Punctuation())
        else:
            decoded = self.mapping.get(num)
            output += chr(decoded)

        if self.switched:
            self.switch_back()
        return output

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

    def decode(self, output, num):
        if num == 27:
            self.latch(Lower())
        elif num == 28:
            self.latch(Alpha())
        elif num == 29:
            self.switch(Punctuation())
        else:
            decoded = self.mapping.get(num)
            output += chr(decoded)

        if self.switched:
            self.switch_back()
        return output

# print(Context([10, 893, 864, 877, 749, 739, 496, 844, 393, 900, 822, 22, 716, 545, 596, 130, 458, 768]).decode())

def decodeMsg(scv_list):
    context = Context(scv_list)
    return context.decode()
