import dataclasses
from collections import defaultdict
from typing import Dict, FrozenSet, Generic, Hashable, Iterable, Optional, Set, Tuple, TypeVar

Symbol = TypeVar("Symbol", bound=Hashable)


@dataclasses.dataclass
class DFAState(Generic[Symbol]):
    is_final: bool = False
    transition: Dict[Symbol, "DFAState"] = dataclasses.field(default_factory=dict)

    def __str__(self) -> str:
        representation = f"DFAState {hex(id(self))}"
        if self.is_final:
            representation += " (final)"
        return representation

    def __hash__(self) -> int:
        return id(self)

    def acceptable_symbols(self) -> Set[Symbol]:
        """
        Return acceptable symbols for this state.
        """

        return set(self.transition)

    def show(self) -> None:
        visited = set()
        queue = [self]
        while queue:
            state = queue.pop(0)
            if state in visited:
                continue
            visited.add(state)
            print(state)
            for symbol, next_state in state.transition.items():
                print(f"  {symbol} -> {next_state}")
                queue.append(next_state)


class DFA(Generic[Symbol]):
    """
    A deterministic finite automaton (DFA)
    """

    @classmethod
    def from_symbol(cls, symbol: Symbol) -> "DFA":
        """
        Create a DFA that accepts a single symbol.
        """
        start = DFAState[Symbol]()
        end = DFAState[Symbol](is_final=True)
        start.transition[symbol] = end
        return cls(start)

    def __init__(self, state: DFAState) -> None:
        self.state = state

    def __repr__(self) -> str:
        return f"DFA(state={self.state})"

    def __str__(self) -> str:
        return f"DFA(state={self.state})"

    def step(self, state: DFAState[Symbol], symbol: Symbol) -> Optional["DFAState"]:
        return state.transition.get(symbol)

    def accepts(self, symbols: Iterable[Symbol]) -> bool:
        state: Optional[DFAState] = self.state
        for symbol in symbols:
            assert state is not None
            state = self.step(state, symbol)
            if state is None:
                return False
        return state.is_final if state else False

    def show(self) -> None:
        visited = set()
        queue = [self.state]
        while queue:
            state = queue.pop(0)
            if state in visited:
                continue
            visited.add(state)
            print(state)
            for symbol, next_state in state.transition.items():
                print(f"  {symbol} -> {next_state}")
                queue.append(next_state)


@dataclasses.dataclass
class NFAState(Generic[Symbol]):
    is_final: bool = False
    transition: Dict[Symbol, "NFAState"] = dataclasses.field(default_factory=dict)
    epsilon_transitions: Set["NFAState"] = dataclasses.field(default_factory=set)

    def __str__(self) -> str:
        representation = f"NFAState {hex(id(self))}"
        if self.is_final:
            representation += " (final)"
        return representation

    def __hash__(self) -> int:
        return id(self)

    def allowed_transitions(self) -> Dict[Symbol, Set["NFAState"]]:
        """
        Return allowed transitions for this state.
        """

        transitions = defaultdict(set)
        stack: Set[Tuple[Optional[Symbol], NFAState]] = {(None, self)}

        while stack:
            prev_symbol, state = stack.pop()
            if prev_symbol is None:
                for symbol, next_state in state.transition.items():
                    transitions[symbol].add(next_state)
                    stack.update((symbol, x) for x in next_state.epsilon_transitions)
            else:
                transitions[prev_symbol].add(state)
            stack.update((prev_symbol, next_state) for next_state in state.epsilon_transitions)

        return {
            symbol: {state for state in states if state.transition or state.is_final}
            for symbol, states in transitions.items()
        }


class NFA(Generic[Symbol]):
    """
    A nondeterministic finite automaton (NFA)
    """

    @classmethod
    def from_epsilon(cls) -> "NFA[Symbol]":
        """
        Create an NFA that accepts an empty string.
        """
        start = NFAState[Symbol]()
        end = NFAState[Symbol](is_final=True)
        start.epsilon_transitions.add(end)
        return cls(start, end)

    @classmethod
    def from_symbol(cls, symbol: Symbol) -> "NFA":
        """
        Create an NFA that accepts a single symbol.
        """
        start = NFAState[Symbol]()
        end = NFAState[Symbol](is_final=True)
        start.transition[symbol] = end
        return cls(start, end)

    def __init__(self, start: NFAState, end: NFAState) -> None:
        self.start = start
        self.end = end

    def __repr__(self) -> str:
        return f"NFA(start={self.start}, end={self.end})"

    def __str__(self) -> str:
        return f"NFA(start={self.start}, end={self.end})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, NFA):
            return self.start == other.start and self.end == other.end
        return False

    def __hash__(self) -> int:
        return hash((self.start, self.end))

    def __add__(self, other: "NFA[Symbol]") -> "NFA[Symbol]":
        """
        Create an NFA that accepts the concatenation of the languages of two NFAs.
        """
        self.end.epsilon_transitions.add(other.start)
        self.end.is_final = False
        return NFA(self.start, other.end)

    def __or__(self, other: "NFA[Symbol]") -> "NFA[Symbol]":
        """
        Create an NFA that accepts the union of the languages of two NFAs.
        """
        start = NFAState[Symbol]()
        start.epsilon_transitions.add(self.start)
        start.epsilon_transitions.add(other.start)
        end = NFAState[Symbol](is_final=True)
        self.end.epsilon_transitions.add(end)
        other.end.epsilon_transitions.add(end)
        self.end.is_final = False
        other.end.is_final = False
        return NFA(start, end)

    def closure(self) -> "NFA[Symbol]":
        """
        Create an NFA that accepts the closure of the language of an NFA.
        """
        start = NFAState[Symbol]()
        end = NFAState[Symbol](is_final=True)
        start.epsilon_transitions.add(end)
        start.epsilon_transitions.add(self.start)
        self.end.epsilon_transitions.add(self.start)
        self.end.epsilon_transitions.add(end)
        self.end.is_final = False
        return NFA(start, end)

    def compile(self) -> DFA[Symbol]:
        """
        Compile a NFA into a DFA.
        """
        visited = set()
        queue = [frozenset({self.start})]

        transitions: Dict[FrozenSet[NFAState[Symbol]], Dict[Symbol, Set[NFAState[Symbol]]]] = defaultdict(dict)

        while queue:
            states = queue.pop(0)
            if states in visited:
                continue
            visited.add(states)

            allowed_transitions = defaultdict(set)
            for state in states:
                for symbol, next_states in state.allowed_transitions().items():
                    allowed_transitions[symbol].update(next_states)
                    queue.append(frozenset(next_states))
            transitions[states] = allowed_transitions

        # TODO: minimize DFA

        dfa_states: Dict[FrozenSet[NFAState[Symbol]], DFAState[Symbol]] = {}
        for nfa_states, nfa_transitions in transitions.items():
            for symbol, next_nfa_states in nfa_transitions.items():
                if frozenset(nfa_states) not in dfa_states:
                    dfa_states[frozenset(nfa_states)] = DFAState[Symbol](
                        is_final=any(state.is_final for state in nfa_states)
                    )
                if frozenset(next_nfa_states) not in dfa_states:
                    dfa_states[frozenset(next_nfa_states)] = DFAState[Symbol](
                        is_final=any(state.is_final for state in next_nfa_states)
                    )

                start = dfa_states[frozenset(nfa_states)]
                end = dfa_states[frozenset(next_nfa_states)]

                start.transition[symbol] = end

        return DFA(dfa_states[frozenset({self.start})])

    def show(self) -> None:
        """
        Print the NFA in a human-readable format.
        """
        visited = set()
        queue = [self.start]
        while queue:
            state = queue.pop(0)
            if state in visited:
                continue
            visited.add(state)
            print(state)
            for symbol, next_state in state.transition.items():
                print(f"  {symbol} -> {next_state}")
                queue.append(next_state)
            for next_state in state.epsilon_transitions:
                print(f"  (ε) -> {next_state}")
                queue.append(next_state)
