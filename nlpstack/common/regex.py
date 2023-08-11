import dataclasses
from collections import defaultdict
from functools import reduce
from typing import Dict, Iterable, List, Sequence, Set


class SimpleRegex:
    @dataclasses.dataclass
    class State:
        is_accepting: bool
        transition: Dict[str, "SimpleRegex.State"] = dataclasses.field(default_factory=dict)
        epsilon_transitions: List["SimpleRegex.State"] = dataclasses.field(default_factory=list)

    @dataclasses.dataclass
    class NFA:
        start: "SimpleRegex.State"
        end: "SimpleRegex.State"

    def __init__(self, pattern: str, vocab: Iterable[str]) -> None:
        self._pattern = pattern
        self._vocab = set(vocab)
        self._nfa = self._compile(self._pattern, self._vocab)

    def match(self, tokens: Sequence[str]) -> bool:
        def add_next_state(
            state: SimpleRegex.State,
            next_states: List[SimpleRegex.State],
            visited: List[SimpleRegex.State],
        ) -> None:
            if state.epsilon_transitions:
                for target in state.epsilon_transitions:
                    if target not in visited:
                        visited.append(target)
                        add_next_state(
                            target,
                            next_states,
                            visited,
                        )
            else:
                next_states.append(state)

        current_states: List[SimpleRegex.State] = []
        add_next_state(self._nfa.start, current_states, [])

        for token in tokens:
            if token not in self._vocab:
                return False
            next_states: List[SimpleRegex.State] = []
            for state in current_states:
                next_state = state.transition.get(token)
                if next_state:
                    add_next_state(next_state, next_states, [])
            current_states = next_states

        return any(state.is_accepting for state in current_states)

    @staticmethod
    def _compile(pattern: str, vocab: Set[str]) -> "SimpleRegex.NFA":
        def add_epsilon_transition(source: SimpleRegex.State, target: SimpleRegex.State) -> None:
            source.epsilon_transitions.append(target)

        def add_transition(source: SimpleRegex.State, target: SimpleRegex.State, token: str) -> None:
            source.transition[token] = target

        def from_epsilon() -> SimpleRegex.NFA:
            source = SimpleRegex.State(False)
            target = SimpleRegex.State(True)
            add_epsilon_transition(source, target)
            return SimpleRegex.NFA(source, target)

        def from_token(token: str) -> SimpleRegex.NFA:
            source = SimpleRegex.State(False)
            target = SimpleRegex.State(True)
            add_transition(source, target, token)
            return SimpleRegex.NFA(source, target)

        def concat(left: SimpleRegex.NFA, right: SimpleRegex.NFA) -> SimpleRegex.NFA:
            add_epsilon_transition(left.end, right.start)
            left.end.is_accepting = False
            return SimpleRegex.NFA(left.start, right.end)

        def union(left: SimpleRegex.NFA, right: SimpleRegex.NFA) -> SimpleRegex.NFA:
            start = SimpleRegex.State(False)
            add_epsilon_transition(start, left.start)
            add_epsilon_transition(start, right.start)

            end = SimpleRegex.State(True)
            add_epsilon_transition(left.end, end)
            add_epsilon_transition(right.end, end)

            left.end.is_accepting = False
            right.end.is_accepting = False

            return SimpleRegex.NFA(start, end)

        def closure(nfa: SimpleRegex.NFA) -> SimpleRegex.NFA:
            start = SimpleRegex.State(False)
            end = SimpleRegex.State(True)

            add_epsilon_transition(start, end)
            add_epsilon_transition(start, nfa.start)
            add_epsilon_transition(nfa.end, end)
            add_epsilon_transition(nfa.end, nfa.start)
            nfa.end.is_accepting = False

            return SimpleRegex.NFA(start, end)

        if not pattern:
            return from_epsilon()

        max_token_length = max(len(token) for token in vocab)

        index: int = 0
        stack: List[SimpleRegex.NFA] = []

        while index < len(pattern):
            if pattern[index] == "*":
                stack.append(closure(stack.pop()))
                index += 1
            elif pattern[index] == "|":
                right = stack.pop()
                left = stack.pop()
                stack.append(union(left, right))
                index += 1
            elif pattern[index] == ".":
                right = stack.pop()
                left = stack.pop()
                stack.append(concat(left, right))
                index += 1
            else:
                tail = index + 1
                backpointer: Dict[int, List[SimpleRegex.NFA]] = defaultdict(list)
                while index < len(pattern):
                    if tail - 1 >= 0 and pattern[tail - 1] in ("*", "|", "."):
                        if tail - 2 < 0 or pattern[tail - 2] != "\\":
                            break
                    chunk = pattern[index:tail]
                    if chunk in vocab:
                        backpointer[index].append(from_token(chunk))
                    if tail - index == max_token_length or tail == len(pattern):
                        index += 1
                        tail = index
                    tail = min(tail + 1, len(pattern))

                index = tail

                def chain_backpointer(nfa: SimpleRegex.NFA, offset: int) -> SimpleRegex.NFA:
                    for token, state in nfa.start.transition.items():
                        for next_nfa in backpointer[len(token) + offset]:
                            nfa = union(nfa, chain_backpointer(concat(nfa, next_nfa), len(token) + offset))
                    return nfa

                if backpointer:
                    start_index = min(backpointer)
                    stack.append(
                        reduce(union, (chain_backpointer(nfas, start_index) for nfas in backpointer[start_index]))
                    )

        return stack.pop()
