from nlpstack.common import DFA, NFA


def test_automaton() -> None:
    dfa = (
        NFA.from_symbol("a") + (NFA.from_symbol("b") | NFA.from_symbol("c")).closure() + NFA.from_symbol("d")
    ).compile()

    assert isinstance(dfa, DFA)
    assert dfa.accepts("abd")
    assert dfa.accepts("accbd")
    assert not dfa.accepts("ad")
    assert not dfa.accepts("dca")
