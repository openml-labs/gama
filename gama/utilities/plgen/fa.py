import re

from .state import FAState
from .transition import NullTransition

# Represents an interconnected set of states and transitions that 
# can be used to recognize a matching sequence of tokens.
# Key recognition operations are match, search, and scan; 
# see FiniteAutomatonManager for an overview of what those mean.
class FiniteAutomaton(object):

    counter = 0

    # Instance variables:
    # name -- Typically named after LHS of pattern.
    # parent -- FAManager we belong to; needed to find other FAs we depend on.
    # states -- Map from state id (an integer representing a string) to state.
    # initial -- Initial state.
    # final -- Map from state to whether it's final.

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)
        self.states = {}
        self.initial = FAState(parent=self)
        self.final = { self.initial.id }
        
    @classmethod
    def new_state_id(cls):
        cls.counter += 1
        return cls.counter

    def get_state(self, state):
        try:
            sid = state.id
        except:
            sid = state
        return self.states[sid]

    def make_final(self, state, final=True):
        try:
            sid = state.id
        except:  # State ID
            sid = state
        if final:
            self.final.add(sid)
        else:
            self.final.discard(sid)

    def is_final(self, state):
        try:
            sid = state.id
        except:
            sid = state
        return sid in self.final

    def get_final_states(self):
        return [self.states[s] for s in self.final]

    # The methods atom, concat, altern, star/plus/opt are typically used
    # to initialize a newly created FA.
    # They're called from, eg, the fa() methods of FARegex's.

    def atom(self, symbol, extract_target):
        final = self.initial.add_transition(symbol)
        self.make_final(self.initial, False)
        self.make_final(final)
        final.extract_target = extract_target
        return self

    def concat(self, fas):
        """Appears to roughly copy the other FAs and assemble them into a larger
        one."""
        for fa in fas:
            init = fa.initial
            final = self.get_final_states()
            statemap = self.eat(fa)
            init = statemap[init.id]
            for state in final:
                state.add_unique_transition(None, init)
                self.make_final(state, False)
            for state in fa.get_final_states():
                self.make_final(statemap[state.id])
        return self

    def altern(self, fas):
        final = self.get_final_states()
        for state in final:
            self.make_final(state, False)
        for fa in fas:
            init = fa.initial
            statemap = self.eat(fa)
            init = statemap[init.id]
            for state in final:
                state.add_unique_transition(None, init)
            for state in fa.get_final_states():
                self.make_final(statemap[state.id])
        return self

    def star(self):
        return self.plus().opt()

    def plus(self):
        init = self.initial
        final = self.get_final_states()
        for state in final:
            state.add_unique_transition(None, init)
        return self

    def opt(self):
        init = self.initial
        final = self.get_final_states()
        for state in final:
            init.add_unique_transition(None, state)
        return self

    def range(self, mn, mx):
        if mn > mx:
            raise ValueError("Range minimum (%d) is greater than maximum (%d)" % (mn, mx))
        if mx == 0:
            return self.opt()
        # Create a sequence of FAs of length mx
        copies = [ self.clone() for _ in range(mx-1) ]
        inits = [self.initial.id]
        for fa in copies:
            init = fa.initial
            final = self.get_final_states()
            statemap = self.eat(fa)
            init = statemap[init.id]
            inits.append(init.id)
            for state in final:
                state.add_unique_transition(None, init)
                self.make_final(state, False)
            for state in fa.get_final_states():
                self.make_final(statemap[state.id])
        # Add shortcut connections for all intermediate points greater than mn
        final = self.get_final_states()
        for i in range(mn, len(inits)):
            init = self.get_state(inits[i])
            for state in final:
                init.add_unique_transition(None, state)
        return self



    def eat(self, fa):
        statemap = dict((s.id, s.clone(self))
                        for s in fa.states.values())
        for sid, newstate in statemap.items():
            state = fa.get_state(sid)
            newstate.transitions = []
            for trans in state.transitions:
                newtrans = trans.clone()
                newtrans.src = newstate.id
                newtrans.dest = statemap[trans.dest].id
                newtrans.fa = self
                newstate.transitions.append(newtrans)
        return statemap

    def dump(self):

        visited = {}

        def do_dump(state):
            """Print name of state, preceded by > for an initial state or @ for a final state, 
            followed by the state's transitions, and then recurse into the transitions' 
            destination states. 
            @param visited -- used to track which transitions have been visited"""
            visited[state.id] = True
            if state == self.initial:
                print(">", end="")
            if self.is_final(state):
                print("@", end="")
            print(state.dump_label())
            for trans in state.transitions:
                trans.dump("  ")
            for trans in state.transitions:
                if trans.dest not in visited:
                    do_dump(self.get_state(trans.dest))

        do_dump(self.initial)

    def loopy(self):
        """Methods that check for loops, used in debugging patterns."""

        visited = {}

        def loopy_at_state(state):
            if state.id in visited:
                return True
            visited[state.id] = True
            for trans in state.transitions:
                next_state = self.get_state(trans.dest)
                if loopy_at_state(next_state):
                    return True
            return False

        return loopy_at_state(self.initial)

    def clone(self):
        fa = FiniteAutomaton(manager=self.manager)
        statemap = dict((id, s.clone(fa)) for id,s in self.states.items())
        fa.states = dict((s.id, s) for s in statemap.values())
        fa.initial = statemap[ self.initial.id ]
        fa.final = set(statemap[sid].id for sid in self.final)
        # Patch the transitions
        for s in fa.states.values():
            for t in s.transitions:
                t.src = statemap[t.src].id
                t.dest = statemap[t.dest].id
        return fa

    def transitive_closure(self, states):
        """ Gives the transitive closure UNDER NULL TRANSITIONS ONLY of the
        collection of states."""
        states = set(states)
        # First follow all null transitions from this state.
        newstates = set(t.dest for s in states for t in self.states[s].transitions
                        if isinstance(t, NullTransition) and t.dest not in states)
        # Now quasi-recursively follow all null transitions, except ones 
        # from callout states, since we've not descended into those yet.
        while len(newstates) > 0:
            states |= newstates
            newstates = (self.states[s] for s in newstates)
            newstates = set(t.dest for s in newstates for t in s.transitions
                            if isinstance(t, NullTransition) and t.dest not in states)

        return states


    def matches(self, toks):

        def matches_at(toks, at, sid):
            tlen = len(toks)
            tc = self.transitive_closure(set((sid,)))
            # If we're at the end of the sequence, we match if we're at any final state
            if at == tlen:
                return any(s in self.final for s in tc)
            else:
                for tc_sid in tc:
                    state = self.get_state(tc_sid)
                    dest = state.transition_to(toks, at)
                    for next_sid in self.transitive_closure(dest):
                        if matches_at(toks, at + 1, next_sid):
                            return True
                return False

        return matches_at(toks, 0, self.initial.id)


    def generate_to(self):
        state = self.initial
        result = []
        trans = state.generate_to()
        while trans is not None:
            next_state, emits = trans
            result.extend(emits)
            state = self.states[next_state]
            trans = state.generate_to()
        return (state, result)

    def generate(self):
        state, emits = self.generate_to()
        return emits

    def __str__(self):
        return object.__str__(self) + "[name=" + (self.name if hasattr(self, "name") else "None") + "]"


