import random

from .transition import Transition, NullTransition

class FAState(object):

    def __init__(self, parent=None):
        self.parent = parent
        self.transitions = []
        self.count = 0
        self.label = None
        if self.parent:
            self.id = self.parent.new_state_id()
            self.parent.states[self.id] = self

    def transition_to(self, toks, at):
        """Return collection of all the states reachable from this state 
        by one (non-null) transition matching the item, 
        (followed by any number of null transitions)."""
        states = set(t.dest for t in self.transitions if t.matches(toks, at))
        return self.parent.transitive_closure(states)

    def transit_arcs_to(self, toks, at):
        """Return collection of all the states reachable from this state 
        by one (non-null) transition matching the item, 
        (followed by any number of null transitions)."""
        next_states = set(((s,toki)
                           for t in self.transitions 
                           for toki in t.arc_matches(toks, at)
                           for s in self.parent.transitive_closure((t.dest,))))
        return next_states

    def null_transition_to(self):
        """Return collection of all the states reachable from this state 
        by one or more null transitions."""
        states = set(t.dest for t in self.transitions if isinstance(t, NullTransition))
        return self.parent.transitive_closure(states)  # null transitions only!
            
    def add_transition(self, item, ostate=None):
        if ostate:
            new_state = self.parent.get_state(ostate)
        else:
            new_state = FAState(parent=self.parent)
        self.transitions.append(Transition.new(self.parent, self, new_state, item))
        return new_state

    def transition_exists(self, item, ostate):
        existing = [t for t in self.transitions if t.same(item, ostate)]
        return len(existing) > 0

    def add_unique_transition(self, item, ostate=None):
        if ostate:
            new_state = self.parent.get_state(ostate)
        else:
            new_state = FAState(parent=self.parent)
        if not self.transition_exists(item, new_state):
            self.transitions.append(Transition.new(self.parent, self, new_state, item))
        return new_state

    def incr(self):
        self.count += 1

    def list_transitions(self):
        par = self.parent
        return [ (t.symbol, par.get_state(t.dest)) for t in self.transitions ]

    def clone(self, fa):
        new_state = FAState(parent=fa)
        new_state.count = self.count
        new_state.label = self.label
        # new_state.id = self.id
        for t in self.transitions:
            new_state.transitions.append(t.clone())
        return new_state

    def dump_label(self):
        if self.label:
            return "%d(%s)" % (self.id, self.label)
        else:
            return "%d" % self.id

    def generate_to(self):
        result = []
        num_choices = len(self.transitions)
        if self.parent.is_final(self):
            num_choices += 1
        choice = random.randint(0, num_choices - 1)
        if choice == len(self.transitions):
            return None
        t = self.transitions[choice]
        generation = t.generate_to()
        next_state, emits = generation
        result.extend(emits)
        return (next_state, result)

    def __str__(self):
        return str(self.id)  # + " " + super(SimpleClass, self).__str__()


class FACalloutState(FAState):

    def clone(self, fa):
        new_state = FACalloutState(parent=fa)
        new_state.count = self.count
        new_state.label = self.label
        new_state.symbol = self.symbol  # formerly called fa
        return new_state

    def dump_label(self):
        if self.label:
            label = "%d(%s)" % (self.id, self.label)
        else:
            label = "%d" % self.id
        return label + ':' + self.symbol

    def generate_to(self):
        fa = self.parent.parent.fas[self.symbol]
        generation = fa.generate_to()
        if generation is None:
            return None
        final_state, emits = generation
        dest = list(self.null_transition_to())
        choice = random.randint(0, len(dest) - 1)
        return (dest[choice], emits)

    def __str__(self):
        return str(self.id) + ":" + self.symbol  # + " " + super(SimpleClass, self).__str__()

