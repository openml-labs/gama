
# Note that the 'symbol' member means different things for different subclasses.
class Transition(object):
    """Abstract base class for an FA transition that can occur between source and 
    destination states (referenced by name) when an input token (aka symbol) 
    is seen.
    (NullTransition is a special case that does not require an input token.)"""

    @staticmethod
    def new(fa, src, dest, emit=None):
        if emit is None:
            return NullTransition(fa=fa, src=src, dest=dest, symbol=None)
        else:
            return TestTransition(fa=fa, src=src, dest=dest, symbol=emit)

    def __init__(self, fa=None, src=None, dest=None, symbol=None):
        self.fa = fa
        self.symbol = symbol
        if src is not None:        
            self.src = self.state_id(src)
        if dest is not None:
            self.dest = self.state_id(dest)

    def state_id(self, state):
        try:
            return state.id
        except:
            return state

    def same(self, item, state):
        return item == self.symbol and self.state_id(state) == self.dest

    def dump(self, leading_str=""):
        print("%s(%s) -> %s" % (leading_str, self.symbol, self.dest))


# 'symbol' member can be a literal token value; if seen, the transition can happen.
# Can also (for "@" expresssion reference "callout" transitions) be the name (LHS) 
# of the expresson.
class SymbolTransition(Transition):

    def matches(self, toks, at):
        return toks[at] == self.symbol

    def clone(self):
        return SymbolTransition(fa=self.fa, symbol=self.symbol, src=self.src, dest=self.dest)

    def generate_to(self):
        return (self.dest, [self.symbol])


# 'symbol' member is None or absent, and is not used.
class NullTransition(Transition):
    
    def matches(self, toks, at):
        return False

    def clone(self):
        return NullTransition(fa=self.fa, symbol=None, src=self.src, dest=self.dest)

    def generate_to(self):
        return (self.dest, [])


# 'symbol' member is the name (LHS) of the token test.
class TestTransition(Transition):

    def matches(self, toks, at):
        test = self.fa.parent.get_test(self.symbol)
#        test = self.fa.parent.tests[self.symbol]
        return test.matches(toks[at])

    def dump(self, leading_str=""):
        print("%s<%s> -> %s" % (leading_str, self.symbol, self.dest))

    def clone(self):
        return TestTransition(fa=self.fa, symbol=self.symbol, src=self.src, dest=self.dest)

    def generate_to(self):
        test = self.fa.manager.tests[self.symbol]
        emit = test.generate()
        return (self.dest, [emit])
