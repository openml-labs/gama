import re

from .fa import FiniteAutomaton

class Regex(object):

    def __init__(self, **kw):
        for k,v in kw.items():
            setattr(self, k, v)

class RegexAtom(Regex):

    def __init__(self, **kw):
        super().__init__(**kw)
        m = re.match('@(\w+(\.\w+)*)$', self.symbol)
        if m:
            self.symbol = m.group(1)
            self.extract_target = True
        else:
            self.extract_target = False

    def fa(self):
        fa = FiniteAutomaton(manager=self.manager)
        fa.atom(self.symbol, self.extract_target)
        return fa

    def dump(self, ind=""):
        print("%sAtom(%s)" % (ind, self.symbol))


class RegexConcat(Regex):
    
    def fa(self):
        fas = [r.fa() for r in self.subs]
        fa = FiniteAutomaton(manager=self.manager)
        fa.concat(fas)
        return fa

    def dump(self, ind=""):
        print("%sConcat" % ind)
        for s in self.subs:
            s.dump(ind + "  ")


class RegexAltern(Regex):
    
    def fa(self):
        fas = [r.fa() for r in self.subs]
        fa = FiniteAutomaton(manager=self.manager)
        fa.altern(fas)
        return fa

    def dump(self, ind=""):
        print("%sAltern" % ind)
        for s in self.subs:
            s.dump(ind + "  ")


class RegexStar(Regex):

    def fa(self):
        return self.sub.fa().star()

    def dump(self, ind=""):
        print("%sStar" % ind)
        self.sub.dump(ind + "  ")


class RegexPlus(Regex):

    def fa(self):
        return self.sub.fa().plus()

    def dump(self, ind=""):
        print("%sPlus" % ind)
        self.sub.dump(ind + "  ")


class RegexOpt(Regex):

    def fa(self):
        return self.sub.fa().opt()

    def dump(self, ind=""):
        print("%sOpt" % ind)
        self.sub.dump(ind + "  ")


class RegexRange(Regex):

    def fa(self):
        return self.sub.fa().range(self.mn, self.mx)

    def dump(self, ind=""):
        print ("%sRange" % ind)
        self.sub.dump(ind + "  ")


class RegexExpression(object):

    TOKENIZER_EXPRESSION = r'@?\w+(?:\.\w+)?|\d+|\S'

    def __init__(self, string=None, manager=None):
        self.string = string
        self.manager = manager

    def tokenize(self, expr):
        return re.findall(self.TOKENIZER_EXPRESSION, expr)

    def parse(self):
        self.toks = self.tokenize(self.string)
        regex = self.altern()
        if len(self.toks) > 0:
            raise ValueError("Extra tokens starting with '%s'" % self.toks)
        return regex

    # altern -> concat concat*
    def altern(self):
        altern = []
        regex = self.concat()
        if regex:
            altern.append(regex)
        while len(self.toks) and self.toks[0] == '|':
            self.toks.pop(0)
            altern.append(self.concat())
        if len(altern) == 0:
            raise ValueError("Empty altern")
        if len(altern) > 1:
            return RegexAltern(subs=altern, manager=self.manager)
        else:
            return altern[0]

    # concat -> operated operated*
    def concat(self):
        concat = []
        regex = self.operated()
        while regex:
            concat.append(regex)
            regex = self.operated()
        if len(concat) == 0:
            raise ValueError("Empty concat")
        if len(concat) > 1:
            return RegexConcat(subs=concat, manager=self.manager)
        else:
            return concat[0]

    # operated -> atom | atom '?' | atom '*' | atom '+' | atom range
    def operated(self):
        regex = self.atom()
        if regex is None:
            return None
        if len(self.toks) == 0:
            return regex
        if self.toks[0] == '?':
            self.toks.pop(0)
            return RegexOpt(sub=regex, manager=self.manager)
        elif self.toks[0] == '*':
            self.toks.pop(0)
            return RegexStar(sub=regex, manager=self.manager)
        elif self.toks[0] == '+':
            self.toks.pop(0)
            return RegexPlus(sub=regex, manager=self.manager)
        elif self.toks[0] == '{':
            mn,mx = self.range()
            return RegexRange(sub=regex, mn=mn, mx=mx, manager=self.manager)
        else:
            return regex

    def range(self):
        # Pop left brace
        self.toks.pop(0)
        tok = self.toks.pop(0)
        if not re.match('\d+$', tok):
            raise ValueError("Expected number in range, got %s" % tok)
        min = int(tok)
        tok = self.toks.pop(0)
        if tok == '}':
            return min, min
        if tok != ',':
            raise ValueError("Expected comma in range, got %s" % tok)
        tok = self.toks.pop(0)
        if not re.match('\d+$', tok):
            raise ValueError("Expected max number in range, got %s" % tok)
        max = int(tok)
        tok = self.toks.pop(0)
        if tok != '}':
            raise ValueError("Expected terminating brace in range, got %s" % tok)
        return min, max



    # atom -> SYMBOL | '(' altern ')'
    def atom(self):
        if len(self.toks) == 0:
            return None
        tok = self.toks[0]
        if tok == '(':
            self.toks.pop(0)
            regex = self.altern()
            if len(self.toks) == 0 or self.toks[0] != ')':
                raise ValueError("Unbalanced ')'")
            self.toks.pop(0)
            return regex
        elif tok == '|' or tok == ')':
            return None
        elif tok == '*' or tok == '?' or tok == '+':
            raise ValueError("Misplaced operator: %s" % tok)
        else:
            self.toks.pop(0)
            return RegexAtom(symbol=tok, manager=self.manager)


