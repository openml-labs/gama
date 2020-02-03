import re
import random
from .emission import Emission

class TokenTest(object):

    def __init__(self, **kw):
        for k,v in kw.items():
            setattr(self, k, v)
        if not hasattr(self, 'primers'):
            self.primers = [ 'SKlearn' ]
        self._primed = False

    def generate(self):
        if not self._primed:
            for p in self.primers:
                self.parent.library.prime(p)
            self._primed = True
        primitives = self.primitives()
        if len(primitives) == 0:
            raise ValueError("No matching primitives")
        return random.choice(list(primitives))


class LiteralTokenTest(TokenTest):

    def primitives(self):
        prims = set(self.parent.library.primitives_with_name(self.name))
        params = None
        if hasattr(self, 'parameters'):
            params = self.parameters
        return set(Emission(primitive=p, parameters=params) for p in prims)

    def matches(self, primitive):
        if self.name not in self.parent.library.primitive_path(primitive):
            return False
        if hasattr(self, 'parameters'):
            if not self.parent.library.primitive_has_parameters(primitive, self.parameters):
                return False
        return True

    def dump(self, indent=0):
        print("%sLIT %s" % ((' ' * indent), self.name))


class SpecTokenTest(TokenTest):

    def primitives(self):
        prims = self.parent.library.primitives_of_type(self.name, self.items)
        params = None
        if hasattr(self, 'parameters'):
            params = self.parameters
        return set([Emission(primitive=p, parameters=params) for p in prims])

    def matches(self, primitive):
        if not self.parent.library.primitive_has_type(primitive, self.name, self.items):
            return False
        if hasattr(self, 'parameters'):
            if not self.parent.library.primitive_has_parameters(primitive, self.parameters):
                return False
        return True

    def dump(self, indent=0):
        print("%SPEC %s" % ((' ' * indent), self.name))


class ReferenceTokenTest(TokenTest):

    def primitives(self):
        ref = self.parent.get_test(self.name)
        return ref.primitives()

    def matches(self, primitive):
        ref = self.parent.get_test(self.name)
        return ref.matches(primitive)

    def dump(self, indent=0):
        print("%sREF %s" % ((' ' * indent), self.name))


class AndTest(TokenTest):

    def primitives(self):
        result = None
        for sub in self.subs:
            prims = sub.primitives()
            if result is None:
                result = set(prims)
            else:
                result &= set(prims)
        return result

    def matches(self, primitive):
        for sub in self.subs:
            if not sub.matches(primitive):
                return False
        return True

    def testlabel(self):
        return "AND"

    def dump(self, indent=0):
        print("%sAND" % (' ' * indent))
        for s in self.subs:
            s.dump(indent+3)

class OrTest(TokenTest):

    def primitives(self):
        result = None
        for sub in self.subs:
            prims = sub.primitives()
            if result is None:
                result = set(prims)
            else:
                result |= set(prims)
        return result

    def matches(self, primitive):
        for sub in self.subs:
            if sub.matches(primitive):
                return True
        return False

    def testlabel(self):
        return "OR"

    def dump(self, indent=0):
        print("%sOR" % (' ' * indent))
        for s in self.subs:
            s.dump(indent+3)

class NotTest(TokenTest):

    def primitives(self):
        diffprims = self.sub.primitives()
        return set(self.parent.library.all_primitives()) - diffprims

    def matches(self, primitive):
        return not self.sub.matches(primitive)

    def testlabel(self):
        return "NOT"

    def dump(self, indent=0):
        print("%sNOT" % (' ' * indent))
        self.arg.dump(indent + 3)


class TokenTestExpression(object):

    TOKENIZER_EXPRESSION = r'{|}|\(|\)|=|\[|\]|,|\w+'

    def __init__(self, manager=None):
        self.manager = manager

    def tokenize(self, expr):
        return re.findall(self.TOKENIZER_EXPRESSION, expr, re.DOTALL)

    def parse(self, expr):
        self.string = expr
        self.toks = self.tokenize(self.string)
        test = self.orexpr()
        if len(self.toks) > 0:
            raise ValueError("Extra tokens starting with '%s'" % self.toks)
        return test

    # orexpr -> andexpr | andexpr 'or' andexpr
    def orexpr(self):
        orexpr = []
        test = self.andexpr()
        if test:
            orexpr.append(test)
        while len(self.toks) and self.toks[0] == 'or':
            self.toks.pop(0)
            orexpr.append(self.andexpr())
        if len(orexpr) == 0:
            raise ValueError("Empty 'or' expression")
        if len(orexpr) > 1:
            return OrTest(subs=orexpr)
        else:
            return orexpr[0]

    # andexpr -> notexpr | notexpr 'and' notexpr
    def andexpr(self):
        andexpr = []
        test = self.notexpr()
        if test:
            andexpr.append(test)
        while len(self.toks) and self.toks[0] == 'and':
            self.toks.pop(0)
            andexpr.append(self.notexpr())
        if len(andexpr) == 0:
            raise ValueError("Empty 'and' expression")
        if len(andexpr) > 1:
            return AndTest(subs=andexpr)
        else:
            return andexpr[0]

    # notexpr -> atom | 'not' atom
    def notexpr(self):
        if len(self.toks) == 0:
            return None
        notted = False
        if self.toks[0] == 'not':
            notted = True
            self.toks.pop(0)
        test = self.atom()
        if test is None:
            raise ValueError("Missing argument starting at " + "'%s'" % self.toks[0])
        if notted:
            return NotTest(arg=test)
        else:
            return test

    # atom -> SPEC | REF | LITERAL | '(' orexpr ')'
    def atom(self):

        if len(self.toks) == 0:
            return None
        tok = self.toks.pop(0)

        # Nested expression
        if tok == '(':
            test = self.orexpr()
            if len(self.toks) == 0 or self.toks[0] != ')':
                raise ValueError("Unbalanced '('")
            self.toks.pop(0)
            return test

        # Reference
        m = re.match('&(\w+|\.)+$', tok)
        if m:
            name = m.group(1)
            return ReferenceTokenTest(name=name, parent=self.manager)

        if not re.match('\w+$', tok):
            raise ValueError("Expected identifier, got '%s'" % tok)

        name = tok

        if len(self.toks) == 0:
            return LiteralTokenTest(name=name, parent=self.manager)

        # Spec
        if self.toks[0] == '{':
            test = self.spec(tok)
            return test

        # Parameterization
        if self.toks[0] == '(':
            test = self.params(name)
            return test

        return LiteralTokenTest(name=name, parent=self.manager)
    
    def spec(self, name):
        # Pop left '{'
        self.toks.pop(0)
        items = []
        while self.toks[0] != '}':
            item = self.toks.pop(0)
            if not re.match('\w+$', item):
                raise ValueError("Confusing spec item: '%s'" % item)
            items.append(item)
        # Pop right '}'
        self.toks.pop(0)
        return SpecTokenTest(name=name, items=items, parent=self.manager)

    def params(self, name):
        # Pop left '('
        self.toks.pop(0)
        params = {}
        for key, value in self.param():
            params[key] = value
        # Pop right ')'
        self.toks.pop(0)
        return LiteralTokenTest(name=name, parameters=params)

    def param(self):
        while self.toks[0] != ')':
            key = self.toks.pop(0)
            if not re.match('\w+$', key):
                raise ValueError("Expected param key, got '%s'" % key)
            assign = self.toks.pop(0)
            if assign != '=':
                raise ValueError("Malformed parameter")
            value = self.param_value()
            yield key, value

    def param_value(self):
        # List
        if self.toks[0] == '[':
            self.toks.pop(0)
            items = []
            for item in self.list_item():
                items.append(item)
            if self.toks[0] != ']':
                raise ValueError("Malformed value list")
            self.toks.pop(0)
            return items
        else:
            tok = self.toks.pop(0)
            return tok

    def list_item(self):
        while self.toks[0] != ']':
            tok = self.toks.pop(0)
            if self.toks[0] == ',':
                self.toks.pop(0)
            yield tok

