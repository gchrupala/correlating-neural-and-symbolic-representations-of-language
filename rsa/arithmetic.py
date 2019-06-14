# Expression -> Op Expression Expression
# Expression -> [0,9]

import random


class Expression:
    
    def __init__(self, value, left=None, right=None):
        self.left = left
        self.right = right
        if left is None and right is None:
            self.value = value
        elif left is not None and right is not None:
            self.op = value
        else:
            raise ValueError("left and right have to be both None or both not None")

    def is_leaf(self):
        return self.left is None and self.right is None

    def prefix(self):
        """Return sequence of expression tree elements in prefix notation."""
        if self.is_leaf():
            yield repr(self.value)
        else:
            yield '('
            yield self.op
            yield from self.left.prefix()
            yield from self.right.prefix()
            yield ')'
            
    def infix(self):
        """Return sequence of expression tree elements in infix notation."""
        if self.is_leaf():
            yield repr(self.value)
        else:
            yield '('
            yield from self.left.infix()
            yield self.op
            yield from self.right.infix()
            yield ')'

    def __repr__(self):
        return ' '.join(list(self.infix()))

    def __eq__(self, other):
        if self.is_leaf() and other.is_leaf():
            return self.value == other.value
        elif not self.is_leaf() and not other.is_leaf():
            return self.op == other.op and self.left == other.left and self.right == other.right
        else:
            return False        

    def content(self):
        if self.is_leaf():
            return self.value
        else:
            return self.op

    def nodes(self):
        """Return the sequence of nodes in the given tree."""
        if self.is_leaf():
            yield self
        else:
            yield from self.left.nodes()
            yield self
            yield from self.right.nodes()

    def depth(self):
        """Return the maximum depth of the expression tree."""
        if self.is_leaf():
            return 0
        else:
            return 1 + max(self.left.depth(), self.right.depth())

    def size(self):
        """Return the number of leaf nodes."""
        if self.is_leaf():
            return 1
        else:
            return self.left.size() + self.right.size()
   
    def evaluate(self, modulo=None):
        """Evluate expression according to arithmetic modulo 10."""
        if self.is_leaf():
            return self.value
        else:
            
            if self.op == '+':
                f = lambda a, b: a+b
            elif self.op == '-':
                f = lambda a, b: a-b
            else:
                raise ValueError("Unknown operator {}".format(self.op))
            if modulo is not None:
                return f(self.left.evaluate(), self.right.evaluate()) % modulo
            else:
                return f(self.left.evaluate(), self.right.evaluate())
                
def generate(p=1.0, decay=2):
    """Generate a random expression."""
    if random.random() <= p:
        # generate branch
        op = random.sample(['+','-'], 1)[0]
        left = generate(p=p/decay, decay=decay)
        right = generate(p=p/decay, decay=decay)
        return Expression(op, left=left, right=right)
    else:
        return Expression(random.sample(range(10), 1)[0])


def vocabify(seq, length=16, pad_end=False):
    """Convert sequence of symbols to ints"""
    seq = list(seq)
    M = {' ': 10, '(':11, ')':12, '+':13, '-':14}
    L = len(seq)
    P = list(' ' * (length-L))
    if pad_end:
        S = seq + P
    else:
        S = P + seq
    for x in S:
        if x in [str(i) for i in range(10)] :
            yield int(x)
        else:
            yield M[x]

    
def generate_batch(p=1.0, decay=2.0, size=32):
    """Generate a batch of expressions and vocabify them."""
    es = [ generate(p=p, decay=decay) for _ in range(size) ]
    L = max([ len(list(e.infix())) for e in es ])
    return L, es   


