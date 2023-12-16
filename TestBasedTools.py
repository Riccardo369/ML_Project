from BasedTools import *

Derivata1 = DerivationLambda(lambda yo, yt: 1/2 * (yo - yt)**2, 0)
Derivata2 = DerivationLambda(lambda yo, yt: 1/2 * (yo - yt)**2, 1)

assert Derivata1(3, 2) == 1
assert Derivata2(3, 2) == -1

################################################################################################

try: CheckParametersFunction(lambda x: x, 2)
except: pass

CheckParametersFunction(lambda x, e: x, 2)

################################################################################################

H1 = HyperParameter(2)

Lambda = lambda x: x*H1()

assert Lambda(1) == 2
assert Lambda(4) == 8

H1.Value = 10

assert Lambda(10) == 100
assert Lambda(2) == 20

e = 2

Lambda1 = lambda x: x*e

assert Lambda1(1) == 2
assert Lambda1(2) == 4

e = 10

assert Lambda1(1) == 10
assert Lambda1(7) == 70