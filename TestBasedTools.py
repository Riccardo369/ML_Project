from BasedTools import *

Derivata1 = DerivationLambda(lambda yo, yt: 1/2 * (yo - yt)**2, 0)
Derivata2 = DerivationLambda(lambda yo, yt: 1/2 * (yo - yt)**2, 1)

assert Derivata1(3, 2) == 1
assert Derivata2(3, 2) == -1

################################################################################################

try: CheckParametersFunction(lambda x: x, 2)
except: pass

CheckParametersFunction(lambda x, e: x, 2)