from __future__ import annotations
from abc import ABC, abstractmethod
import math

#########################################################################################################
# Error corrector program made by:
# Steven Esguerra
# 2019 - 05959
# CoE 164 - FWX
#
#
# Decoder program made by:
# Mark Guiang
# 2018 - 07118
# CoE 164 - FWX
#########################################################################################################

########################################################################################################
#Start of Decoder Program
########################################################################################################
class Context:
    #State Machine Global Context
    def __init__(self, SCV) -> None:
        self._state = None
        self.SCV = SCV
        self.setState(Alpha())

    def setState(self, state: State):
        self._state = state
        self._state.context = self

    def get_HL(self, x):
        if x <= 29:
            return 0, x
        H = x//30
        if H >= 30:
            H = 29
        L = x - H*30
        return H, L

    def get_data(self, scv_list):
        output = []
        for i in range(scv_list[0] - 1):
            if scv_list[i+1] == 900:
                continue
            output.extend(self.get_HL(scv_list[i+1]))

        if output[-1] == 29:
            output.pop()
        return output

    def decode(self):
        decoded = ''
        converted = self.get_data(self.SCV)
        for element in converted:
            #state.decode method handles state switching and latching
            decoded = self._state.decode(decoded, element)
        return decoded

class State(ABC):
    #Abstract Class named State for the different Submodes
    #Each State have a reference to the Context Class, being able to switch its state
    #Each have their own mapping dictionary
    #Decode methods are different for each submode, this method also handles switching and latching between states
    def __init__(self):
        self.mapping = self.prepare_mapping()
        self.switched = False
        self.lastclass = None

    @property
    def lastclass(self) -> str:
     return self._lastclass
    @lastclass.setter
    def lastclass(self, last: str) -> None:
        self._lastclass = last

    @property
    def switched(self) -> bool:
        return self._switched
    @switched.setter
    def switched(self, from_switch: bool) -> None:
        self._switched = from_switch

    @property
    def context(self) -> Context:
        return self._context
    @context.setter
    def context(self, context: Context) -> None:
        self._context = context

    @property
    def mapping(self) -> dict:
        return self._mapping
    @mapping.setter
    def mapping(self, map_dict: dict) -> None:
        self._mapping = map_dict

    @abstractmethod
    def prepare_mapping(self) -> dict:
        pass

    @abstractmethod
    def decode(self, output: str, num: int) -> None:
        pass

    def switch(self, state: State) -> None:
        state.switched = True
        state.lastclass = self.__class__.__name__
        self.context.setState(state)
    def switch_back(self) -> None:
        self.context.setState(globals()[self.lastclass]())

    def latch(self, state: State) -> None:
        self.context.setState(state)

class Alpha(State):
    def prepare_mapping(self):
        mapping = dict(zip(range(26), range(65,91)))
        mapping[26] = 32
        return mapping

    def decode(self, output, num):
        if num == 27:
            self.latch(Lower())
        elif num == 28:
            self.latch(Mixed())
        elif num == 29:
            self.switch(Punctuation())
        else:
            decoded = self.mapping.get(num)
            output += chr(decoded)

        if self.switched:
            self.switch_back()
        return output


class Punctuation(State):
    def prepare_mapping(self):
        ascii_list = [
                59, 60, 62, 64, 91, 92, 93, 95, 96, 126,
                33, 13, 9, 44, 58, 10, 45, 46, 36, 47, 34,
                124, 42, 40, 41, 63, 123, 125, 39
                ]
        mapping = dict(zip(range(29), ascii_list))
        return mapping

    def decode(self, output, num):
        if num == 29:
            self.Latch(Alpha())
        else:
            decoded = self.mapping.get(num)
            output += chr(decoded)

        if self.switched:
            self.switch_back()
        return output

class Lower(State):
    def prepare_mapping(self):
        mapping = dict(zip(range(26), range(97, 123)))
        mapping[26] = 32
        return mapping

    def decode(self, output, num):
        if num == 27:
            self.switch(Alpha())
        elif num == 28:
            self.latch(Mixed())
        elif num == 29:
            self.switch(Punctuation())
        else:
            decoded = self.mapping.get(num)
            output += chr(decoded)

        if self.switched:
            self.switch_back()
        return output

class Mixed(State):
    def prepare_mapping(self):
        mapping = dict(zip(range(10), range(48, 58)))
        ascii_list = [
                38, 13, 9, 44, 58, 35, 45, 46, 36, 47, 43,
                37, 42, 61, 94
                ]
        mapping.update(dict(zip(range(10, 25), ascii_list)))
        mapping[26] = 32
        return mapping

    def decode(self, output, num):
        if num == 27:
            self.latch(Lower())
        elif num == 28:
            self.latch(Alpha())
        elif num == 29:
            self.switch(Punctuation())
        else:
            decoded = self.mapping.get(num)
            output += chr(decoded)

        if self.switched:
            self.switch_back()
        return output

def decodeMsg(scv_list):
    context = Context(scv_list)
    return context.decode()

########################################################################################################
#Start of Error Correction Program
########################################################################################################
def genPowerAlpha(alpha, n): # generates powers of alpha
    if n == 0:
        return 1
    else:
        x = alpha ** n
        if x < 929:
            return x
        if x >= 929:
            y = x % 929
            return y

# def inverse(a): #Helper function to compute the inverse of a number within the field
    # n = 929
    # for x in range(n):
    #     if (a * x) % n == 1:
    #         return x
    # p = 929
    # t = 0
    # nextT = 1

    # while a > 0:
    #     q = p // a
    #     r = p % a
    #     t = t - q * nextT
    #     nextT = t
    #     p = a
    #     a = r
    #     nextT = t
    # t = t % p

    # return t

def egcd(a, b): #Helper function to help inverse()
    if a == 0:
        return (b, 0, 1)
    else:
        g, y, x = egcd(b % a, a)
        return (g, x - (b // a) * y, y)

def inverse(a, m = 929): #Helper function to compute the inverse of a number within the field
    g, x, y = egcd(a, m)
    if g != 1:
        raise Exception('modular inverse does not exist')
    else:
        return x % m

def polynomialAdd(A,B): #Helper function to add polynomials
    m = len(A)
    n = len(B)
    size = max(m,n)
    sum = [0 for i in range(size)]

        # Initialize the product polynomial

    for i in range(0, m, 1):
        sum[i] = A[i]

        # Take ever term of first polynomial
    for i in range(n):
        sum[i] += B[i]

    return sum

def polynomialMult(a,b): #Helper function to multiply polynomials
    s1 = a
    s2 = b
    res = [0]*(len(s1)+len(s2)-1)
    for o1,i1 in enumerate(s1):
        for o2,i2 in enumerate(s2):
            res[o1+o2] += i1*i2 % 929
    return res

def polynomialEval(poly, x): #Evaluates a polynomial at a given x
    output = 0
    for i in range(len(poly)):
        a = poly[i] * (x ** i)
        if a >= 929:
            a = a % 929
        output += a
    return output

def derivative(a): #Helper function to compute the derivative of a polynomial
    dPolynomial = [a[i] * i for i in range(1, len(a))]

    for i in range(len(dPolynomial)):
        if dPolynomial[i] >= 929:
            dPolynomial[i] = dPolynomial[i] % 929
    return dPolynomial

def polynomialMsg(msg,alpha,s): #Turns message into a polynomial
    msg_poly = []
    degree = len(msg)-1
    for i in range(len(msg)):
        x = (msg[i]*genPowerAlpha(alpha, degree*s))
        #print(f"x before modulo: {x}")
        x = x % 929
        #print(f"x after modulo: {x}")
        msg_poly.append(x)
        degree -= 1
    return msg_poly

def computeSyndrome(m,eccL): #Computes syndrome and then returns the syndromes
    ecc_level = eccL
    ecc_cword_count = int(math.pow(2,(ecc_level+1)))
    s = []
    for i in range(ecc_cword_count):
        x = polynomialMsg(m,alpha,i+1)
        x = sum(x) % 929
        #print(f "x: {x}")
        s.append(x)
    return s

def computeD(Lx, s, ne, i): #Helper function for computing ELP, computes discrepancy
    output = 0

    for j in range(ne+1):
        output += (Lx[j] * s[i-j]) % 929

    if output >= 929:
        output = output % 929

    return output

# def computeD(Lx, s, ne, i):
#     output = 0
#     for k in range(ne+1):
#         if i-k > 0:
#             output += Lx[k] * s[i-k]
#     if output > 929:
#         output = output % 929
#     else:
#         return output

def computeLx(Lx, d, dp, m, Lpx): #Helper function for computing ELP, computes lambda
    Lx = Lx
    dp = dp
    Lpx = Lpx
    d = d
    m = m
    mPadded = []

    for i in range(m):
        mPadded.append(0)
    mPadded.extend(Lpx)

    #print(f"mPadded: {mPadded}")

    for i in range(len(mPadded)):
        a = d * inverse(dp) * mPadded[i]
        if a >= 929:
            a = a % 929
        mPadded[i] = a

    for i in range(len(mPadded) - len(Lx)):
        Lx.append(0)

    for i in range(len(mPadded)):
        mPadded[i] = 929 - mPadded[i]

    Lx = polynomialAdd(Lx, mPadded)

    for i in range(len(Lx)):
        if Lx[i] >= 929:
            Lx[i] = Lx[i] % 929

    return Lx

# def computeLx(Lx, d, dp, m, Lp):
#     Lx = Lx
#     dp = dp
#     Lp = Lp
#     d = d
#     m = m
#     mPadded = []
#     q = d * inverse(dp) % 929
#     print(f"q: {q}")  # d/dp

#     for i in range(m):
#         mPadded.append(0)
#     mPadded.append(q)

#     #print(mPadded)

#     a = polynomialMult(Lp, mPadded)
#     print(f"a: {a}")

#     while len(Lx) != len(a):
#         Lx.append(0)

#     for i in range(len(Lx)):
#         Lx[i] = Lx[i] + (929 - a[i])

#     return Lx


# def computeLx(Lx, d, dp, m, Lpx):
#     secondTerm = 0
#     quotient = (d * -1 * dp) % 929 * -1
#     zeroes = [0 for _ in range(m)]
#     temp = [element * quotient for element in Lpx]
#     temp = zeroes + temp
#     if len(temp) > len(Lx):
#         padding = [0 for _ in range(len(temp)-len(Lx))]
#         Lx = Lx + padding
#     Lx = [LxElem - LpxElem for LxElem, LpxElem in zip(Lx, temp)]
#     return Lx

def findErrorPolynomial(s): #Helper function to find ELP
    Lx = [1]
    ne = 0
    dp = 1
    Lp = [1]
    m = 1
    for i in range(len(s)):
        d = computeD(Lx, s, ne, i)
        #print(f"d: {d}")
        if d == 0:
            m += 1
        else:
            oldLx = Lx
            Lx = computeLx(Lx, d, dp, m, Lp)
            #print(f"L(x): {Lx}")
            if 2*ne <= i:
                Lp = oldLx
                ne = i + 1 - ne
                dp = d
                m = 1
            else:
                m += 1

    for i in range(len(Lx)):
        if Lx[i] >= 929:
            Lx[i] = Lx[i] % 929

    return Lx, ne

def chienSearch(Lx, ne): #Finds the roots of the ELP
    root_idxs = []
    ELP_roots = []
    index_locations = []

    for i in range(929):
        a = genPowerAlpha(3,i)
        b = polynomialEval(Lx, a)

        if b >= 929:
            b = b % 929

        if b == 0:
            #print(f"Root found at index {i}")
            c = genPowerAlpha(3,i)
            ELP_roots.append(i)
            d = inverse(c)
            #print(f"Root: {c}, Inverse: {d}")
            root_idxs.append(d)
            for i in range(929):
                if d == genPowerAlpha(3,i):
                    index_locations.append(i)
                    break

    ELP_roots = ELP_roots[:ne]
    # print(f"ELP roots: {ELP_roots}")
    return root_idxs[:ne], index_locations[:ne]

def computeErrorPolynomials(Lx, s, root_idxs, ne): #Computes the error polynomial
    ELP = Lx
    syndrome = s
    rootIndx = root_idxs
    ne = ne
    O = polynomialMult(syndrome, ELP)
    O = O[:ne]
    #print(f"O: {O}")
    DLx = derivative(Lx)
    #print(f"DLx: {DLx}")
    e_coeffs = []
    #print(f" Root Index: {rootIndx}")
    for i in range(len(rootIndx)):
        a = inverse(genPowerAlpha(3, rootIndx[i]))
        #print(f"Root Indices: {rootIndx}")
        #print(f"a: {a}")
        oPoly = polynomialEval(O, a)
        dPoly = polynomialEval(DLx, a)
        e = ((-1 * oPoly) * (inverse(dPoly))) % 929
        e_coeffs.append(e)
    # print(f"ELP: {ELP}")
    # print(f"Syndrome: {syndrome}")
    # print(f"O: {O}")
    # print(f"Error Polynomial: {e_coeffs}")
    # print(f"DLx: {DLx}")
    return e_coeffs

def computeTrueMessage(m, index_location, e_coeffs): #Computes the true message
    if len(index_location) == 0:
        return m
    if len(index_location) != len(e_coeffs):
        return ("Error: index_location and e_coeffs must be the same length")

    true_message = []
    index_location = index_location
    e_coeffs = e_coeffs
    e_x = [0 for _ in range(len(m))]
    msg = m

    for i in range(len(e_coeffs)):
        e_x[index_location[i]] = e_coeffs[i]
    e_x.reverse()

    #print(f"e_x: {e_x}")

    for i in range(len(msg)):
        a = msg[i] - e_x[i]
        if a < 0:
            a = a % 929
        if a >= 929:
            a = a % 929
        true_message.append(a)

    return true_message

def stringify(a): #By some chances, escape characters are appearing in the OJ, to fix this, this function is used
    a = a.replace("\n", r"\n")
    a = a.replace("\t", r"\t")
    a = a.replace("\b", r"\b")
    a = a.replace("\f", r"\f")
    a = a.replace("\r", r"\r")
    a = a.replace("\v", r"\v")

    return a

################ TESTER ############### #This part is for manually testing inputs, debugging, etc

# Inputs (typed in manually)
# ecc_level = input()
# ecc_level = ecc_level.split()
# ecc_level = [int(x) for x in ecc_level]
# ecc_level = ecc_level[0]
# msg = input()
# msg = msg.split()
# msg = [int(x) for x in msg]

# # Inputs (as list)
# #msg = [7, 87, 447, 146, 841, 184, 905, 879, 523]
# #ecc_level = 3

# alpha = 3
# print(f"msg: {msg}, ecc_level: {ecc_level}, alpha: {alpha}")

# # Compute Syndromes
# s = computeSyndrome(msg, ecc_level)
# print(f"Syndromes: {s}")

# ELP, ne = findErrorPolynomial(s)
# print(f"ELP: {ELP}")

# # Find ELP roots
# root_idxs, index_locations = chienSearch(ELP, ne)
# print(f"root raw value: {root_idxs} and index location (exponent): {index_locations}")

# # Compute Error Polynomials, since ne takes over from ELP which is not yet implemented we just declare it
# e_coeffs = computeErrorPolynomials(ELP, s, index_locations, ne)
# print(f"Error Polynomials: {e_coeffs}")

# # Compute True Message
# true_message = computeTrueMessage(msg, index_locations, e_coeffs)
# print(f"True Message: {true_message}")

# decoded_message = decoder.decodeMsg(true_message)
# print(f"Decoded Message: {decoded_message}")

################## PROCESSING ###################### #This is for processing input data

entries = int(input())
alpha = 3

for i in range(entries):
    ecc_level = input()
    #print(ecc_level)
    ecc_level = ecc_level.split()
    #print(ecc_level)
    ecc_level = [int(x) for x in ecc_level]
    #print(ecc_level)
    ecc_level = ecc_level[0]
    msg = input()
    #print(msg)
    msg = msg.split()
    #print(msg)
    msg = [int(x) for x in msg]
    #print(msg)

    s = computeSyndrome(msg, ecc_level)
    ELP, ne = findErrorPolynomial(s)
    root_idxs, index_locations = chienSearch(ELP, ne)
    e_coeffs = computeErrorPolynomials(ELP, s, index_locations, ne)
    true_message = computeTrueMessage(msg, index_locations, e_coeffs)
    decoded_message = decodeMsg(true_message)

    #print(f"Done with {i+1}")

    print(f"Case #{i+1}:")
    y = true_message
    y = [str(x) for x in y]
    y = " ".join(y)

    decoded_message = stringify(decoded_message)

    print(f"{len(e_coeffs)} {y}")
    print(f"{decoded_message}")
