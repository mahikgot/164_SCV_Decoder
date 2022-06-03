import collections
import math
from re import L

def genPowerAlpha(alpha,n): # generates powers of alpha
    if n == 0:
        return 1
    else:
        x = alpha ** n
        if x <= 929:
            return x
        if x > 929:
            y = x % 929
            return y

def polynomialMsg(msg,alpha,s): #does polynomial multiplication
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

def computeSyndromerome(m,eccL): #Computes syndrome and then returns s
    ecc_level = eccL
    ecc_cword_count = int(math.pow(2,(ecc_level+1)))
    s = []
    for i in range(ecc_cword_count):
        x = polynomialMsg(m,alpha,i+1)
        x = sum(x) % 929
        #print(f "x: {x}")
        s.append(x)
    return s

def inverse(a):
    n = 929
    for x in range(n):
        if (a * x) % n == 1:
            return x

def polynomialAdd(A,B):
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

def polynomialMult(a,b):
    s1 = a
    s2 = b
    res = [0]*(len(s1)+len(s2)-1)
    for o1,i1 in enumerate(s1):
        for o2,i2 in enumerate(s2):
            res[o1+o2] += i1*i2
    
    for i in range(len(res)):
        if res[i] >= 929:
            res[i] = res[i] % 929

    return res

def polynomialEval(poly, x):
    output = 0
    for i in range(len(poly)):
        a = poly[i] * (x ** i)
        if a >= 929:
            a = a % 929
        output += a
    return output

def derivative(a):
    dPolynomial = [a[i] * i for i in range(1, len(a))]

    for i in range(len(dPolynomial)):
        if dPolynomial[i] >= 929:
            dPolynomial[i] = dPolynomial[i] % 929
    return dPolynomial


# def computerLe():
    

# def computeT(ne):
#     T = [1]
#     for i in range(ne):
#         x = 3**i
#         T.append(x)
#     return T

def chienSearch(Lx, ne):
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
    print(f"ELP roots: {ELP_roots}")
    return root_idxs[:ne], index_locations[:ne]
                
def computeErrorPolynomials(Lx, s, root_idxs, ne):
    ELP = Lx
    syndrome = s
    rootIndx = root_idxs
    ne = ne
    O = polynomialMult(syndrome, ELP)
    O = O[:ne]
    print(f"O: {O}")
    DLx = derivative(Lx)
    print(f"DLx: {DLx}")
    e_coeffs = []
    for i in range(len(rootIndx)):
        a = inverse(genPowerAlpha(3, rootIndx[i]))
        print(f"Root Indices: {rootIndx}")
        print(f"a: {a}")
        oPoly = polynomialEval(O, a)
        dPoly = polynomialEval(DLx, a)
        e = ((-1 * oPoly) * (inverse(dPoly))) % 929
        e_coeffs.append(e)
    # print(f"ELP: {ELP}")
    # print(f"Syndrome: {syndrome}")
    print(f"O: {O}")
    # print(f"Error Polynomial: {e_coeffs}")
    return e_coeffs
    
def computeTrueMessage(m, index_location, e_coeffs):
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
    

def computeLx(Lx, d, dp, m, Lp):
    Lx = Lx
    dp = dp
    Lp = Lp
    d = d
    m = m
    mPadded = []
    q = -1 * (d * inverse(dp)) % 929 # d/dp

    for i in range(m):
        mPadded.append(0)
    mPadded.append(q)

    #print(mPadded)

    a = polynomialMult(Lp, mPadded)
    b = polynomialAdd(Lx, (-1*a))

    return b
    

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

def computeD(Lx, s, ne, i):
    output = s[0]
    for k in range(ne):
        if i-k > 0:
            output += Lx[k] * s[i-k]
    if output > 929:
        output = output % 929
    else:
        return output

def findErrorPolynomial(s):
    s = s
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
    
    return Lx, ne

    ## TO DO ##

    
################ TESTER ###############

# Inputs
msg = [10, 813, 864, 477, 749, 739, 196, 844, 393, 900, 822, 22, 716, 545, 596, 130, 458, 0]
ecc_level = 2
# msg = [4, 817, 209, 900, 465, 632]
# ecc_level = 0
alpha = 3
print(f"msg: {msg}, ecc_level: {ecc_level}, alpha: {alpha}")

# Compute Syndromes
s = computeSyndromerome(msg, ecc_level)
print(f"Syndromes: {s}")

# Find ELP (not yet implemented)
# ELP = [1, 902] 
# ne = 1

ELP = [1, 166, 738, 31, 922, 0, 0, 0] 
ne = 4
print(f"ELP: {ELP}")

# ELP, ne = findErrorPolynomial(s)
# print(f"ELP: {ELP}")

# Find ELP roots
root_idxs, index_locations = chienSearch(ELP, ne)
print(f"root raw value: {root_idxs} and index location (exponent): {index_locations}")

# Compute Error Polynomials, since ne takes over from ELP which is not yet implemented we just declare it
e_coeffs = computeErrorPolynomials(ELP, s, index_locations, ne)
print(f"Error Polynomials: {e_coeffs}")

# Compute True Message
true_message = computeTrueMessage(msg, index_locations, e_coeffs)
print(f"True Message: {true_message}")

# print(computeD([1], [238, 852], 0, 0))
# print(computeLx([1, 902], 877, 238, 2, [1]))
