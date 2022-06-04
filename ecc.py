import collections
import math
import decoder

def genPowerAlpha(alpha,n): # generates powers of alpha
    if n == 0:
        return 1
    else:
        x = alpha ** n
        if x < 929:
            return x
        if x > 929:
            y = x % 929
            return y

def inverse(a): #Helper function to compute the inverse of a number within the field
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
        if res[i] > 929:
            res[i] = res[i] % 929

    return res

def polynomialEval(poly, x):
    output = 0
    for i in range(len(poly)):
        a = poly[i] * (x ** i)
        if a > 929:
            a = a % 929
        output += a
    return output

def derivative(a):
    dPolynomial = [a[i] * i for i in range(1, len(a))]

    for i in range(len(dPolynomial)):
        if dPolynomial[i] > 929:
            dPolynomial[i] = dPolynomial[i] % 929
    return dPolynomial

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

def computeD(Lx, s, ne, i):
    output = 0

    for j in range(ne+1):
        output += (Lx[j] * s[i-j]) % 929
    
    if output > 929:
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

def computeLx(Lx, d, dp, m, Lpx):
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

    mPadded = [(d * inverse(dp)) * k % 929 for k in mPadded]

    for i in range(len(mPadded) - len(Lx)):
        Lx.append(0)
    
    for i in range(len(mPadded)):
        mPadded[i] = 929 - mPadded[i]

    Lx = polynomialAdd(Lx, mPadded)
    
    for i in range(len(Lx)):
        if Lx[i] > 929:
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

def findErrorPolynomial(s):
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

def chienSearch(Lx, ne):
    root_idxs = []
    ELP_roots = []
    index_locations = []

    for i in range(929):
        a = genPowerAlpha(3,i)
        b = polynomialEval(Lx, a)

        if b > 929:
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
                
def computeErrorPolynomials(Lx, s, root_idxs, ne):
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
        if a > 929:
            a = a % 929
        true_message.append(a)
    
    return true_message

    ## TO DO ##

################ TESTER ###############

# # Inputs
# # msg = [76, 819, 612, 450, 793, 720, 570, 414, 26, 824, 827, 810, 823, 816, 823, 834, 810, 807, 567, 27, 417, 207, 476, 597, 357, 27, 236, 870, 847, 808, 364, 390, 518, 326, 820, 626, 848, 808, 458, 308, 538, 508, 458, 308, 536, 813, 17, 255, 3, 26, 50, 19, 799, 132, 13, 320, 792, 133, 98, 396, 13, 780, 320, 804, 13, 206, 12, 38, 356, 304, 544, 375, 19, 13, 86, 398, 191, 805, 483, 94, 317, 317, 308, 168, 355, 672, 130, 244, 741, 241, 345, 862]
# # ecc_level = 3
# # msg = [62, 574, 360, 416, 34, 518, 300, 416, 828, 617, 19, 795, 3, 26, 834, 810, 823, 816, 807, 117, 267, 387, 27, 327, 567, 627, 116, 570, 458, 237, 10, 798, 13, 190, 26, 544, 574, 330, 236, 10, 626, 290, 375, 26, 414, 19, 13, 720, 794, 510, 697, 414, 26, 540, 836, 19, 330, 236, 364, 390, 518, 329, 142, 288, 72, 2, 443, 16, 195, 703, 62, 267, 665, 239, 523, 651, 45, 277]
# # ecc_level = 2
# alpha = 3
# print(f"msg: {msg}, ecc_level: {ecc_level}, alpha: {alpha}")

# # # Compute Syndromes
# s = computeSyndromerome(msg, ecc_level)
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

################## PROCESSING ######################

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

    s = computeSyndromerome(msg, ecc_level)
    ELP, ne = findErrorPolynomial(s)
    root_idxs, index_locations = chienSearch(ELP, ne)
    e_coeffs = computeErrorPolynomials(ELP, s, index_locations, ne)
    true_message = computeTrueMessage(msg, index_locations, e_coeffs)
    decoded_message = decoder.decodeMsg(true_message)

    print(f"Case #{i+1}")
    y = true_message
    y = [str(x) for x in y]
    y = " ".join(y)
    
    print(f"{len(e_coeffs)} {y}")
    print(f"Decoded Message: {decoded_message}")
