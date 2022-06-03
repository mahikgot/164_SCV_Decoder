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

def compSyndrome(m,eccL): #Computes syndrome and then returns s
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

def polynomialAdd(A,B,m,n):
    size = max(m,n)
    sum = [0 for i in range(size)]
    
        # Initialize the product polynomial
        
    for i in range(0, m, 1):
        sum[i] = A[i]
    
        # Take ever term of first polynomial
    for i in range(n):
        sum[i] += B[i]
    
    return sum

def derivative(a):
    dPolynomial = [a[i] * i for i in range(1, len(a))]
    return dPolynomial

# def computerLe():
    

# def computeT(ne):
#     T = [1]
#     for i in range(ne):
#         x = 3**i
#         T.append(x)
#     return T

def chienSearch(Lx):
    root_idxs = []
    ELP_roots = []
    index_locations = []

    for i in range(929):
        x = []
        a = genPowerAlpha(3,i)
        #print(f"a: {a}")

        for j in range(len(Lx)): #[1 902]
            b = Lx[j] * a**j
            #print(f"b: {b}")
            x.append(b)
            #print(f"x: {x}")
            
            if sum(x) % 929 == 0:
                ELP_roots.append(a)
                y = ((i * -1) % 929) - 1
                z = genPowerAlpha(3,y)
                root_idxs.append(z)
                index_locations.append(y)

    return root_idxs, index_locations
                
#def computeErrorPolynomials(Lx, s, root_idxs):
    
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
    

# def computeLx(Lx, d, dp, m, Lp):
#     Lx = Lx
#     dp = dp
#     Lp = Lp
#     d = d
#     m = m

#     q = -1 * d * inverse(dp) % 929 # d/dp
#     print(Lx, [0, q])
#     c = polynomialAdd(Lx, [q], 1, 2)
#     print(c)


def computeLx(Lx, d, dp, m, Lpx):
    secondTerm = 0
    quotient = (d * -1 * dp) % 929 * -1
    zeroes = [0 for _ in range(m)]
    temp = [element * quotient for element in Lpx]
    temp = zeroes + temp
    if len(temp) > len(Lx):
        padding = [0 for _ in range(len(temp)-len(Lx))]
        Lx = Lx + padding

    Lx = [LxElem - LpxElem for LxElem, LpxElem in zip(Lx, temp)]

    return Lx

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
        print(f"d: {d}")
        if d == 0:
            m += 1
        else:
            oldLx = Lx
            Lx = computeLx(Lx, d, dp, m, Lp)
            print(f"L(x): {Lx}")
            if 2*ne <= i:
                Lp = oldLx
                ne = i + 1 - ne
                dp = d
                m = 1
            else:
                m += 1
    
    return Lx

    ## TO DO ##

    
    

################ TESTER ###############

msg = [4, 817, 209, 900, 465, 632]
ecc_level = 0
alpha = 3

#print((genPowerAlpha(3,4) * 817 % 929))
#print(polynomialMsg(msg, 3, 1))
#print(compSyndrome(msg, ecc_level))
#print(findErrorPolynomial(compSyndrome(msg, ecc_level)))
#print(computeD([1],[238, 852],0,0))
#print(computeLx([1], 238, 1, 1, [1]))
#print(computeLx([1, 691], 238, 691, 1, [1])) #Lx d dp m Lpx
#print(chienSearch([1, 902]))
#print(computeTrueMessage(msg, [3,2,5], [869, 817, 209]))
#print(multInverse(238))
#print(derivative([1,2,3,4,5]))