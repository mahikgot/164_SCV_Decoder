import math

def genPowerAlpha(alpha,n): # generates powers of alpha
    if n == 0:
        return 1
    else:
        x = math.pow(alpha,n) 
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

def computeLx(Lx, ne):
    Lx = [1]
    None

    ## TO DO ##


def computeD(Lx, s, ne):
    x = [s[0]]
    for i in range(ne):
        y = Lx[i] * s[i]
    
    if sum(x) > 929:
        x = sum(x) % 929
    else:
        x = sum(x)

    return sum (x) 

    ## TO DO ##

def findErrorPolynomial(s):
    s = s
    Lx = [1]
    ne = 0
    dp = 1
    Lp = [1]
    m = 1
    for i in range(len(s)):
        d = computeD(Lx, s, ne)
        if d == 0:
            m += 1
        else:
            oldLx = Lx
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
print(compSyndrome(msg, ecc_level))
#print(findErrorPolynomial(compSyndrome(msg, ecc_level)))