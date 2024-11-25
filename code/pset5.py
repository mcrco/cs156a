from math import exp

def p5p6():
    E = lambda u, v: (u * exp(v) - 2 * v * exp(-u)) ** 2
    dEdu = lambda u, v : 2 * (u * exp(v) - 2 * v * exp(-u)) * (exp(v) + 2 * v * exp(-u)) 
    dEdv = lambda u, v : 2 * (u * exp(v) - 2 * v * exp(-u)) * (u * exp(v) - 2 * exp(-u)) 
    lr = 0.1
    u, v = 1, 1 
    iters = 0

    while E(u, v) >= 1e-14:
        u_, v_ = u, v
        u -= lr * dEdu(u_, v_)
        v -= lr * dEdv(u_, v_)
        iters += 1

    print(f"Number of iterations before error falls below 1e-14: {iters}")
    print(f"Final values for (u, v): {(u, v)}")

def p7():
    E = lambda u, v: (u * exp(v) - 2 * v * exp(-u)) ** 2
    dEdu = lambda u, v : 2 * (u * exp(v) - 2 * v * exp(-u)) * (exp(v) + 2 * v * exp(-u)) 
    dEdv = lambda u, v : 2 * (u * exp(v) - 2 * v * exp(-u)) * (u * exp(v) - 2 * exp(-u)) 
    lr = 0.1
    u, v = 1, 1 

    for _ in range(15):
        u -= lr * dEdu(u, v)
        v -= lr * dEdv(u, v)

    print(f"Final error for coordinate descent after 15 iterations: {E(u, v)}")

if __name__ == "__main__":
    p5p6()
    p7()
