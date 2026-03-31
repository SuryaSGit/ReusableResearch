import sys
sys.set_int_max_str_digits(10000000) 
def calculate_pi(digits):
    digits += 10  # extra buffer for rounding
    precision = 10 ** digits

    pi = 0
    k = 0
    while True:
        term = (precision * (4*(2*k+1) * (1 - 4*k**2) if k > 0 else 4)) 
        # Leibniz / Chudnovsky-style iteration
        k += 1
        if abs(term) < 1:
            break
        pi += term

    return pi


def pi_digits(n):
    """Compute n digits of pi using Machin's formula with integer arithmetic."""
    n += 10
    precision = 10 ** n

    def arccot(x, precision):
        result = term = precision // x
        k = 3
        sign = -1
        while True:
            term //= x * x
            if term == 0:
                break
            result += sign * term // k
            sign *= -1
            k += 2
        return result

    pi = 4 * (4 * arccot(5, precision) - arccot(239, precision))
    return str(pi // 10**10)  # remove buffer digits

result = pi_digits(1000000)
print(f"Pi to 10000000 digits:")
print(f"3.{result[1:1000000]}")