
def to_bits(value, bits):
  ret = [0]*bits
  for bit in range(bits-1, -1, -1):
    if value >= 2**bit:
      ret[bits-bit-1] = 1
      value -= 2**bit
  return ret 
