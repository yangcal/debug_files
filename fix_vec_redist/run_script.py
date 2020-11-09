import ctf

vec = (1+.5j) * ctf.random.random([150994944])
tsr = vec.reshape(64,64,64,2,2,12,12)
print("done", tsr.shape, vec.size)
