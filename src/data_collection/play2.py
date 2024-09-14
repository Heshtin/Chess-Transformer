import time
x = 10
start = time.perf_counter()
x *= 10
end = time.perf_counter()
print("Time for *= operation:", end-start)

y = 10
start = time.perf_counter()
y = y * 10
end = time.perf_counter()
print("Time for = * operation:", end-start)

# import time
# x=10
# start = time.time()
# x *= 10
# end = time.time()
# print(end-start)
# y=10
# start = time.time()
# y = y * 10
# end = time.time()
# print(end-start)