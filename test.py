from engine import *

a=NVal([2.0,3.0,4.0],req_grad=True)
b=NVal([1.0,1.0,1.0],req_grad=True)
c=NVal(2)
d=NVal(3)
e=NVal(0)

# print(a)
# print(b)
# print(a+b)
# print(-a)
# print(a-b)
print("c ",c)
print("d ",d)
print("c+d ",c+d)
print("c-d ",c-d)
print("c*d ",c*d)
print("c^d ",c**d)
print("c/d ",c/d)
# print("c/e ",c/e)

# k=(a+b);req_grad=True
# l=a-k;req_grad=True
# l.backprop()

# print(l.grad)