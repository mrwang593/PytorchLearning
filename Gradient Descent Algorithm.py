import matplotlib.pyplot as plt
#导入数据
x_data=[1.0,2.0,3.0,4.0]
y_data=[2.0,5.0,6.0,7.0]

w=1.0
alpha=0.01

#线性模型
def f(x):
    return x*w

#损失函数
def cost(xs,ys):
    cost=0
    for x_val,y_val in zip(xs,ys):
        cost+=(x_val*w-y_val)**2
    return cost/len(xs)

#梯度
def gradient(xs,ys):
    grad=0
    for x,y in zip(xs,ys):
        grad+=2*x*(x*w-y)
    return grad/len(xs)

i_list=[]
mse_list=[]


print("Predict before training:",5,f(5))
for i in range(100):
    mse=cost(x_data,y_data)
    grad=gradient(x_data,y_data)
    i_list.append(i)
    mse_list.append(mse)
    w-=alpha*grad
    print("i=",i,"w=",w,"mse:",mse)
print("Predict after training:",5,f(5))

plt.plot(i_list,mse_list)
plt.xlabel ("i")
plt.ylabel("mse")
plt.show()
