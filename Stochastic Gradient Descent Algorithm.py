import matplotlib.pyplot as plt
#导入数据
x_data=[1.0,2.0,3.0,4.0]
y_data=[2.0,5.0,6.0,7.0]

w=1.0
alpha=0.01
l=0
#线性模型
def f(x):
    return x*w

#损失函数
def loss(x,y):
    loss=(f(x)-y)**2
    return loss

#梯度
def gradient(x,y):
    grad=2*x*(x*w-y)
    return grad

print("Predict before training:",5,f(5))
for i in range(100):
    for x,y in zip(x_data,y_data):
        grad=gradient(x,y)
        w-=alpha*grad
        print("\tgrad:",x,y,grad)
        l = loss(x, y)
    print("i=",i,"w=",w,"loss:",l)

print("Predict after training:",5,f(5))


