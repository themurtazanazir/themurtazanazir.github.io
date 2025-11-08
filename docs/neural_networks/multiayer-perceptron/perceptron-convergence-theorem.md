---
title:  Perceptron Convergence Theorem
description: A Mathematical proof of perceptron convergence theorem
tags: 
 - neural networks
 - perceptron
 - deep learning
 - feedforward networks
 - proof
 - mathematics
hasplot: true
coll: mlp
---

[← Multi-Layer Perceptron](index.md)

# Perceptron Convergence Theorem

## Introduction
The Perceptron Convergence Theorem is an important result as it proves the ability of a perceptron to achieve its result. This proof will be purely mathematical. There are some geometrical intuitions that need to be cleared first. This proof requires some prerequisites - _concept of vectors_, _dot product of two vectors_.
We will use the `train` function that we developed in the <a href="/notes/perceptron" target="_blank">Mathematics Behind Perceptron </a> post. It will be included in a `utils.py` file which you can download <a href="/downloads/utils.py"> here</a>. Also, ignore the visualization code, if that seems too complex.



## Planes

Since, we know that a Perceptron classifies only linearly separable data with a linear hyper-plane, let's get some things clear about planes.

The equation of an n-dimensional hyper plane is:

$$a_1x_1 + a_2x_2 + \dots + a_nx_n + d = 0 \tag{1}$$

where \\(x_1,x_2,\dots,x_n\\) are orthogonal axes.


Another way of describing a plane is in the form of a vector \\(\mathbf{n} = (a_1,a_2, \dots , a_n)\\) (which is normal to that plane) and a point(position vector) \\(\mathbf{x_0}\\) which resides on that plane.

Now any vector between point, \\(\mathbf{x_0}\\), and any other general point \\(\mathbf{x} = (x_1,x_2,\dots,x_n)\\) on the plane is perpendicular to that plane, i.e:

$$\begin{align} \mathbf{n} \cdot \mathbf{(x-x_0)}&=0 \\ \implies  (\mathbf{n} \cdot \mathbf{x}) - (\mathbf{n} \cdot \mathbf{x_0}) &= 0\end{align} \tag{2}$$

According to Eq. 1 and 2, using the dot product rule,

$$d = - \mathbf{n} \cdot \mathbf{x_0} \tag{3}$$

The shortest distance(with sign) of the plane from the origin, \\(p\\) is given by:

$$p=\frac{d}{\lVert \mathbf{n} \rVert}\tag{4}$$

If we divide Eq.1 by \\(\lVert \mathbf{n} \rVert\\), then,

$$ \frac{a_1x_1 + a_2x_2 + \dots + a_nx_n}{\lVert \mathbf{n} \rVert} = -\frac{d}{\lVert \mathbf{n} \rVert}\\ \implies \frac{\mathbf{n} \cdot \mathbf{x}}{\lVert \mathbf{n} \rVert} = -\frac{d}{\lVert \mathbf{n} \rVert} $$

Using \\(\frac{\mathbf{n}}{\lVert \mathbf{n} \rVert}=\hat{\mathbf{n}}\\) and Eq. 4, we have

$$\hat{\mathbf{n}} \cdot \mathbf{x} = - p \tag{5}$$




The above equation is called the **Hessian Normal Form** of a plane.

In this form, we need a unit vector perpendicular to plane and the distance of the plane from the origin to define the plane.

### Planes with Perceptron

The Perceptron prediction rule is as follows:

$$\sigma=g(h)=\begin{cases}1&\text{if }h>0 \\ 0&\text{if }h\leq0 \\ \end{cases}\tag{6}$$

and,

$$h=\sum_{i=0}^m{w_ix_i}\tag{7}$$

where \\(w_0\\) is the threshold of the neuron and \\(x_0\\) can be any constant, except Zero (we use -1). 

The decison boundary is formed, when \\(h=0\\), so the equation of the decision hyper plane is:

$$\sum_{i=0}^m{w_ix_i}=0\tag{8}$$

Comapring the above equation with the general equation of a plane (Eq. 1), the normal vector, which we will represent by \\(\mathbf{w}\\), is:

$$ \mathbf{n} = \mathbf{w} = (w_1,w_2,\dots,w_m) \tag{9}$$

which means the constant term in the general equtaion, \\(d\\), will be:

$$ d = w_0x_0 \tag{10}$$

which means the perpendicular distance of the plane from origin is,\\(p\\), (from Equation 4):

$$ p = \frac{w_0x_0}{\lVert \mathbf{w} \rVert} \tag{11}$$




Now we have the perpendicular vector as well as the perpendicular distance. 

Let's see an example of the OR dataset


```python
import numpy as np
from utils import train
np.set_printoptions(precision=2)
# the dataset
X = np.array([[0,0],
             [0,1],
             [1,0],
             [1,1]])
T=np.array([[0],[1],[1],[1]])
```


```python
# fit the data
weights = train(X,T,0.25,15,random_state=42)
```

    Accuracy: 1.0



```python
#plot decision boundaries, weight vector

inp1 = np.linspace(0,0.5,10000)

fig  = go.Figure(layout=dict(height=800,width=800,
                            xaxis_title="Input 1", yaxis_title="Input 2",
                            title="Decision Boundary with Scatter",
                            autosize=False,
                            legend=dict(x=0.5,y=1),
                            annotations=[dict(x=weights[1,0],y=weights[2,0],ax=0,ay=0,arrowhead=1,
                                             startarrowsize=4,axref="x",ayref="y"),
                                        dict(x=weights[1,0],y=weights[2,0],showarrow=False,text=f"Weight Vector {weights[1:,0]}",
                                             yanchor="bottom"),
                                        dict(x=0.07,y=0.14,showarrow=False,text="Distance, p",
                                             yanchor="middle", xanchor="center", textangle=-40),
                                        
                                        ],
                             shapes=[go.layout.Shape(type="path",path="M 0,0 L 0,0.04 L 0.18673,0.21764 L 0.22673,0.21764",
                                                     line_color="MediumPurple",),
        
                                     
    ]
                            )
                )
fig.add_trace(go.Scatter(x=X[:,0:1][T==1],y=X[:,1:][T==1],name= "Output: 1",mode="markers",marker=dict(size=20)))
fig.add_trace(go.Scatter(x=X[:,0:1][T==0],y=X[:,1:][T==0],name= "Output: 1",mode="markers",marker=dict(size=20)))
fig.add_trace(go.Scatter(x=inp1,y=(-(weights[1,0]*inp1) + weights[0,0])/weights[2,0],name= "Decision Boundary",mode="lines",marker=dict(size=20)))

fig.show()
```



So we can represent the decision boundary with the weight vector and the bias term. 

The next step is to develop a comparison technique between two decision boundaries. The idea is to show that the decision boundary that our model is creating is close to the actual decision boundary. We will talk more about it in a moment.

We can just compare the unit vectors of the two weight vectors and the constant bias terms to find if the decision boundaries are similar or even same.

But this solution needs multiple comparisons, a vector comparison and a bias weight comparison. A better way is to consider this is to consider the bias weight as part of the weight vector. 

So, in our OR example, the data will be considered 3-dimensional, with the 3rd dimension being constant(we have chosen -1).

For 3-dimensions, the decision boundary will be a plane, whose general equation is:

$$a_1x_1+a_2x_2+a_3x_3+d=0$$

and expanding Equation 8 for our OR data set, equation of our decision boundary is:

$$w_1x_1+w_2x_2+w_0x_0=0$$

As we we can see, our bias input \\(x_0\\) is the 3rd dimension in this equation, which is constant (-1) in our dataset. The constant term \\(d\\) is Zero.

Since \\(d=0\\), for our decision boundary, it means the distance between the hyper-plane and the origin, \\(p=\frac{d}{\lVert \mathbf{w} \rVert} = 0\\). Which implies the plane always passes through the origin of that higher coordinate system. 

The above result can be generalized for any linearly separable n-dimensional data, that if we consider the bias inputs as part of the data, the resulting n+1 dimensional data will have its decision boundary passing through origin of the new coordinate system.

The above fact stands because in the higher dimension, the constant term is Zero, and so the origin satisfies the new decision boundary.




Let's try to verify this on our linearly separable OR-dataset. Our bias input is constant(-1). Let's plot the data along with the decision boundary represented by our weights (now including the threshold/bias weight as well).


```python
xx,yy=np.meshgrid(np.arange(-2,1.25,0.1),
                  np.arange(-2,1.25,0.1))
Z = (-xx.ravel()*weights[1,0] - yy.ravel()*weights[1,0])/weights[0,0]
Z = Z.reshape(xx.shape)

fig = go.Figure()


#scatter points
fig.add_trace(
    go.Scatter3d(x=X[:,0], # First input
                y=X[:,1], # second input
                z=[-1]*X.shape[0], # bias additional input
                 mode="markers",
                marker=dict(size=10,color=T.squeeze(),colorscale="Viridis"),name="Data Point"
                )
                
)


#decision boundary
fig.add_trace(
    go.Surface(x=xx[0],y=yy[:,0],z=Z,showscale=False,colorscale="Viridis",opacity=0.8
              ,name="Decision Boundary")
)


# weight vector
fig.add_trace(
    go.Cone(x=weights[1], y=weights[2], z=weights[0], u=weights[1], v=weights[2], w=weights[0], sizemode="scaled",
        sizeref=0.15, anchor="tip", showscale=False, name="Weight Vector", cmax=0, cmin=0, colorscale=[[0,'rgb(0,0,0)'],[1,'rgb(0,0,0)'],]
    )
)
fig.add_trace(
    go.Scatter3d(x=[0,weights[1,0]], y=[0,weights[2,0]], z=[0,weights[0,0]], mode="lines", line=dict(width=4,color="black"), name="Weight Vector"
    )

)


#origin
fig.add_trace(
    go.Scatter3d(
        x=[0],y=[0],z=[0],name="origin", mode="markers", marker=dict(symbol="diamond", size=5,color="green")
    )
)


fig.update_layout(
    scene=go.layout.Scene(
        camera=dict(
            eye=dict(
                x=0,
                y=0,
                z=2.2
            ),
            up=dict(
                x=1,
                y=0,
                z=0
            )
        ),
        dragmode="turntable",
        xaxis=dict(
            title_text="Input 1",
            range=[-0.3,1.5],
        ),
        yaxis=dict(
            title_text="Input 2",
            range=[-0.3,1.5],

        ),
        zaxis=dict(
            title_text="Bias Input",
            range=[-1.3,0.5],
        ),
    ),
    showlegend=False,
    height=700,
    width=700
)

fig.show()
```


If you look through the top of the bias axis, you can see the data points are same as they were in the 2D figure, and the weight vector looks the same as well. And the moment you rotate the above figure you can see this new higher dimensional decision boundary also classifies the data and infact passes through the previous decision boundary(the lower dimensional). Further more, you can also see that this plane has also passed through the origin of the coordinate axis. 

With this intuition in mind, we can conclude that a higher dimensional decision boundary achieves the same goal, and to represent this higher dimensional hyper-plane, we just need the weight vector(which includes the bias weight as well!), since we already know it's distance from the origin will be Zero. 

Now that we have established, that only the complete weight vector(it's unit vector) defines the decision hyper-plane, two same vectors(or vectors in the same direction), will define the same hyper plane.

# The Proof

The Perceptron Convergence theorem proved by Rosenblatt in 1962, states that:

*__"given a linearly separable data, the perceptron will converge to a solution within \\(R/ \gamma^2\\) iterations, where \\(\gamma\\) is the distance between the separating hyperplane and the closest datapoint to it."__*

However there are some assumptions about it:
1. The data should be linearly separable.
2. For every input vector \\(x\\), \\(\lVert \mathbf{x} \rVert\\) is bound by some constant R. In our proof we will assume \\(\lVert \mathbf{x} \rVert \leq R\\).
3. Also the learning rate is chosen to be 1.


Now to the proof,

We know that the data is linearly separable, which means there exists a set of weights which represent the the seperating hyperplane. Let's say these weights are \\(\mathbf{w^\*}\\).

Our learning algorithm tries to find some vector \\( \mathbf{w} \\) that is parallel to \\( \mathbf{w^\*} \\). To see if the vectors are parallel we use the inner product (also called the dot product) \\( \mathbf{w^\*} \cdot \mathbf{w} \\).

So,

$$ \mathbf{w^*} \cdot \mathbf{w} = \lVert \mathbf{w^*}\rVert \   \lVert \mathbf{w}\rVert \cos\theta \tag{12}$$

now if two vectors are parallel the angle is \\(0\\), and \\(\cos{0}=1\\) and so the inner product is maximum. **If we show that after each update \\(\mathbf{w^\*} \cdot \mathbf{w}\\) increases, we have shown that the perceptron converges.** However we need to be a bit more careful as \\(\mathbf{w^\*} \cdot \mathbf{w}\\) can also increase if \\(\lVert \mathbf{w} \rVert\\) increases, we also need to check the length of \\(\mathbf{w}\\) doesn't increase too much.

Keeping all that in mind, let's move on.

Suppose at the \\(i^{th}\\) iteration of the algorithm, the network sees a particular input vector \\(\mathbf{x}\\) that has a target(ground truth) \\(t\\), and the output(prediction) was \\(y\\). 

Now, if the output was wrong, then

$$(t-y)(\mathbf{w^{(i-1)}} \cdot \mathbf{x}) < 0 \tag{13}$$

To see this result clearly, look at a few examples:

e.g, if the target was 1 and output was 0, then,

$$t-y = 1$$ 

and 

$$\begin{align}\mathbf{w^{(i-1)}} \cdot \mathbf{x} < 0\\ \implies (t-y)(\mathbf{w^{(i-1)}} \cdot \mathbf{x}) < 0 \end{align} $$

or if the target was 0 and output was 1, then,

$$t-y=-1$$ 

and 

$$\begin{align}\mathbf{w^{(i-1)}} \cdot \mathbf{x} > 0 \\ \implies (t-y)(\mathbf{w^{(i-1)}} \cdot \mathbf{x}) < 0\end{align}$$

Now that there is error, the weights need to be updated according to the perceptron updation rule,

$$ \mathbf{w^{(i)}} = \mathbf{w^{(i-1)}} - \eta (y-t)\mathbf{x} \tag{14}$$

Note: *Please clear it that the above equations have been generalised as \\(\mathbf{w^{(i)}}\\) is not a single weight, but a vector of weights to a neuron from the input nodes at \\(i^{th}\\) iteration and \\(\mathbf{x}\\) is not a single input but the input vector(including the bias input), \\(y\\) and \\(t\\) are output and target of a single neuron and the symbol \\(\cdot\\) represents inner product of two vectors.*

Coming back to above equation, our proof assumes the learning rate of 1 and let's use \\(t-y\\) instead of \\(y-t\\)(which changed sign, ofcourse!).

$$\implies \mathbf{w^{(i)}} = \mathbf{w^{(i-1)}} + (t-y)\mathbf{x} \tag{15}$$

Now to show that \\(w^\* \cdot w\\) increases with iterations, using Equation 15,

$$ \begin{align} \mathbf{w^*} \cdot \mathbf{w^{(i)}} &= \mathbf{w^*} \cdot (\mathbf{w^{(i-1)}} + (t-y)\mathbf{x})\\ &= \mathbf{w^*} \cdot \mathbf{w^{(i-1)}} + \mathbf{w^*} \cdot (t-y)\mathbf{x}\\ &= \mathbf{w^*} \cdot \mathbf{w^{(i-1)}} + (t-y)(\mathbf{w^*} \cdot \mathbf{x})\\ \end{align} \tag{16} $$

This is where we take a break  and think what \\(\mathbf{w^\*} \cdot \mathbf{x}\\) represents.

If you have any idea about the signed distance of a point \\(\mathbf{x}\\) from a hyper-plane(which passes through origin) with coefficient vector \\(\mathbf{a}\\) is:

$$\begin{align} D = \frac{\mathbf{a} \cdot \mathbf{x}}{\lVert \mathbf{a} \rVert} \\ \implies D \lVert \mathbf{a} \rVert = \mathbf{a} \cdot \mathbf{x} \end{align} \tag{17}$$

Similarly \\(\mathbf{w^\*} \cdot \mathbf{x}\\) represents \\(\lVert \mathbf{w^\*} \rVert\\) times the *signed* distance between the point \\(\mathbf{x}\\) and the plane represented by our vector \\(\mathbf{w^\*}\\) (which is the correct decision boundary)

By signed distance, I mean the sign regarding to what side of the plane the data point lies. Now if we have made an error and misclassified the point, then \\((t-y)\\) will be the opposite the sign of the sign of the distance given by the correct boundary. Give it a little thought, work out on different cases, you'll get it.

So \\((t-y)(\mathbf{w^\*} \cdot \mathbf{x})\\) represents \\(\lVert \mathbf{w^\*} \rVert\\) times the *magnitude* of distance between the point \\(\mathbf{x}\\) and the plane represented by our vector \\(\mathbf{w^\*}\\) (which is the correct decision boundary). And the smallest distance between the correct decision boundary and any datapoint is \\(\gamma\\).

So,

$$(t-y)\mathbf{w^*} \cdot \mathbf{x} \geq \gamma \lVert \mathbf{w^*} \rVert \tag{18}$$

Using the above equation in Equation 16,

$$ \mathbf{w^*} \cdot \mathbf{w^{(i)}} \geq \mathbf{w^*} \cdot \mathbf{w^{(i-1)}} + \gamma \lVert \mathbf{w^*} \rVert \tag{19} $$

where \\(\gamma\\) is the minimum distance between the optimal hyperplane defined by \\(\mathbf{w^\*}\\) and the closest datapoint to it.

Now according to above equation, \\(\mathbf{w^\*} \cdot \mathbf{w^{(i)}}\\) always increases by at least \\(\gamma\\) and we initialize the weights to be small random numbers(positive and negative), so after \\(i\\) iterations

$$\mathbf{w^*} \cdot \mathbf{w^{(i)}} \geq i\gamma \lVert \mathbf{w^*} \rVert \tag{20}$$

So we have proved that \\(\mathbf{w^\*} \cdot \mathbf{w^{(i)}}\\) increases as iterations increase.

Since the R.H.S of the above equation is positive,

$$\lvert \mathbf{w^*} \cdot \mathbf{w^{(i)}} \rvert \geq i\gamma \lVert \mathbf{w^*} \rVert \tag{21}$$


Also we use **Cauchy–Schwarz inequality**,

$$ \lvert \mathbf{w^*}  \cdot \mathbf{w^{(i)}} \rvert \leq \lVert \mathbf{w^*} \rVert \ \lVert \mathbf{w^{(i)}} \rVert \tag{22} $$

Using Equations 21 and 22,

$$\begin{align} & i\gamma \lVert \mathbf{w^*} \rVert \leq \lvert \mathbf{w^*} \cdot \mathbf{w^{(i)}} \rvert \leq \lVert \mathbf{w^*} \rVert \ \lVert \mathbf{w^{(i)}} \rVert \\ \implies & i\gamma \lVert \mathbf{w^*} \rVert \leq \lVert \mathbf{w^*} \rVert \ \lVert \mathbf{w^{(i)}} \rVert \end{align} \tag{22}$$

\\(\lVert \mathbf{w^\*} \rVert\\) is positive and can be cancelled without affecting the inequality.

$$ i\gamma \leq \lVert \mathbf{w^{(i)}} \rVert \tag{23} $$

Here we have put a lower limit to \\(\lVert \mathbf{w^{(i)}}\rVert\\). Now we have to make sure that \\(\lVert \mathbf{w^{(i)}} \rVert\\) does not increase too much. 

For that we will use the Equation 15 and square the norm on both sides and use the Vector addition rules:
$$ \begin{align} \lVert \mathbf{w^{(i)}}\rVert ^2 &= \lVert \mathbf{w^{(i-1)}} + (t-y)\mathbf{x} \rVert ^2\\ &= \lVert \mathbf{w^{(i-1)}}\rVert ^2 + (t-y)^2 \lVert \mathbf{x}\rVert ^2 + 2(t-y)\mathbf{w^{(i-1)}} \cdot \mathbf{x}  \tag{24} \end{align} $$

Now using Perceptron rule, assumption 2(i.e, \\( \lVert \mathbf{x} \rVert \\) is bound by some constant \\( R \\)) and Equation 13 repectively, we have

$$\begin{align}(t-y)^2 = 1\\  \lVert \mathbf{x} \rVert \leq R\\  (t-y)(\mathbf{w^{(i-1)}} \cdot \mathbf{x}) < 0 \end{align} \tag{25}$$

Remember, the last equation here is when the output is incorrect.

So using the above equations (25):

$$(t-y)^2 \lVert \mathbf{x}\rVert ^2 + 2(t-y)\mathbf{w^{(i-1)}} \cdot \mathbf{x} \leq R$$

Using the above equation in Equation 24,

$$ \lVert \mathbf{w^{(i)}}\rVert ^2 \leq \lVert \mathbf{w^{(i-1)}}\rVert ^2 + R \tag{26} $$

Which shows \\(\lVert \mathbf{w^{(i)}}\rVert ^2\\) does not increase more than what the input data is bound by. If we normalize the input before training, then the data will be bound by 1.

Now according to Equation 26, after i iterations,

$$ \lVert \mathbf{w^{(i)}}\rVert ^2 \leq iR \tag{27} $$

Here we have put an upper limit on \\(\lVert \mathbf{w^{(i)}}\rVert\\).

We have shown that \\(\mathbf{w^\*} \cdot \mathbf{w^{(i)}}\\) increases by atleast \\(i\gamma \lVert \mathbf{w^\*}\rVert\\) and \\(\lVert \mathbf{w^{(i)}}\rVert\\) does not increase more than \\(iR\\). Which means the angle between the vectors is decreasing and so the predicting decision boundary is getting close to the actual decision boundary, and after certain weight updates, the algorithm will converge to the actual boundary.

Using Equation 23 and 27,

$$i\gamma \leq \lVert \mathbf{w^{(i)}}\rVert  \leq \sqrt{iR}\\ \implies i \leq \frac{R}{\gamma^2} \tag{28}$$

So within \\(R/\gamma^2\\) iterations, the algorithm must have converged.

We have shown that if the data is linearly separable, then the algorithm will converge, and the time it will take is a function distance between the separating hyperplane and the nearest point. This is actually called **margin**.

***Note: The perceptron stops learning as soon as it gets all the data correctly classified, and so there is no guarantee that it will find the largest margin, just that if there is a separator, it will find it.***

Note: By weight update we mean just update weight when the algorithm makes an error and so only those weight updates count to the iterations. We do not count the examples which the algorithm classifies correctly.
