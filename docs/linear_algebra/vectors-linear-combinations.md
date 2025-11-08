---
title:  Vectors, Linear Combinations, Eliminations 
description: Basics of Linear Algebra
coll: Linear Algebra
tags: 
 - vectors
 - elimination matrix
 - permutation matrix
 - linear combinations
hasplot: true
coll: lin_alg
---

[← Home](../index.md)

# Vectors, Linear Combinations, Eliminations 

### Introduction

The whole field of linear algebra, as the name suggests, is based on linear combinations of different "things". We will get to know what these "things" are. How can we represent the different combinations and what these different combinations represent, if they represent anything. We will basically work with vectors and then with matrices. So let's begin. 

## Vectors

### Introduction

A vector is, simply put, a data holding structure. It can hold entries of data. We can store values of a specific feature in a vector. We can store coordinates of any point in a n-dimensional space. 

The specific function of a vector holding coordinates of a point is used the most. The same data can represent an arrow from origin to the point stored in the vector(the definition physicists usually identify vectors by).

So a vector can represent just $n$ numbers(data), Arrow from origin, or a point in a space. The vector is said to be $n$-dimensional.

### Representation of a Vector

How do we represent vectors? There are basically two ways to do it. A column vector and a row vector. In a column vector, we stack all the numbers in a single vertical fashion and in a row vector, we stack them in horizontal fashion. While both are fine and can be converted from one form to another, the column vector is conventionally used.

e.g: a vector containing 2 and 3 is represented as:

$$ \mathbf{a} = \begin{bmatrix} 2 \\ 3 \end{bmatrix}$$ 

This vector corresponds to point $(2,3)$.

A column vector(or a row vector) can also be represented by paranthesis. So the above vector can also be $(2,3)$. It does not mean it is a row vector. It is actually a column vector(in this case), because we were using column vector. It is just easier to write it.

A 3-dimensional column vector $\begin{bmatrix} x  \\ y  \\ z \end{bmatrix}$ can also be written as $( x,y,z)$ and it sill is column vector. A row vector will be $\begin{bmatrix} x & y & z \end{bmatrix}$.

For now we will primarily represent vectors as columns, unless specified otherwise.

We can do it in a simple way in julia.


```julia
vector = [1, 2, 3]
```




    3-element Array{Int64,1}:
     1
     2
     3



It can be also created like:


```julia
vector = [1; 2; 3]
```




    3-element Array{Int64,1}:
     1
     2
     3




```julia
vector = [1
          2
          3]
```




    3-element Array{Int64,1}:
     1
     2
     3



### Operations on Vectors

In linear algebra, we work with two important operations, **multiplication by a scalar** and **addition**.

When we multiply a vector by a scalar, all the values in the vector (called the *components*) are multiplied by the same number.

e.g:

$$  c\begin{bmatrix} 1 \\ 2 \\ 3\end{bmatrix} = \begin{bmatrix} c \\ 2c \\ 3c\end{bmatrix}$$ 

We can add two vectors if their dimensions are same. The resultant vector is the vector of corresponding sums of components of two vectors.

e.g:

$$  \begin{bmatrix} 1 \\2 \\3 \end{bmatrix} + \begin{bmatrix} 6 \\7\\8\end{bmatrix} = \begin{bmatrix} 7 \\ 9 \\11 \end{bmatrix}$$ 

>If you are familiar with vectors from basic physics, you know the addition of two vectors is the third side of a triangle formed by placing the tail of one vector at the head of the other vector.

If we combine the above two operations on any number of vectors, i.e we multiply each vector by a scalar and then add all the vectors, we get a **linear combination** of those vectors.

So for any two vectors, $\mathbf{w}$ and$\mathbf{v}$, a linear combination is:

$$  c \mathbf{w} + d\mathbf{v}$$ 

where $c,d$ are scalars. 

for any $n$ vectors, $\mathbf{v}_1,\mathbf{v}_2,\dots$ a linear combination will be:

$$  c_1 \mathbf{v}_1 + c_2 \mathbf{v}_2 + \cdots + c_n\mathbf{v}_n $$ 

We will talk about what all these linear combinations for all $c$ and $d$ represent, later.

Example in julia for three vectors with scalars being 2,3,-1.


```julia
v1 = [1;2;3;4]
v2 = [4;3;5;1]
v3 = [22,45,0,1]
c1,c2,c3 = 2,3,-1
c1*v1 + c2*v2 + c3*v3
```




    4-element Array{Int64,1}:
      -8
     -32
      21
      10



Besides the two important operations in linear algebra, we can do other operations on vectors.

**Dot Product:** We can multiply two vectors to get a scalar. It is called a dot product. It can occur between two vectors of same dimensions. We just multiply the corresponding elements of the two vectors and sum up the products.

e.g:


$$ \begin{bmatrix} 1 \\ 2 \end{bmatrix} \cdot \begin{bmatrix} 4 \\ 5\end{bmatrix} = (1)(4)+(2)(5) = 14$$ 

We can achive this in julia as:


```julia
using LinearAlgebra
```


```julia
v1 = (1,2);
v2 = (4,5);

dot(v1,v2)
```




    14



It can be also calculated as:

$$ \mathbf{v_1} \cdot \mathbf{v_2} = \mathbf{v_1}^T \mathbf{v_2}$$  

$\mathbf{v_1}^T \mathbf{v_2}$ is also called the **inner product** and the $\mathbf{v_1} \mathbf{v2}^T$ is called the **outer product**. The outer product produces a matrix.



For two vectors, $\mathbf{w}$ and $\mathbf{v}$, the dot product is also calculated as:

$$  \mathbf{w} \cdot \mathbf{v} = \lVert \mathbf{w} \rVert \lVert \mathbf{v} \rVert \cos \theta $$ 

where $\lVert \mathbf{x} \rVert$ is the length of vector $\mathbf{x}$ and $\theta$ is the angle between the two vectors.

Because of the cosine, the dot prouct of perpendicular vectors is zero. Also dot product of a vector with itself gives the square of its length.

Since $\cos \theta \leq 1$,

$$  \lvert \mathbf{w} \cdot \mathbf{v} \rvert \leq \lVert \mathbf{w} \rVert \lVert \mathbf{v} \rVert $$ 

It is the Cauchy-Schwarz-Buniakowsky inequality.

## Matrices

As we have established, we save data in vectors. Now we work with multiple vectors (for their linear combinations of course!). We will save multiple vectors in a matrix. Let's do some examples.

Let's say we have three vectors:

$$  \mathbf{u} = \begin{bmatrix} 1 \\ -1 \\0\end{bmatrix} \quad ; \quad \mathbf{v} = \begin{bmatrix} 0 \\ 1 \\-1\end{bmatrix} \quad ;\quad \mathbf{w} = \begin{bmatrix} 0 \\ 0 \\1\end{bmatrix}$$ 

Their linear combination, $\mathbf{b}$ for scalars $c_1$, $c_2$ and $c_3$  is:

$$ \mathbf{b} = c_1\mathbf{u} + c_2 \mathbf{v} + c_3 \mathbf{w} = c_1\begin{bmatrix} 1 \\ -1 \\0\end{bmatrix} + c_2 \begin{bmatrix} 0 \\ 1 \\-1\end{bmatrix} + c_3 \begin{bmatrix} 0 \\ 0 \\1\end{bmatrix} = \begin{bmatrix} c_1 \\ c_2 - c_1 \\ c_3- c_2 \end{bmatrix}$$ 

Now if we use all our vectors as columns of a matrix, $A$, that matrix multiplies the vector $\mathbf{c} = (c_1,c_2,c_3)$:

$$  A\mathbf{c} = \begin{bmatrix} 1 & 0 & 0 \\ -1 & 1 & 0 \\ 0 & -1 & 1 \end{bmatrix}\ \begin{bmatrix}c_1\\ c_2 \\ c_3 \end{bmatrix} = \begin{bmatrix} c_1 \\ c_2 - c_1 \\ c_3- c_2 \end{bmatrix} = \begin{bmatrix} b_1 \\ b_2 \\ b_3 \end{bmatrix} = \mathbf{b} $$ 



Now instead of being just numbers, $c_1,c_2,c_3$ are now also forming a vector. This gives a new view-point to look at it. The output vector, $A\mathbf{c}$(or $\mathbf{b}$) is a combination of the **columns** of $A$.

It can also be thought as the matrix $A$ above, acts on the vector $\mathbf{c}$ and transforms into vector $\mathbf{b}$. This specific matrix $A$ above is a __"difference matrix"__ because $\mathbf{b}$ contains differences of vector $\mathbf{c}$. The vector $\mathbf{c}$ is the input and the output is $\mathbf{b}$. The top difference is $c_1-c_0 = c_1 -0$

## Linear Equations

### Introduction

Let's change the view again. Upto now, the vector $\mathbf{c}$ was known. The right hand side $\mathbf{b}$ was unknown. Now we think of $\mathbf{b}$ as known and look for $\mathbf{c}$. 

We were earlier asking to compute the linear combination of $c_1\mathbf{u} + c_2 \mathbf{v} + c_3 \mathbf{w}$ to find $\mathbf{b}$. Now we are asking which combination of $\mathbf{u,v,w}$ produces a particular vector $\mathbf{b}$?

From now on we will call the unknown vector $\mathbf{x}$ instead of $\mathbf{c}$, because duh!.

This problem is the __inverse problem__ i.e to find the input vector $\mathbf{x}$ that gives the desired output $\mathbf{b}= A\mathbf{x}$.

**Equations:**
For our previous matrix, $A$,

$$ A\mathbf{x} = \mathbf{b}\\ \implies \begin{bmatrix} 1 & 0 & 0 \\ -1 & 1 & 0 \\ 0 & -1 & 1 \end{bmatrix}\ \begin{bmatrix} x_1 \\ x_2 \\ x_3 \end{bmatrix} = \begin{bmatrix} b_1 \\ b_2 \\ b_3 \end{bmatrix}$$ 

| Equations | Solutions|
---|---|
| $x_1 = b_1$  | $x_1 = b_1$  |
| $-x_1 +x_2 = b_2$ | $x_2 = b_1 + b_2$  |
| $-x_2 +x_3 = b_3$  | $x_3 = b_1 + b_2 + b_3$  |

So the solution vector $\mathbf{x}$, for the above equation is:

$$ \begin{align}\mathbf{x} &= \begin{bmatrix} b_1 \\ b_1 + b_2 \\ b_1 + b_2 + b_3 \end{bmatrix} \\ &= \begin{bmatrix} 1 & 0 & 0 \\ 1 & 1 & 0 \\ 1 & 1 & 1 \end{bmatrix}\ \begin{bmatrix} b_1 \\ b_2 \\ b_3 \end{bmatrix}\end{align}$$ 

Now let's compare the effects on this vector with $A$. $A$ acted on a vector and gave the "differences" of the vector elements and this new matrix gives the "sums" of the elements of the vector it acts on. The new matrix is the inverse of the original $A$ and is shown as $A^{-1}$.

So if,

$$ \begin{align} A\mathbf{x} &= \mathbf{b} \\ \implies \mathbf{x} &= A^{-1}\mathbf{b}\end{align}$$ 


This was the solution for this problem because the matrix $A$ was invertible(we will talk more about it later). 

Let's talk about when there is more or less than one solution.

Let's say we have three equations:

$$  x_1 - x_3 = b_1$$ 

$$  x_2 - x_1 = b_2$$ 

$$  x_3 - x_2 = b_3$$ 


Now if $b_1 = b_2 = b_3 = 0$, then

$\begin{bmatrix} x_1 - x_3 \\ x_2 - x_1 \\ x_3 - x_2 \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \\ 0 \end{bmatrix}$ is solved by all vectors $\begin{bmatrix} x_1 \\ x_2 \\ x_3 \end{bmatrix} = \begin{bmatrix} k \\ k \\ k \end{bmatrix}$



where $k$ is any constant.

Now if, say, $b_1 =1, b_2 =1 , b_3 =1$, then:

$\begin{bmatrix} x_1 - x_3 \\ x_2 - x_1 \\ x_3 - x_2 \end{bmatrix} = \begin{bmatrix} 1 \\ 1 \\ 1 \end{bmatrix}$ has no solution. Left side sums upto 0 and right side to 9. 

But Let's open the problem in the form $A\mathbf{x} = \mathbf{b}$,

$$  A\mathbf{x} = \begin{bmatrix} 1 & 0 & -1 \\ -1 & 1 & 0 \\ 0 & -1 & 1 \end{bmatrix}\  \begin{bmatrix} x_1 \\ x_2 \\ x_3 \end{bmatrix}= \begin{bmatrix} 1 \\ 1 \\ 1 \end{bmatrix} = \mathbf{b} $$ 

Now our question transforms into finding the linear combination of columns of $A$ that is equal to $\mathbf{b}$.

Now if we think about it geometrically, no combination of the columns will the vector $\mathbf{b} =(1,1,1)$. The combinations(of columns) don't fill up the whole three-dimensional space but a plane given by $x + y + z =0$


```julia
A = [1 0 -1; -1 1 0; 0 -1 1]
b=[1; 1; 1];
```


```julia
using Plotly
```

```julia
xx= [j for i=-0.7:0.01:0.7,j=-0.7:0.01:0.7]
yy = xx';
z = -xx - yy

trace1 = scatter3d(x=[0, A[1,1]], y=[0, A[2,1]], z=[0, A[3,1]], mode="lines", line=attr(width=5), name="vector 1")
trace2 = scatter3d(x=[0, A[1,2]], y=[0, A[2,2]], z=[0,A[3,2]], mode="lines", line=attr(width=5), name="vector 2")
trace3 = scatter3d(x=[0, A[1,3]], y=[0, A[2,3]], z=[0,A[3,3]], mode="lines", line=attr(width=5), name="vector 3")
trace4 = scatter3d(x=[0, b[1]], y=[0, b[2]], z=[0,b[3]], mode="lines", line=attr(width=5), name="vector b")
trace5 = surface(x=xx, y=yy, z=z, opacity=0.8, showscale=false, name="Plane")


plot([trace1,trace2,trace3,trace4,trace5], Layout(scene_camera=attr(eye=attr(x=1, y=1.2, z=1))))
```

<div id="plot1">
</div>



Here $A\mathbf{x}$ represents all the linear combination of columns (for all values of $\mathbf{x}$, of course!). Now the sum of two vectors in same direction will have all its linear combinations in that direction. Similarly two vectors in different directions will have all their linear combinations in the plane defined by these two vectors. Now for three vectors, any two vectors will form a plane and if the third vector is not in that plane, then all three vectors are said to be **independent** and their linear combinations will fill up the whole three dimensional space. But if the third vector is a linear combination of the first two (i.e, it is in the same plane), then the vectors are said to be **dependent** and the third vector does not contribute anything new and so the result will be a plane.

Now in our matrix, the third column is a linear combination of the first two and hence the linear combination of these three vectors can only form a plane and we would have a solution if the vector $\mathbf{b}$ was in that plane. And since the vector $(1,1,1)$ is not in plane determined by the columns of the matrix, this equation has no solution. See Figure above.

If all this seems a bit too fast or shallow, don't worry we will cover it in detail later.

### Solving Linear Equations

#### Introduction

Let's just simply start with solving two equations with two variables. A two variable equation represents a line in a two dimensional space and all the points on that line satisfy that equation.

Now if we have two equations and we have to find solutions(s) that satisfy both these equations. Now based on the coefficients, a system of equations may have a unique solution, an infinite number of solution, or no solution.

We have a unique solution, if both the lines intersect at one point and that point is the solution to the system. We have infinite solutions when both lines run over each other. We have no solution if the lines don't intersect at all(like parallel lines).

Let's see an example of a unique solution case.

Let's say we have two equations:


$$  4x - y = 9 \\ 3x + 2y = 4$$ 

The first equation $4x-y=9$ produces a straight line in the $xy$ plane. There are infinite points on it but a special point $(2,-1)$ also satisfies it and is on the line.

The line by second equation $3x + 2y =4$ is another line that also passes through $(2, -1)$ and so the solution is $x=2$ and $y=-1$.


```julia
x = [i for i=-5:1:5]
y1 = 4*x .- 9
y2 = (4 .- (3*x))/2

trace1 = scatter(x=x,y=y1, mode="lines")
trace2 = scatter(x=x,y=y2, mode="lines")

plot([trace1, trace2], Layout(showlegend=false, annotations=[attr(x=2,
            y=-1,
            text="(2,-1)")], title="Equations", xaxis_title="x", yaxis_title="y"))

```


<div id='plot2'>

</div>





This was the _row picture_. Let's look at the column picture.

We will recognize the same system as a vector system. Instead of using numbers, we will use vectors. It can be shown as:

$$ x\begin{bmatrix}4 \\3 \end{bmatrix} + y \begin{bmatrix}-1 \\2 \end{bmatrix} = \begin{bmatrix}9 \\4 \end{bmatrix}$$ 

Now the same problem has transformed into "find the combination of those two vectors that equals the vector on the right. Choosing $x=2$ and $y=-1$ (same as before!), we will get the right result.

Let's see how the column picture reveals the answer. Let's draw each vector, then the vector multipied and the vector on the right.


```julia
vec1 = [4;3]
vec2 = [-1;2]
b = [9;4]

trace1 = scatter(x=[0,vec2[1]], y=[0,vec2[2]],mode="lines", name= "vector 2")
trace2 = scatter(x=[0,vec1[1]], y=[0,vec1[2]],mode="lines", name= "vector 1")
trace3 = scatter(x=[0,b[1]], y=[0,b[2]],mode="lines", name= "b", line=attr(color="red"))
trace4 = scatter(x=[0,2*vec1[1]], y=[0,2*vec1[2]],mode="lines", name= "2 x vector 1", line = attr(color="green"), opacity=0.5)
trace5 = scatter(x=[0,-1*vec2[1]], y=[0,-1*vec2[2]],mode="lines", name= "-1 x vector 2", line= attr(color="blue"))
trace6 = scatter(x=[2*vec1[1],b[1]], y=[2*vec1[2], b[2]], mode="lines", name= "-1 x vector 2", line = attr(color="blue", dash="dash"))
trace7 = scatter(x=[-1*vec2[1],b[1]], y=[-1*vec2[2], b[2]], mode="lines",name= "2 x vector 1", line = attr(color="green", dash="dash"))



plot([trace1,trace2,trace3, trace4,trace5,trace6,trace7],
    Layout(xaxis=attr(scaleanchor="y", scaleratio=1),  width=800))
```




<div id="plot3">

</div>




Here you can see the combination of these vectors lead to the final vector $\mathbf{b}$. 

$$ 2\begin{bmatrix} 4 \\3 \end{bmatrix} -1\begin{bmatrix} -1 \\2 \end{bmatrix} = \begin{bmatrix} 9 \\4 \end{bmatrix}$$ 

By the simple look of the eye the row picture looks better than the column picture. It's all in your right to choose, but I think it is easier to see a combinaton of four vectors in four dimensional space, than to visualize four hyper-planes meeting at a point.

Same problem, different pictures, same solutions. We combine the problem into a matrix problem as:

$$ A\mathbf{x}= \mathbf{b}\\ \implies \begin{bmatrix} 4 & -1 \\ 3 & 2\ \end{bmatrix}\ \begin{bmatrix} x \\ y \end{bmatrix} = \begin{bmatrix} 9 \\ 4 \end{bmatrix}  $$ 

The matrix, $A = \begin{bmatrix} 4 & -1 \\ 3 & 2\ \end{bmatrix}$ is called the **coefficient matrix**. Its rows give the row picture and its columns give you the column picture. And we can try it in julia like.


```julia
A = [ 4 -1; 3 2]
x = [2;-1]
A*x
```




    2-element Array{Int64,1}:
     9
     4



And it has given the same result as the vector $\mathbf{b}$.

In  three-equations-three-variables, the row picture will be that every equation will repersent a plane in the 3-D world and the solution will be all the intersecting points of these planes.

The column picture is three vectors (the vector of coefficients of $x$ in all the equations, the vector of coefficients of $y$ and that of $z$) lying in a 3-D space and their combination resulting in the vector $\mathbf{b}$.

The matrix form represents these both pictures.

#### Elimination

Until now, we just guessed the right answer to a system of equations and just verified the solution. But let's now try to actually find the solution to a linear system.

Let's start with 2 variables and 2 equations and return back to the system example we used before.

$$  4x - y = 9 \\ 3x + 2y = 4$$ 

Now, let's see we can get the answer $(x,y) = (2,-1)$.

This equation is pretty easy to solve. I'm sure you can do as well. 

I would like to eliminate the $x$-part of the second equation in the system. So, I multiply the first equation by $\frac{3}{4}$ and then subtract it from the second equation. 

Resulting,

$$  \begin{align}4x &- y = 9 \\ &\frac{11}{3}y = \frac{-11}{3}\end{align}$$ 

The new equation gives $y = -1$, quickly and substituting that in the first equation we get $x=2$.

That was simple and you have done it tons of times. But I would like to introduce some terms here and maybe even formalize this procedure.

We used the the coefficient of $x$ in the first equation to eliminate the first term in the second equation. This is called a __pivot__ and the variable associated with it (here, $x$) is called a __pivot variable__.  

The pivot can never be zero, as we cannot eliminate a non zero coefficient by it. No matter what we multiply it by we will always get a zero and that subtracted from other equation will not change them at all. 

The resulting system was in a _"upper triangular form"_ (if we align the variables). The pivots always lie on the diagonal of this traingular form after completed elimination.


We have a few more terminology to introduce. But let's look at a few 
more cases of this system. The solution we got before was simple as the solution was unique.

But if solution isn't unique, two cases arise: No solution and infinite solutions.

Let's say the system is:

$$ 3x + 4y = 7 \\ 6x + 8y = 3$$ 

Let's do the simple elimination again. Multiply the first equation by 2 and subtract from the second one to get the triangluar form:

$$ \begin{align}3x + &4y = 7 \\  &0y = -11\end{align}$$ 

There is no solution $0y=-11$. Usually we would divide -11 by second pivot, but here is no second pivot(Zeros are not pivots!) making it a no solution system. In the row form it is represented by two parallel lines. The column picture shows that the vector $\begin{bmatrix} 3 \\6 \end{bmatrix}$ and the vector $\begin{bmatrix} 4 \\8 \end{bmatrix}$ are in the same direction. And so all it's linear combinations will be restricted on a line along that direction. And the vector $\mathbf{b} = \begin{bmatrix} 7 \\3 \end{bmatrix}$ does not lie on that line and so no combination can do it. Hence no solution.

For the infinite soution let's just change the vector $\mathbf{b}$ to a vector which is on that line, say $\begin{bmatrix} 1 \\2 \end{bmatrix}$.

So the equations are:

$$ 3x + 4y = 1 \\ 6x + 8y = 2$$ 

Eliminating..., we get:

$$ \begin{align}3x + &4y = 7 \\  &0y = 0\end{align}$$ 

Every $y$ satisfies $0y=0$. The $y$ is said to be a __"free variable__", i.e we can choose $y$ freely and $x$ is then computed using the first equation. There is actually just one equation $3x + 4y =7$. This also has just one pivot. 

In the row picture, both lines are the same. In the column picture the vector $\mathbf{b} = \begin{bmatrix} 1 \\ 2 \end{bmatrix}$ lies on the line determined by the coefficient vectors. In fact it is one-third times the first column and it can also be defined as one-fourth times the second column and hence we can have an infinte combinations of these vectors that can end up on that vector $\mathbf{b}$.

Sometimes we may have to exchange rows. It is because we want to have that traingular form, there might occur a Zero where there should be a pivot and we can exchange that row with the nearest row below which has a non-zero number at that place. e.g, a system is:

$$ 0x + 3y = 9 \\ 3x + 4y = 0$$ 

Here we have a Zero where there should be a pivot in the first row. So we cannot perform elimination. But we can exchange it with the row below to get a pivot at that place and proceed with elimination.

$$  \begin{align}3x + &4y = 0 \\  &3y = 9\end{align} $$ 


Now this form is already in the trianglular form and can be back-substituted. So, $y=3$ from equation 2 and so $x = -4$ from the first. 

To understand Gaussian elimination, we need to go beyond 2-equations and two variables. Let's try a 3-equation and 3-variable system.

Let's say a system is:

$$ \begin{align}\mathbf{2}&x + 4y - 6z = -2 \\ 4&x+3y-8z = 0 \\ -2&x + 6 y + 7z =3\end{align} $$ 

The first pivot is in the bold-face. Now as we have to make it traingular, we will have to eliminate all the $x$ terms in the second and third equation. 

So subtracting $2$ times the first equation from second and $-1$ times the first equation from third.

$$ \begin{align}2x + 4y - 6z &= -2 \\ \mathbf{-5}y+4z &= 4 \\  10y + z &=1\end{align}$$ 

Now the second pivot is shown in bold-face. We need to eliminate the $y$ term in the third equation with it. So subtracting $-2$ times the second equation from the third,

$$ \begin{align}2x + 4y - 6z &= -2 \\ -5y+4z &= 4 \\   \mathbf{9}z &=9\end{align}$$ 

The third pivot is in bold-face. We have finished the forward pass of the Elimination. The pivots are 2,-5,9. The last two pivots were hidden in original system but elimination revealed them. Now it is ready for back substitution and

$z = 1$ and so $y = 0$ and finally $x = 2$.

The row picture has three planes which meet at one point $(2,0,1)$.

The column picture has three vectors in the 3d space whose specific combination $2\mathbf{v_1} + 0 \mathbf{v_2} + 1 \mathbf{v_3}$ produce the vector $\mathbf{b}$, the output vector.

So the process of elimination can be summarized as, for any n by n problem:

1. Use first equation\* to create zeroes below the first pivot.

2. Use the second equation\* to create zeroes below the second pivot.

3. Keep going untill you get an triangular form or all the equations below have Zero coefficients.

4. \*Do row exhanges wherever necessary, as Zero cannot be a pivot.

5. If you get a complete triangular form, then back substitution will reveal the unique solution. If you get all the coefficients to be Zero then if the cooresponding output is also zero, you have infinite solution, and if the corresponding output is not zero then there is no solution.

#### Matrix Multiplication 

Before talking about further elimination, let's talk about matrix multiplication a bit more.

Let's say we multiply a matrix and a column vector. The result will be another column vector., e.g:

$$ \begin{bmatrix} 2 & 4 & -1 \\ 1 & 0 & 2 \end{bmatrix}\ \begin{bmatrix}1 \\ 2\\ 3\end{bmatrix}$$ 

This multiplication, by the simple formula of multiplying rows and columns and sum upto get one element of the resultant matrix, gives us the result:


$$ \begin{bmatrix} 2 & 4 & -1 \\ 1 & 0 & 2 \end{bmatrix}\ \begin{bmatrix}1 \\ 2\\ 3\end{bmatrix} = \begin{bmatrix} (2)(1) + (4)(2) + (-1)(3) \\ (1)(1) + (0)(2) + (2)(3) \end{bmatrix} = \begin{bmatrix} 7 \\ 7 \end{bmatrix}$$ 

Although this multiplication can also be represented as a linear combination of columns of the matrix using the elements of the vector:

$$ \begin{bmatrix} 2 & 4 & -1 \\ 1 & 0 & 2 \end{bmatrix}\ \begin{bmatrix}1 \\ 2\\ 3\end{bmatrix} = 1 \begin{bmatrix} 2 \\ 1 \end{bmatrix} + 2 \begin{bmatrix} 4 \\ 0 \end{bmatrix} + 3 \begin{bmatrix} -1 \\ 2 \end{bmatrix} =  \begin{bmatrix} 7 \\ 7 \end{bmatrix} $$ 

Similarly if we multiply two matrices, each column of the resultant matrix will be a linear combination of columns of **left matrix** using the elements of the corresponding **column** of **right matrix** as multipliers. If it seems too much, let's see an example:

$$ \begin{bmatrix} 2 & 4 & -1 \\ 1 & 0 & 2 \end{bmatrix}\ \begin{bmatrix}1 & 4 \\ 2 & 5\\ 3 & 6\end{bmatrix} = \begin{bmatrix}7 & 22\\ 7 & 16 \end{bmatrix}$$ 

It can broken into two parts: 

1. The left matrix multiplying the **first column** of the right matrix to give the **first column** of the resultant matrix.

2. The left matrix multiplying the **second column** of the right matrix to give the **second column** of the resultant matrix.

So in the above example the <mark style="background-color: cyan;">first column</mark> of the resultant matrix is the first column plus twice the second column plus three times the third column of the left matrix,i.e a linear combination of columns of left matrix with the multipliers being the elements of the <mark style="background-color: cyan;">first column</mark> of right matrix.

Similarly we can see the <mark style="background-color: yellow;">second</mark> <mark style="background-color: lightgreen;">column</mark> of the resultant matrix again being the linear combination of <mark style="background-color: lightgreen;">columns</mark> of the left matrix with the numbers of the <mark style="background-color: yellow;">second</mark> <mark style="background-color: lightgreen;">column</mark> of the right matrix as multipliers.

Again if we multiply a row vector and a matrix,

$$ \begin{bmatrix} 2 & 4 & -1 \end{bmatrix}\ \begin{bmatrix}1 & 4 \\ 2 & 5\\ 3 & 6\end{bmatrix}$$ 

The result will be a row vector which will be linear combination of **rows** of the matrix with multipliers being elements of the row vector.

$$ \begin{bmatrix} 2 & 4 & -1 \end{bmatrix}\ \begin{bmatrix}1 & 4 \\ 2 & 5\\ 3 & 6\end{bmatrix} = 2 \begin{bmatrix} 1 & 4 \end{bmatrix} + 4 \begin{bmatrix} 2 & 5 \end{bmatrix} + (-1) \begin{bmatrix} 3 & 6 \end{bmatrix} = \begin{bmatrix} 7 & 22 \end{bmatrix}$$ 

Similarly, if we multiply two matrices we can show that each row is a linear combination of rows of the **right** matrix with elements from the corresponding **row** of the **left** matrix are the multipliers.

So a matrix multiplication can be shown as linear combination of rows(of the right matrix) as well as a linear combination of columns(of the left matrix).

We will put the above facts, particularly about the row combination to our use to perform subtraction of equations and hence elimination.

#### Elimination Using Matrices

Let's use the system again we used before for this,

$$ \begin{align}2&x + 4y - 6z = -2 \\ 4&x+3y-8z = 0 \\ -2&x + 6 y + 7z =3\end{align} $$ 

This system can be conveniently changed into,


$$  \begin{bmatrix} \mathbf{2} & 4 & -6 \\ 4 & 3 & -8 \\ -2 & 6 & 7\end{bmatrix}\ \begin{bmatrix}x \\ y \\ z \end{bmatrix} = \begin{bmatrix}-2 \\ 0 \\3 \end{bmatrix}$$ 

and so can be also written as $A\mathbf{x} = \mathbf{b}$


```julia
A = [2 4 -6;
     4 3 -8;
    -2 6  7]

b = [-2; 0; 3];
```

Now, to perform first step of elimination, we have to remove the elements below the first pivot (bold face) using row subtractions. Now whatever row changes we make in $A$, we have to make in $\mathbf{b}$ as well.

We have to subtract 2 times the first row from the second and -1 times the first row from the third. To perform this operation, I will multiply both sides by a matrix $E_1$.

I would say the matrix $E_1$ is:

$$  E_1 = \begin{bmatrix}1 & 0 & 0 \\ -2 & 1 & 0 \\ 1 & 0 & 1 \end{bmatrix}$$ 


```julia
E1 = [1 0 0;
     -2 1 0
      1 0 1];
```

This matrix, when pre-multiplied, will always perform the above operation on any matrix(which is compatible for matrix multiplication, of course!). Let's see why.

Let's think of the resultant matrix. The first row of that matrix will be a combination of the rows of the right matrix (on which the operation is being done on) with the first row of $E_1$ being the multipliers. 

First row of resultant matrix: $1 \times \text{row1}$ (of original matrix) + $0 \times \text{row2} + 0 \times \text{row3} = \text{row1}$ 

So the first row is not changed.

Similarly second row of resultant matrix: $-2 \times \text{row1} + 1 \times \text{row2} + 0 \times \text{row3} = \text{row2} - (2)\ \text{row1}$

So we have subtracted twice row 1 from row 2.

Third row of resultant matrix: $1 \times \text{row1} + 0 \times \text{row2} + 1 \times \text{row3} = \text{row3} - (-1)\text{row1}$

So we have subtracted -1 times row 1 from row 3.

Let's now perform $E_1A$ and $E_1\mathbf{b}$:


$$ E_1A = \begin{bmatrix}1 & 0 & 0 \\ -2 & 1 & 0 \\ 1 & 0 & 1 \end{bmatrix}\ \begin{bmatrix} 2 & 4 & -6 \\ 4 & 3 & -8 \\ -2 & 6 & 7\end{bmatrix} = \begin{bmatrix} 2 & 4 & -6 \\ 0 & -5 & 4 \\ 0 & 10 & 1\end{bmatrix}$$ 

$$ E_1\mathbf{b} = \begin{bmatrix}1 & 0 & 0 \\ -2 & 1 & 0 \\ 1 & 0 & 1 \end{bmatrix}\ \begin{bmatrix} -2 \\ 0\\ 3\end{bmatrix} = \begin{bmatrix} -2 \\ 4\\ 1\end{bmatrix}$$ 


```julia
E1*A
```




    3×3 Array{Int64,2}:
     2   4  -6
     0  -5   4
     0  10   1




```julia
E1*b
```




    3-element Array{Int64,1}:
     -2
      4
      1



So now our equation is:

$$ E_1A\mathbf{x} = \begin{bmatrix} 2 & 4 & -6 \\ 0 & \mathbf{-5} & 4 \\ 0 & 10 & 1\end{bmatrix}\ \begin{bmatrix} x \\ y\\z\end{bmatrix} = \begin{bmatrix} -2 \\ 4\\ 1\end{bmatrix} = E_1\mathbf{b}$$ 

The second pivot(bold face) has to remove the elements below it. Now we have to subtract -2 times row from row 3. We will make another Elimination matrix, $E2$ to perform this operation. But let's now make it row by row so we understand how it works.

So in the row1 resultant matrix, we do not want any change in the matrix.

hence row1 of elimination matrix = $\begin{bmatrix} 1 & 0 & 0\end{bmatrix}$ i.e we want 1 of the first row and none of any other row, giving us the original row back.

In row2 of resultant matrix we again do not want any change,

so row2 of elimination matrix = $\begin{bmatrix} 0 & 1 & 0\end{bmatrix}$ i.e we want 1 of second row and none of any other row.

Now in row3 we want to eliminate 10 using pivot -5, so we want to subtract -2 times the row 2 from row 3 which is also adding 2 times row 2 to row 3:

so row3 of elimination matrix = $\begin{bmatrix} 0 & 2 & 1\end{bmatrix}$ i.e we want twice of row2 and row 1, which will eliminate the required numbers.

So the final elimination matrix will be:


$$ E_2 = \begin{bmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 2 & 1 \end{bmatrix}$$ 


```julia
E2 = [1 0 0; 0 1 0; 0 2 1];
```

So let's now perform $E_2(E_1A)$ and $E_2(E_1\mathbf{b})$:

$$ E_2(E_1A) =\begin{bmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 2 & 1 \end{bmatrix}\ \begin{bmatrix} 2 & 4 & -6 \\ 0 & -5 & 4 \\ 0 & 10 & 1\end{bmatrix} = \begin{bmatrix} 2 & 4 & -6 \\ 0 & -5 & 4 \\ 0 & 0 & 9 \end{bmatrix}$$ 

$$ E_2(E_1\mathbf{b}) = \begin{bmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 2 & 1 \end{bmatrix}\ \begin{bmatrix} -2 \\ 4\\ 1\end{bmatrix} = \begin{bmatrix} -2 \\ 4 \\ 9 \end{bmatrix}$$ 


```julia
E2*(E1*A)
```




    3×3 Array{Int64,2}:
     2   4  -6
     0  -5   4
     0   0   9




```julia
E2*(E1*b)
```




    3-element Array{Int64,1}:
     -2
      4
      9



So now the equation is:

$$ E_2(E_1A)\mathbf{x} = \begin{bmatrix} 2 & 4 & -6 \\ 0 & -5 & 4 \\ 0 & 0 & 9 \end{bmatrix} \ \begin{bmatrix} x \\ y \\ z \end{bmatrix} = \begin{bmatrix} -2 \\ 4 \\ 9 \end{bmatrix} = E_2(E_1\mathbf{b}) $$ 

Here we have reached to the upper triangular form and hence have completed the forward pass. the backward pass is simple.

So we apply elimination matrices to change $A \rightarrow U$, where $U$ is an upper traingular matrix. We apply the same matrices to change $\mathbf{b} \rightarrow \mathbf{b}'$ and finally solve $U\mathbf{x} = \mathbf{b'}$ using back-substitution(if there is a unique solution).

One more thing to mention about matrix multiplication is that it holds on associativity. So, if $P, Q, R$ are three matrices(compatible for matrix multiplication in the order), then:

$$ P(QR) = (PQ)R$$ 

So basically we can multiply all our elimination matrices first, and then multiply that with our $A$ and $\mathbf{b}$.

$$E = E_nE_{n-1}\cdots E_2E_1$$  

Let's see how this matrix $E$ looks like in our example.

$$ E = E_2E_1  = \begin{bmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 2 & 1 \end{bmatrix} \ \begin{bmatrix}1 & 0 & 0 \\ -2 & 1 & 0 \\ 1 & 0 & 1 \end{bmatrix} = \begin{bmatrix} 1 & 0 & 0 \\ -2 & 1 & 0 \\ \mathbf{-3} & 2 & 1 \end{bmatrix}$$ 


So we multiply by this elimination matrix to our matrix $A$ to form an Upper triangular matrix, $U$, 

$$ E_2(E_1A) = (E_2E_1)A = EA = U$$ 

and

$$ E_2(E_1\mathbf{b}) = (E_2E_1)\mathbf{b} = E\mathbf{b} =\mathbf{b'} $$ 


```julia
E = E2*E1
```




    3×3 Array{Int64,2}:
      1  0  0
     -2  1  0
     -3  2  1



Now the matrix $E$ has almost all the same multipliers that we use in individual except a few(shown in bold-face of matrix $E$). It is because the effect of the first elimination is shown in second and so on. In our example the second row is subtracted by **2** times the first row and the third row is  added by **1** times first row. Then the _new_ third row is added **2** times the _new_ second row. So essentially, the _original_ third row was subtracted **3** times the first row and **2** times the original second row.


However to show it in a better way, we can use the inverse of $E$,

$EA = U$ can be shown as $A = E^{-1}U$


Let's look at $E^{-1}$

$$ E^{-1} = \begin{bmatrix} 1 & 0 & 0 \\ 2 & 1 & 0 \\ -1 & -2 & 1 \end{bmatrix}$$ 


```julia
L=inv(E)
```




    3×3 Array{Float64,2}:
      1.0   0.0  0.0
      2.0   1.0  0.0
     -1.0  -2.0  1.0



The $E^{-1}$ has the correct values that mutiply the pivots, before **subtracting** them from the lower rows going from $A$ to $U$. SInce it is an lower traingular matrix, we represent it by $L$. 

$$ A = LU$$ 

The matrix $L$ can be used at many places.

1. It can be used to factorize $A$.

2. It has the memory of pivot multipliers before subtraction.

3. \*It can be used to compute transformed right hand side,$\mathbf{b'}$, as $\mathbf{b} = L\mathbf{b'}$.

\*Although we can also concatenate $A$ and $\mathbf{b}$ as $\begin{bmatrix} A & \mathbf{b} \end{bmatrix}$ and then run it though the forward elimination but most applications tend to do them separately.

#### Row Exchanges

Here we had prepared everything but considering no row exchange will be needed. Let's remind that row exchanges are needed when zeros occur at pivot places.

What matrix to use when we have to exchange two rows? Let's take an example:
end{bmatrix}

$$ A = \begin{bmatrix}\mathbf{1} & 3 & 5 \\ 3 & 9 & 7 \\ 4 & 8 & 9\end{bmatrix} $$ 

Let's first eliminate the first column using the first pivot(bold-face),

$$ E_1A = \begin{bmatrix}1 & 0 & 0 \\ -3 & 1 & 0 \\ -4 & 0 & 1 \end{bmatrix} \ \begin{bmatrix}\mathbf{1} & 3 & 5 \\ 3 & 9 & 7 \\ 4 & 8 & 9\end{bmatrix} = \begin{bmatrix} 1 & 3 & 5 \\ 0 & 0 & -8 \\ 0 & -4 & -11 \end{bmatrix}$$ 

As you can see the second diagonal element is zero, which cannot be a pivot. So we will exchange the second and third row. the matrix we will apply is called a _permutation matrix_. 


$$ P_{23} = \begin{bmatrix} 1 & 0 & 0 \\ 0 & 0 & 1 \\ 0 & 1 & 0 \end{bmatrix}$$ 

So the first resultant row will be the same. The resultant second row will be zero times first and second row and 1 times third row ,i.e the third row. The resultant third row will be zero times first and third row but 1 times the second, i.e the second row. So, essentially the second and the third row have been exchanged. A permutation matrix is basically identity matrix with the same rows exchanged which need to be changed in the coefficient matrix.

$$ P_{23}(E_1A) = \begin{bmatrix} 1 & 0 & 0 \\ 0 & 0 & 1 \\ 0 & 1 & 0 \end{bmatrix} \ \begin{bmatrix} 1 & 3 & 5 \\ 0 & 0 & -8 \\ 0 & -4 & -11 \end{bmatrix} = \begin{bmatrix} 1 & 3 & 5 \\ 0 & \mathbf{-4} & -11 \\ 0 & 0 & -8  \end{bmatrix}$$ 

Now we have the new pivot and we can move with elimination. 

>**Note:** Row exchanges will alter the final the elimination matrix, $E$ as well as the inverse $L$ which will not have the values at the same place.  

Now one way to keep it in the same way is if we perform all the row exchanges first, using the product of all permutation matrices, $P$ and then perform the elimination using $E$.

Final equation is:

$$ PA = LU$$ 

This is a good stopping point in basics. We will next see what does the infinite solutions mean, how we represent them. What are vector spaces and subspaces, rank, invertibility and more.

{% include linalg/1.basics/plot1.html %}


{% include linalg/1.basics/plot2.html %}


{% include linalg/1.basics/plot3.html %}
