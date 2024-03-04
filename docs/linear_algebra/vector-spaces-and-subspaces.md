---
title: Introduction to Vector Spaces and Sub Spaces, Rank and Invertibility
description: This post provides an introduction to the vector sapces and subspaces. It goes on to define column space and null space. With that it defines rank, and inveribility.
tags:
  - Vector
  - Space
  - Column
  - Space
  - "Null"
  - Space
  - Rank
  - Reduced
  - Roe
  - Echelon
  - Form
  - Invetibility
---

# Introduction to Vector Spaces and Sub Spaces, Rank and Invertibility

### Introduction

This article will dive deep into **Vector Spaces, Subspaces, Column Spaces, Null Space, Span, Rank, Invertibility** and much more.  

### Vector Spaces

A vector space is a collection of vectors where each vector can be defined as a linear combination of all other vectors. It can also be defined as "if $n$ vectors are in a space, then all the linear combinations of all these vectors are also in that space."

e.g: for 2-D vectors, the $xy$ plane is a space. It contains all the 2-D vectors. All of these vectors can be represented as a linear combination of all other vectors(some scalar multipliers might be zero).

The above space is denoted by $\mathbb{R}^2$ and is called the real space in two dimensions(or simply the **two dimensional space**). It the biggest space possible for two dimensional _real_ vectors. If we include complex numbers as well, then that space will be $\mathbb{C}^2$.

Similarly, $\mathbb{R}^1$, $\mathbb{R}^3$, $\mathbb{R}^4$, $\mathbb{R}^n$ are the real spaces in their respective dimensions. These contain all the real respective-dimensional vectors.

So if, for example, we have two vectors 

$$\mathbf{v_1}, \mathbf{v_2} \in \mathbb{R}^5$$

then 

$$c\mathbf{v_1} + d\mathbf{v_2} \in \mathbb{R}^5 \qquad \forall\  c, d \in \mathbb{R}\text{(i.e scalars)}$$

But any vector in $\mathbb{R}^2$ is not in the $\mathbb{R}^5$ as no linear combination of 5-dimensional vectors can form a 2-dimesional vector.

Based on the that every space must have a _zero vector_ (when all the scalar multipliers in a linear combination are zero).

We can even extend the concept of spaces from traditional "column vectors" to other "vectors". Basically, we need a multiplication by a scalar with a "vector" keep the resulting "vector" in that same space. Also sum of two "vectors" in a space should also be in the same space. 

> **Note:** A space may not necessarily contain column vectors. In the definition of *vector space*, vector addition, $v_1 + v_2$, and scalar mutiplication, $cv$, should follow the following eight rules:
1. $v_1 + v_2 = v_2 + v_1$
2. $v_1 + (v_2 + v_3) = (v_1 + v_2) + v_3$
3. There exists a zero vector $0$ such that $v+0 = v$.
4. There is a unique vector $-v$, for each vector $v$, such that $v+(-v)=0$.
5. Multiplication of vector $v$ by unity, 1 ,scalar, does not change the vector,i.e $1v = v$
6. $(c_1c_2)v = c_1(c_2v)$
7. $c(v_1 + v_2) = cv_1 + cv_2$
8. $(c_1 + c_2)v = c_1v + c_2v$

Any "vectors" following those eight rules can have a concept of spaces in them. For example: matrices. Matrices(of same dimensions) follow all the above rues and hence concept of spaces is possible. 

All real $3 \times 4$ matrices form a space. Any two such shaped matrices can be added to give the same shaped matrix. Any such shaped matrix can be multipied(with a scalar, ofcourse!) to give the same shaped matrix again. It also incudes the zero matrix. This space can be called $\mathbb{R}^{3 \times 4}$

We can also consider all real functions. They will also form a space. Also has zero function. A function space is infinite-dimensional.

In both above examples the eight conditions are also easily checked.

We saw multiple dimensional spaces but there is another special dimensional space, the zero dimensional space $\mathbb{Z}$. By the simple definition if it has zero dimensions, it means it has no components and so one might think there is no vector. But it only contains exactly one vector.

So at many times we can think matrices and functions as vectors but mostly we refer to vectors when we mean column vectors and with that let's talk about subspaces.

### Subspaces

A subspace is a space inside another space. This subspace is a part of the whole space as well as satisfies all the criteria to be called a space itself.

For example, in three dimensional vector space, any plane passing through origin is a subspace of $\mathbb{R}^3$. 

Every vector in that plane is a linear combination of other vectors in the same plane. It is also passing via origin and so has the zero vector in it. One thing to note is that a plane in three-dimensional space is **not** $\mathbb{R}^2$. The vectors are in $\mathbb{R}^3$ as they have three components, they just lie on a plane.

Also note that any plane that does not pass through the origin is not a subspace as it does not account for the combination when all the multipliers of a linear combination are zero. 

Apart from all the planes passing through origin, the whole $\mathbb{R}^3$ is also a subspace of itself!. The list of all subspaces of $\mathbb{R}^3$ is:
1. The whole space itself.
2. Any plane passing through origin.
3. Any line passing through origin.
4. The Zero Vector.

Just like $\mathbb{R}^3$, $\mathbb{R}^2$ also has its subspaces: any line passing through origin,for example.

We also talked about matrices forming spaces. So they can also form subspaces. For example in the vector space of all $4 \times 4$ real matrices i.e $\mathbb{R}^{4 \times 4}$,a subspace will be all the $4 \times 4$ diagonal real matrices as all linear combinations of these matrices will also reult in a $4 \times 4$ real diagonal matrix. We can also have a subspace of traiangular matrices. Further more, diagonal matrices are the subspaces of triangular matrix space(of the same dimensions!).

Also, all these spaces also have the Zero matrix in them, which is also an important check for a space.

### Column Space of $A$: $C(A)$

A column space is related to a matrix. A column space of a matrix $A$ is, the space containing _only_ all linear combinations of columns of a matrix. It is represented as $C\left(A\right)$ It is explained in the context of solving the equation: 

$$A\mathbf{x} = \mathbf{b}$$

For all the possible $\mathbf{x}$, $A\mathbf{x}$ is linear combinations of columns of $A$.

e.g: If

$$A = \begin{bmatrix}2 & 3  \\ 6 & 1 \\ -1 & 8 \end{bmatrix}$$

then,

$$A\mathbf{x} = \begin{bmatrix}2 & 3 \\ 6 & 1 \\ -1 & 8 \end{bmatrix} \ \begin{bmatrix} x_1 \\ x_2 \end{bmatrix}$$

By our definition of matrix vector multiplication, in the previous chapter:

$$A\mathbf{x} = x_1 \begin{bmatrix}2 \\6 \\ -1 \end{bmatrix} + x_2 \begin{bmatrix}3 \\1 \\ 8 \end{bmatrix} $$



So for all $\mathbf{x}$, $A\mathbf{x}$ is all the linear combinations of columns of $A$. **So $A\mathbf{x}$ represents the whole column space, for all $\mathbf{x}$**. For a specific $\mathbf{x}$, say $\mathbf{x_1}$ we have a specific vector $A\mathbf{x_1}$ which lies in the column space of $A$.

**So for $A\mathbf{x} = \mathbf{b}$ to have any solution, $\mathbf{b}$ has to be in the column space of $A$.** Otherwise there is no solution. If $\mathbf{b}$ is in $C\left(A\right)$, then the vector of multipliers of that combination is the solution of this equation.

For any matrix of shape $m$ by $n$, each column has $m$ components, so the column space of that matrix is a subspace of $\mathbb{R}^m$.

For our example above the column space will form a plane in $\mathbb{R}^3$ passing via origin.


```julia
using Plotly
```



```julia
# using multiple values for x
x= [[i;j] for i=-0.7:0.01:0.7,j=-0.7:0.01:0.7]

# create matrix A
A = [2 3; 6 1; -1 8]
# calculate the linear combination for each vector (x_1,x_2)
linear_combos = [A*xx for xx in x]

#fetch the x, y and z cordinates from each vector
x_s = [xx[1] for xx in linear_combos]
y_s = [xx[2] for xx in linear_combos]
z_s = [xx[3] for xx in linear_combos]

# plot column 1
trace1 = scatter3d(x=[0, A[1,1]], y=[0, A[2,1]], z=[0, A[3,1]], mode="lines", line=attr(width=5), name="vector 1")
#plot column 2
trace2 = scatter3d(x=[0, A[1,2]], y=[0, A[2,2]], z=[0,A[3,2]], mode="lines", line=attr(width=5), name="vector 2")
# plot the combinations
trace3 = surface(x=x_s, y=y_s, z=z_s, opacity=0.8, showscale=false, name="Plane")

p = plot([trace1,trace2,trace3], Layout(scene_camera=attr(eye=attr(x=1.5, y=1.2, z=1)), title="Column Space of a matrix"))
```

<div id="plot1">
</div>



The plane represents all the linear combinations of these two vectors lie on that plane which also passes through the origin. It is the column space of the matrix. It is a subspace of the whole $\mathbb{R}^3$ space. For all the vectors $\mathbf{b}$ lying in that plane, $A\mathbf{x} = \mathbf{b}$ has solution. But for those $\mathbf{b}$ which do not lie in that plane (i.e are not in the column space), have no solution.

> **Span:** 
Now instead of just columns we can have a set of vectors $V$ and their linear combinations form a space $VV$. This span is the **smallest** space that contains all the linear combinations of these vectors. This can also be said as *The subspace $VV$ is the __span__ of set $V$.* So the columns "span" the column space.

So, as per above observations, it entirely depends on the vector $\mathbf{b}$ to have a solution to $A\mathbf{x}=\mathbf{b}$. If the vector $\mathbf{b}$ is in the column space of $A$ or not. But there is one vector for $\mathbf{b}$ we would always have solution(s) for. The only vector(of the same dimension) which lies in all subspaces. The zero vector.

That means, $A\mathbf{x} = 0$ will always have solution(s). ($0$ here is a zero vector, not a scalar.)

This gives introduction to another important subspace called the **Null Space**. 

### Null Space of $A$: $N(A)$

We will now deal with special equations where the right hand side is always a zero vector, i.e $\mathbf{b} =0$.

Let's look at the possible solutions of $A\mathbf{x} = 0$:

1. $\mathbf{x} = 0$ is always a solution to it.
2. If $\mathbf{x_1}$ is a solution, then so is $c_1\mathbf{x_1}$, $c_1$ being a scalar.


If

$$A\mathbf{x_1} = 0$$

then

$$Ac_1\mathbf{x_1} = c_1A\mathbf{x_1} = 0$$

3. If $\mathbf{x_1}$ and $\mathbf{x_2}$ are two solutions, then $x_1 + x_2$ is also a solution. It can be shown as:

If $\mathbf{x_1}$ and $\mathbf{x_2}$ are solutions, then:

$$A\mathbf{x_1} = 0$$

and

$$A\mathbf{x_2} = 0$$

and so,

$$\begin{align}A\mathbf{x_1} + A\mathbf{x_2} = 0 \\ \implies A \left(\mathbf{x_1} + \mathbf{x_2}\right) = 0\end{align}$$

so $\mathbf{x_1} + \mathbf{x_2}$ is also a solution.

4. By the above two points we can show that _all_ the linear combinations of the solutions are also solutions to $A\mathbf{x} = 0$.

So we can conclude the solutions of $A\mathbf{x} = 0$ form a subspace. This subspace is called the Null Space of matrix $A$. It is denoted by $N(A)$.

If the matrix $A$ is $m$ by $n$, then the null space, $N(A)$ is a subspace of $\mathbb{R}^n$ (while the column space, $C(A)$, was in $\mathbb{R}^m$).

#### Finding the Null Space

The column space is simply the space of all linear combinations of columns of $A$ and the columns are known. But for null space, we need to find the solutions.

We know that the solutions were found by elimination. We will do the same, but with some special extensions. We have used elimination on the square matrices (i.e $m$ equations $m$ variables, but let's generalize it to rectangluar matrices).

Let's say we have two equations and two variables:

$$\begin{align}x_1 + 7x_2 &= 0 \\ 2x_1 + 14x_2 &=0\end{align}$$

If we eliminate, we get:

$$\begin{align}x_1 + &7 x_2 = 0 \\ &0x_2 = 0\end{align}$$

Here $x_2$ is called the **free variable**. The above system of equations is equivalent to just one of the above eqautions(both are same!). 

We can set the free variable to anything and get corresponding $x_1$ for that solution. Preferably, we set $x_2=1$, and so $x_1 = -7$.

This solution $(-7,1)$ is a specific solution to the equation above. The nullspace is all the linear combinations of this solution. Since we are having linear combinations of just one vector, it will result in a _line_ through origin. That line represents the nullspace of the equation system. 

So, $c\begin{bmatrix}-7 \\ 1\end{bmatrix}$ is the nullspace of this matrix for all scalars, $c$.

**Another Example**

Let's say we have two equations and four variables, e.g:

$$\begin{align}3x_1 + 2x_2 + 4x_3 - 3x_4 &=0 \\ 6x_1 + 3x_2 - x_3 + 4x_4 &= 0\end{align}$$

It can be written in matrix form as, $A\mathbf{x} = 0$, where

$$A = \begin{bmatrix} \mathbf{3} & 2 & 4 & -3 \\ 6 & 3 & -1 & 4 \end{bmatrix}$$

and

$$\mathbf{x} = \begin{bmatrix} x_1 \\ x_2 \\ x_3\\ x_4 \end{bmatrix}$$

So, essentially we are trying to find the nullspace of $A$. We will proceed with elimination, but note that we need not to perform the same elimination on Right Hand Side as it is a Zero vector and it won't change.

Eliminate using the first pivot (bold-face in matrix above):

$$E_1A = \begin{bmatrix} 3 & 2 & 4 & -3 \\ 0 & \mathbf{-1} & -9 & 10 \end{bmatrix}$$


And now we are done. This is the closest to "upper traingular" system we can get. The Pivots are: 3, -1. The two columns, which have pivots are called **pivot columns**. The rest of the columns are called **free columns**.


$$\begin{align}E_1A = U =  &\begin{bmatrix} 3 & 2 & 4 & -3 \\ 0 & -1 & -9 & 10 \end{bmatrix}\\
& \underbrace{\uparrow \hspace{28px} \uparrow}_\text{pivot columns} \hspace{10px} \underbrace{\uparrow \hspace{28px} \uparrow}_\text{free columns}\end{align}$$


The variables corresponding to the free columns are the free variables, in this case, $x_3$ and $x_4$. Now to get the nullspace, we will first need the specific solutions and then the nullspace will formed by all the linear combinations of those specific solutions. 

For each specific solution, we will set each free variable equal to 1 and all other free varaibles to zero. Since this has two free columns, which means two free variables, we have two specific solutions.

One will have $x_3=1$,$x_4=0$ and the other will have $x_3=0$, $x_4=1$. 

For specific solution 1,

Use $x_3=1,x_4=0$ in the second row and we get $x_2=-9$. and then backsubstituting $x_2,x_3,x_4$ into row 1, we get $x_1 = \frac{14}{3}$ 

Our first specific solution is 

$$s_1 = \begin{bmatrix} \frac{14}{3} \\ -9 \\ 1 \\0\end{bmatrix}$$

For second specific solution, we will keep $x_3 = 0$ and $x_4 = 1$.

Back substituting,

$x_2 = 10$ and $x_1 = \frac{-17}{3}$

Second specific solution is 

$$s_2 = \begin{bmatrix} \frac{-17}{3} \\ 10 \\ 0 \\1\end{bmatrix}$$

The nullspace contains all the linear combinations of $s_1$ and $s_2$. If we put these two specific solutions in a matrix.

$$ S = \begin{bmatrix} \frac{14}{3} & \frac{-17}{3} \\ -9 & 10 \\ 1 & 0 \\0 & 1 \end{bmatrix}$$

We can also say the nullspace of $A$ was the columnspace of $S$.

$$ N(A) = C(S)$$ 

#### Reduced Row Echelon Form

We most often take a step further from converting the matrix $A$ to upper triangular $U$ during elimination. We convert it into, what's called, a *reduced row echelon form*, $R$. It gives us a better view at the nullspace of $A$ which is same as $U$ and as we will see for $R$ as well.

In this form we do not stop at the upper triangular $U$ form but we remove the elements above the pivots as well. And we also divide each pivot row to make its pivot unity.



In our above example, we had reduced till upper traingular:

$$ U =  \begin{bmatrix} \mathbf{3} & 2 & 4 & -3 \\ 0 & \mathbf{-1} & -9 & 10 \end{bmatrix}$$

Our pivots (in bold-face) will eliminate all the elements above them as well. The first pivot has not to change anything. But we can remove the elements above the second pivot.

$$\begin{align}\text{row}_1 \leftarrow \text{row}_1 + 2\text{row}_2\\ E_2U =  \begin{bmatrix} \mathbf{3} & 0 & -14 & 17 \\ 0 & \mathbf{-1} & -9 & 10 \end{bmatrix}\end{align}$$

Now we have removed the entries, we have to make our pivots unity.

$$\begin{align}\text{row}_1 \leftarrow \frac{\text{row}_1}{3}\text{ ; }\text{row}_2 \leftarrow \frac{\text{row}_2}{-1}\\ E_3U = R =  \begin{bmatrix} \mathbf{1} & 0 & \frac{-14}{3} & \frac{17}{3} \\ 0 & \mathbf{1} & 9 & -10 \end{bmatrix}\end{align}$$

Here we have reached the reduced row echelon form. The pivot columns form a sub identity matrix. But here is the good part. If we look at our two special solutions,$s_1$ and $s_2$, the free columns seem to have half of the solution. Well actually the negative of the free columns.

If we show our reduced matrix in the form of this identity and free columns, i.e:

$$R = \begin{bmatrix} I & F \end{bmatrix}$$

Then our special solution matrix is:

$$S = \begin{bmatrix} -F \\I \end{bmatrix}$$

Let's look at one more example, in which I will try to encapsulate all cases this form can have. Let's have a 3 by 5 matrix and find its nullspace, i.e three equations and 5 variables.




$$ A = \begin{bmatrix} \mathbf{3} & 4 & 2 & -3 & 8 \\ 6 & 8 & 3 & 4 & 2 \\ 12 & 16 & 7 & -2 & 18 \end{bmatrix}$$


Now to find it's null space, we will find it's special solutions for $A\mathbf{x} = 0$. For that we will use elimination to convert it into $U$ form. Eliminating using the first pivot (bold-face).


$$\begin{align}\text{row}_2 \leftarrow \text{row}_2 - 2\text{row}_1\\ \text{row}_3 \leftarrow \text{row}_3 - 4\text{row}_1\\ E_1A = \begin{bmatrix} 3 & 4 & 2 & -3 & 8 \\ 0 & 0 & -1 & 10 & -14 \\ 0 & 0 & -1 & 10 & -14  \end{bmatrix}\end{align}$$


Now the number at second pivot has become ZERO, which calls for a row exchange, but every element below it is also zero and hence no row to exchange with. So, elimination done? No. This whole column is actually a **free column** and not a **pivot column**. The next column to it is the pivot column and $-1$ (bold-face, below) is the second pivot. 

$$E_1A = \begin{bmatrix} 3 & 4 & 2 & -3 & 8 \\ 0 & 0 & \mathbf{-1} & 10 & -14 \\ 0 & 0 & -1 & 10 & -14\end{bmatrix}$$


Now going on with the elimination with this pivot, we get:

$$\begin{align}\text{row}_3\leftarrow \text{row}_3-\text{row}_2\\E_2E_1A = U = \begin{bmatrix} 3 & 4 & 2 & -3 & 8 \\ 0 & 0 & -1 & 10 & -14 \\ 0 & 0 & 0 & 0 & 0  \end{bmatrix}\end{align}$$


Now we have finished the elimination and reached the $U$ phase.

> **Note: Each column and row can have at most one pivot.**

At this point, usually we would check if our right hand side (which was $\mathbf{b}$) had zero at the last entry or not $-$ to check for no solution. But in null space, all the RHS are zero, so we don't need to worry.

So the pivot columns are column 1 and column 3, and free columns are column 2, column 4 and column 5, which means $x_2$, $x_4$ and $x_5$ are free variables. That means we have three specific solutions, one when $(x_2, x_4, x_5)=(1,0,0)$, one when $(x_2, x_4, x_5)=(0,1,0)$ and another when $(x_2,x_4,x_5)=(0,0,1)$.  

Back substitute to find the solution of:

$$ U\mathbf{x} = \begin{bmatrix} 3 & 4 & 2 & -3 & 8 \\ 0 & 0 & -1 & 10 & -14 \\ 0 & 0 & 0 & 0 & 0  \end{bmatrix}\ \begin{bmatrix}x_1\\x_2\\x_3\\x_4\\x_5\end{bmatrix} = \begin{bmatrix}0 \\ 0 \\ 0 \\ 0\\ 0\end{bmatrix}$$

Back-substituting $x_2=1$, $x_4=0$ and $x_5=0$ to compute $s_1$:

1. The last equation(row) contributes nothing.

2. The second equation(row) gives $x_3 = 0$

3. And the first equation gives $x_1 = \frac{-4}{3}$

and so our first specific solution is:

$$s_1 = \begin{bmatrix} \frac{-4}{3} \\ 1 \\0 \\ 0\\0\end{bmatrix}$$


Back-substituting $x_2=0$, $x_4=1$ and $x_5=0$ to compute $s_2$:

1. The last equation(row) contributes nothing.

2. The second equation(row) gives $x_3 = 10$

3. And the first equation gives $x_1 = \frac{-17}{3}$

and so our first specific solution is:

$$s_2 = \begin{bmatrix} \frac{-17}{3} \\ 0 \\10 \\ 1 \\ 0\end{bmatrix}$$


Back-substituting $x_2=0$, $x_4=0$ and $x_5=1$ to compute $s_3$:

1. The last equation(row) contributes nothing.

2. The second equation(row) gives $x_3 = -14$

3. And the first equation gives $x_1 = \frac{20}{3}$

and so our first specific solution is:

$$s_3 = \begin{bmatrix} \frac{20}{3} \\ 0 \\-14 \\ 0 \\ 1\end{bmatrix}$$


And so the null space is all the linear combinations of these three vectors. The null space will be 3-D surface inside a 4-Dimensional space. We can also group these solutions into a matrix $S$, and so $N(A) = C(S)$.

$$S = \begin{bmatrix} \frac{-4}{3}& \frac{-17}{3} & \frac{20}{3}  \\ 1 & 0 & 0 \\0 & 10 & -14 \\ 0 & 1& 0 \\ 0 & 0 & 1\end{bmatrix}$$

Keeping this $S$ in mind, we move on to the Row Echelon form of A. Let's check our $U$ highlighting its pivots.

$$U=\begin{bmatrix} \mathbf{3} & 4 & 2 & -3 & 8 \\ 0 & 0 & \mathbf{-1} & 10 & -14 \\ 0 & 0 & 0 & 0 & 0  \end{bmatrix}$$

Reduce the elements above the pivots as well and make pivots unity.

$$\begin{align}\text{row}_1 \leftarrow \text{row}_1+2\text{row}_2\\ E_3U = \begin{bmatrix} \mathbf{3} & 4 & 0 & 17 & -20 \\ 0 & 0 & \mathbf{-1} & 10 & -14 \\ 0 & 0 & 0 & 0 & 0  \end{bmatrix}\end{align}$$

$$\begin{align}\text{row}_1 \leftarrow \frac{\text{row}_1}{3}\\ \text{row}_2 = -1(\text{row}_2)\\ E_4E_3U = R = \begin{bmatrix} \mathbf{1} & \frac{4}{3} & 0 & \frac{17}{3} & \frac{-20}{3} \\ 0 & 0 & \mathbf{1} & -10 & 14 \\ 0 & 0 & 0 & 0 & 0  \end{bmatrix}\end{align}$$



All pivots are Zero and no element above or below the pivots are non-zero. By definition, we have reached the reduced row echelon form.

If we look at the matrix $S$, there are negative of elements of free columns of $R$ in matrix $S$. 

Let me tell you how it relates, if our reduced row echelon form to looks like:

$$ R = \begin{bmatrix} I & F \\ 0 & 0 \end{bmatrix}$$

where:

$F$ are the free columns.

$\begin{bmatrix}0 & 0 \end{bmatrix}$ are all the zero rows until the end of the matrix.

then our null solution matrix is:

$$S = \begin{bmatrix} -F \\ I' \end{bmatrix} $$

where $I'$ is the identity matrix of the same size as the number of free columns in $R$.

But our current example isn't looking like the form we want, for that we can do **column exchanges**, but that changes the system and so we need to change the variable vector as well. Let's first write our full equation:

$$R\mathbf{x} = 0$$

in our example,

$$\begin{bmatrix} \mathbf{1} & \frac{4}{3} & 0 & \frac{17}{3} & \frac{-20}{3} \\ 0 & 0 & \mathbf{1} & -10 & 14 \\ 0 & 0 & 0 & 0 & 0  \end{bmatrix} \ \begin{bmatrix} \pmb{x_1} \\ x_2 \\ \pmb{x_3} \\ x_4 \\x_5 \end{bmatrix} = 0$$


If we want to exchange our 2nd and 3rd column of $R$, we need to exchange our 2nd and 3rd variable as well to preserve the multiplication.

$$\begin{bmatrix} \mathbf{1} & 0 & \frac{4}{3} & \frac{17}{3} & \frac{-20}{3} \\ 0  & \mathbf{1} & 0 & -10 & 14 \\ 0 & 0 & 0 & 0 & 0  \end{bmatrix} \ \begin{bmatrix} \pmb{x_1} \\ \pmb{x_3} \\ x_2 \\ x_4 \\x_5 \end{bmatrix} = 0$$

Notice I have exchanged $x_2$ and $x_3$ as well. Now we have the form 

$$\begin{bmatrix} I & F \\ 0 & 0 \end{bmatrix}$$

So our null matrix should be 

$$\begin{bmatrix} -F \\ I' \end{bmatrix}$$

$$-F = \begin{bmatrix} \frac{-4}{3} & \frac{-17}{3} & \frac{20}{3} \\ 0 & 10 & -14\end{bmatrix}$$

and since there are 3 free columns $I'$ is a 3 $\times$ 3 identity matrix.

$$ I' = \begin{bmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{bmatrix}$$

And so our null matrix is:

$$ \begin{bmatrix} -F \\ I' \end{bmatrix}  = \begin{bmatrix} \frac{-4}{3} & \frac{-17}{3} & \frac{20}{3} \\ 0 & 10 & -14 \\ 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{bmatrix} $$

and we are done? No. A small change needs to be made. This null matrix corresponds to the changed variable vector, i.e in each column the first entry corresponds to $x_1$, **second entry to $x_3$ (not $x_2$)**, **third entry to $x_2$ (not $x_3$)**, fourth entry to $x_4$ and fifth entry to $x_5$. This change was because of the column exchange we did back there.

So to get back the correct form of the null matrix we exchange the row2 and row 3 of the above matrix, and so:

$$S = \begin{bmatrix} \frac{-4}{3} & \frac{-17}{3} & \frac{20}{3} \\  1 & 0 & 0 \\ 0 & 10 & -14 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{bmatrix}$$

And now we have finished. These three columns are the special solutions we have to find the whole null space.

However it is not necessary, to turn $R$ into the said form and do column exchanges, the $R$ form itself makes the back substitution very easy.

But if we use the $\begin{bmatrix} I & F \\ 0 & 0 \end{bmatrix}$, we must know that sometimes there may not be a bunch of zero rows and so our form might be $\begin{bmatrix} I & F \end{bmatrix}$ which will again have the same solution. The zero rows do not contribute to the nullspace. However if the RHS were not a zero vector, these rows would decide the possibility of no solution.

If our matrix $A$ is of size $m \times n$ and,

1. If $n > m$, i.e more variables than equations(like in our previuos example), we will **atleast** have $n-m$ free columns (because each row and column can at most have one pivot), so there will be atleast $n-m$ special solutions, which means we will have infinite "*non-zero*" solutions, which will form a nullspace. These will have a form of  $R = \begin{bmatrix} I & F \\ 0 & 0 \end{bmatrix}$ or $\begin{bmatrix} I & F \end{bmatrix}$.

2. If $m > n$, i.e more equations than variables, we will atleast have $m-n$ rows of zeros at the bottom. Since the columns are less than rows, all the columns can be pivots, so it may or may not have any $F$ column. The possible scenarios are $\begin{bmatrix} I & F \\ 0 & 0 \end{bmatrix}$ and $\begin{bmatrix} I\\ 0 \end{bmatrix}$. In the first case there are special solutions and hence infinite solutions. In the second case, since there is no free column, there is only one vector in nullspace, the zero vector, i.e there is no non-zero null vector.

3. If $m=n$, i.e equal number of equations and variables. It has many possibilities. It may or may not have $F$ and even it may or may not have zero rows. So possibilities are $\begin{bmatrix} I & F \\ 0 & 0 \end{bmatrix}$, $\begin{bmatrix} I \\ 0 \end{bmatrix}$ , $\begin{bmatrix} I & F \end{bmatrix}$ and even $\begin{bmatrix} I\end{bmatrix}$. The last case also has just one null vector, the zero vector. If a matrix is able to reduce to the last case, it is said to be *invertible*. More on it later.

I highly recommend you to work on some problems yourself and try to get the feel for this form.

### The Rank of a Matrix 

Now we are introducing an important concept for our matrices, the rank of the matrix.

Our matrix defines a linear system. If a matrix is $m \times n$, that is supposed to be it's size. But is it it's *true* size? Well, what do I mean by true size? 

Let's dig into some basic questions. What did the pivot columns  signify? Why does elimination cause some rows to be Zero? What do free columns signify?

For that let's revisit our example of reducing a matrix to reduced row echelon form and finding the pivots.

Our initial matrix was:

$$ A = \begin{bmatrix} 3 & 4 & 2 & -3 & 8 \\ 6 & 8 & 3 & 4 & 2 \\ 12 & 16 & 7 & -2 & 18 \end{bmatrix}$$

and the reduced row echelon form of $A$ was:

$$R = \begin{bmatrix} \mathbf{1} & \frac{4}{3} & 0 & \frac{17}{3} & \frac{-20}{3} \\ 0 & 0 & \mathbf{1} & -10 & 14 \\ 0 & 0 & 0 & 0 & 0  \end{bmatrix}$$


The size of $A$ is $3 \times 5$.

The last row turned to be zero. It is because it did not add anything new to the system. It was a linear combination of the rows above it. Actually,

$$\text{row}_3 \text{ of }A = 2(\text{row}_1 \text{ of }A) + \text{row}_2 \text{ of }A$$

So the true size of this matrix from the row size is actually 2 not 3.

Now let's look at pivot columns and free columns. Why was the second column declared as a free column? Because it did not have a pivot. But what does not having a pivot mean? It means this column is a linear combination of all the pivot columns before it. 

Every free column is a linear combination of the pivot columns before them. The special solutions tell us these combinations.

$$\begin{align}\text{col}_2 \text{ of }A &= \frac{4}{3}\left(\text{col}_1 \text{ of }A\right) ; & s_1 = \left(\frac{-4}{3}, 1 , 0, 0, 0\right) \\ \text{col}_4 \text{ of }A &= \frac{17}{3}\left(\text{col}_1 \text{ of }A\right) + (-10)\left(\text{col}_3 \text{ of }A\right) ; & s_2 = \left(\frac{-17}{3}, 0 , 10, 1, 0\right) \\ \text{col}_5 \text{ of }A &= \frac{-20}{3}\left(\text{col}_1 \text{ of }A\right) + 14\left(\text{col}_3 \text{ of }A\right) ; & s_2 = \left(\frac{20}{3}, 0 , -14, 0, 1\right) \end{align}$$

So these three columns also don't provide any new information to the linear system. So the _true_ size of this matrix from the column size is actually 2 not 5.

So, we can say the true size of this matrix is $2 \times 2$.

The number of the pivot columns is the true size of every matrix from the column size. And we can argue that non-zero rows are all the pivot rows. And the number of pivot rows is same as number of pivot columns(becasue a column and a row can have at most one pivot).

So the number of _true_ columns and _true_ rows is actually the same no matter the size of the matrix and that is called the rank of the matrix. 

**The rank of the matrix is defined as the number of pivots.**

So in our example,

$$r(A) = 2$$

So if a matrix of size $m \times n$ has rank $r$, it means it has $r$ pivots and so $n-r$ free columns which means $n-r$ special solutions.

The concept of Rank is very important and it will pop up every once in a while. We will not go too deep into this concept as we will lose track of what we are doing. We will revisit it's applications whenever needed. But for now, let's move on to finding solutions to a linear system.

### Solving $A\mathbf{x}=\mathbf{b}$

We have already solved the form $A\mathbf{x} = \mathbf{b}$ for a unique solution. We will revisit these briefly and also include a new method to do it along with for the infinite solutions. We will see how reduced row echelon form, $R$ makes both the solutions easy and see the role of rank in it.

We know $A\mathbf{x}=\mathbf{b}$ is solvable only if $\mathbf{b}$ is in column space of $A$. We used to reduce this form to $U\mathbf{x} = \mathbf{b'}$ and then back substituted. But now we will go ahead and reduce it further to $R\mathbf{x} = \mathbf{d}$.

Now $R$ has the general form of $\begin{bmatrix} I & F \\ 0 & 0 \end{bmatrix}$. This matrix ($R$ as well as $A$) has a rank $r$.

If the matrix is $m \times n$ and the reduced form turned out to be $\begin{bmatrix} I & F\end{bmatrix}$ or $\begin{bmatrix} I \end{bmatrix}$ (i.e no Zero rows and $m \leq n$), which means all the rows have a pivot and so the rank of this matrix is $r = m \leq n$. This is called the **full row rank matrix**. In this form there is always either 1 or infinte number of solutions to $A\mathbf{x} = \mathbf{b} (\text{or } R\mathbf{x} = \mathbf{d})$. 



In case of the reduced form turned out to be $\begin{bmatrix} I \\ 0\end{bmatrix}$ or $\begin{bmatrix} I \end{bmatrix}$ (i.e no free columns and $n \leq m$), which means all the columns have a pivot and so the rank of this matrix is $r = n \leq m$. This is called **full column rank matrix**. This form can have zero or one solution.

The intersection of these two cases, i.e if a matrix is reduced to $\begin{bmatrix} I \end{bmatrix}$, it is an **invertible matrix**. It's rank is $r=m=n$. It has exactly one solution. $\mathbf{x} = A^{-1}\mathbf{b}$. More on it in a moment.

Coming back to the general case, when $R = \begin{bmatrix} I & F \\ 0 & 0 \end{bmatrix}$. We will now augment $A$ and $\mathbf{b}$ as $\begin{bmatrix} A & \mathbf{b}\end{bmatrix}$ and perform elimination on the full matrix to turn it into $\begin{bmatrix} R & \mathbf{d}\end{bmatrix}$.

$$E\begin{bmatrix} A & \mathbf{b}\end{bmatrix} = \begin{bmatrix} R & \mathbf{d}\end{bmatrix}$$

where:

* $R = EA$
* $\mathbf{d} = E\mathbf{b}$
* $E$ is the elimination matrix.

Let's talk about the no-solution case first. It is only possible if $R$ has zero rows. If $A$ of size $m \times n$ has rank $r$, then the number of zero rows in $R$ = $m-r$, and so the last $m-r$ elements of $\mathbf{d}$ must be zero inorder to have any solution. If they are not zero, we don't have any solution.

Now we have eliminated the possibility of no-solution, by checking all the last $m-r$ elements of $\mathbf{d}$ are zero. We find all the solutions of $A\mathbf{x} = \mathbf{b}$ by finding a particular solution, any solution, $\mathbf{x_p}$ to $A\mathbf{x} = \mathbf{b}$. Once we have found one solution, all solutions are this particular solution plus some vector in nullspace of $A$; $\mathbf{x_n} \in N(A)$.

$$\begin{align}\mathbf{x} &= \mathbf{x_p} + \mathbf{x_n} \\ \implies A\mathbf{x} &= A(\mathbf{x_p} + \mathbf{x_n}) \\ A\mathbf{x} &= A\mathbf{x_p} + A\mathbf{x_n}\\A\mathbf{x} &= \mathbf{b} + 0 \\A\mathbf{x} &= \mathbf{b}\end{align}$$ 

The particular solution can be any solution, but one easy to find is to set all the free variables as zero. Let's see an example:

$$ A \mathbf{x} = \begin{bmatrix} 3 & 4 & 2 & -3 & 8 \\ 6 & 8 & 3 & 4 & 2 \\ 12 & 16 & 7 & -2 & 18 \end{bmatrix} = \begin{bmatrix} 5 \\ 9 \\ 0\end{bmatrix} = \mathbf{b}$$

Let's use augmented matrix and perform elimination:

$$\begin{bmatrix} A & \mathbf{b}\end{bmatrix} =  \begin{bmatrix} \mathbf{3} & 4 & 2 & -3 & 8 & \vdots & 5 \\ 6 & 8 & 3 & 4 & 2 & \vdots & 9 \\ 12 & 16 & 7 & -2 & 18 & \vdots & 0\end{bmatrix}$$

$$\begin{align}\text{row}_2 \leftarrow \text{row}_2 - 2\text{row}_1\\
\text{row}_3 \leftarrow \text{row}_3 - 4\text{row}_1\\
\text{row}_1 \leftarrow \frac{\text{row}_1}{3}\\
E_1 \begin{bmatrix} A & \mathbf{b}\end{bmatrix} =  \begin{bmatrix} 1 & \frac{4}{3} & \frac{2}{3} & -1 & \frac{8}{3} & \vdots & \frac{5}{3} \\ 0 & 0 & \mathbf{-1} & 10 & -14 & \vdots & -1 \\ 0 & 0 & -1 & 10 & -14 & \vdots & -20\end{bmatrix}\end{align}$$

$$\begin{align}\text{row}_3 \leftarrow \text{row}_3 - \text{row}_2\\
\text{row}_1 \leftarrow \text{row}_1 + \frac{2}{3}\text{row}_2\\
\text{row}_2 \leftarrow -1(\text{row}_2)\\
E_2E_1 \begin{bmatrix} A & \mathbf{b}\end{bmatrix} = E \begin{bmatrix} 1 & \frac{4}{3} & 0 & \frac{17}{3} & \frac{-20}{3} & \vdots & 1 \\ 0 & 0 & \mathbf{1} & -10 & 14 & \vdots & 1 \\ 0 & 0 & 0 & 0 & 0 & \vdots & -19\end{bmatrix} =  \begin{bmatrix} R & \mathbf{d}\end{bmatrix}\end{align}$$

Elimination done. Two pivots, hence rank $r=2$. Last row is zero for $R$, but the corresponding element for $\mathbf{d}$ isn't zero. Hence no solution.

Let's take the same matrix with a different RHS.

$$\mathbf{b} = \begin{bmatrix}7 \\ 12 \\26 \end{bmatrix}$$

Since the matrix is same, we will apply the same elimination to the new RHS.


$$\begin{align}\mathbf{d} &= E\mathbf{b}\\ \implies \mathbf{d} &= \begin{bmatrix} 1 \\ 2 \\ 0 \end{bmatrix}\end{align}$$

The new eliminated augmented matrix is:

$$\begin{bmatrix} R & \mathbf{d} \end{bmatrix} = \begin{bmatrix} 1 & \frac{4}{3} & 0 & \frac{17}{3} & \frac{-20}{3} & \vdots & 1 \\ 0 & 0 & \mathbf{1} & -10 & 14 & \vdots & 2 \\ 0 & 0 & 0 & 0 & 0 & \vdots & 0\end{bmatrix}$$

Rank, as already discussed, was 2 and now the last $m-r$ rows of $\mathbf{d}$ are also zero, so we will have solution(s). Now we need the null vector of $A$ and a particular solution of this.

Null vector is any linear combination of the specific solutions to $A\mathbf{x} = 0$. The specific solutions are easy to find with $R$ form and in this example, we have three free columns and so three specific solutions.

Null vector:

$$\mathbf{x_n} = c_1 \begin{bmatrix}\frac{-4}{3} \\ 1 \\ 0 \\ 0 \\ 0\end{bmatrix} + c_2 \begin{bmatrix}\frac{-17}{3} \\ 0\\ 10 \\ 1 \\ 0\end{bmatrix} + c_3 \begin{bmatrix}\frac{20}{3} \\ 0 \\ -14 \\ 0 \\ 1\end{bmatrix}$$

where $c_1, c_2, c_3$ are scalar multpliers.

Now we need a particular solution. We can have any one by setting arbitrary values to free variables and back substituting to find the whole. One interesting is to set zero for all free values. Let's try that. 

Set $x_2 = x_4 = x_5 = 0$.

Backsubstitute, we get $x_3 = 2$, $x_1 = \frac{5}{3}$.

If you did the backsubstitution along, you would have certainly found that those values came directly from $\mathbf{d}$. Set the free variables to be zero and the pivot variables come from $\mathbf{d}$. Keep this in the back of your mind, it will come back later.

So our particular solution is:

$$x_p = \begin{bmatrix}1 \\ 0 \\ 2 \\ 0 \\ 0 \end{bmatrix}$$

Now the whole set of solutions is:

$$\begin{align}\mathbf{x} &= \mathbf{x_p} + \mathbf{x_n} \\ \implies \mathbf{x} & = \begin{bmatrix}1 \\ 0 \\ 2 \\ 0 \\ 0 \end{bmatrix} + c_1 \begin{bmatrix}\frac{-4}{3} \\ 1 \\ 0 \\ 0 \\ 0\end{bmatrix} + c_2 \begin{bmatrix}\frac{-17}{3} \\ 0\\ 10 \\ 1 \\ 0\end{bmatrix} + c_3 \begin{bmatrix}\frac{20}{3} \\ 0 \\ -14 \\ 0 \\ 1\end{bmatrix}; \qquad \forall c_1,c_2,c_3 \in R \end{align}$$

This is the complete solution of this example. Both parts of our solution, $\mathbf{x_p}$ and $\mathbf{x_n}$, came from the same Reduced augmented matrix.

> Does the set of all the solutions to $A\mathbf{x} = \mathbf{b}$ for any $A$ and $\mathbf{x}$ form a subspace? 

I highly recommend to practice on a few different forms of questions to get the hang of this.

### Invertibility

A matrix $A$ is said to be invertible if there exists another matrix $A^{-1}$ such that:

$$ AA^{-1} = A^{-1}A = I$$

The fact that $AA^{-1} = A^{-1}A$, implies that both these matrices have to be square and of same size. So rectangular matrices cannot have inverses. 

Also, each invertible matrix has a unique inverse. It has a simple proof:

Let's say $B$ and $C$ are two inverses of $A$, then:

$$B = BI = B(AC) = (BA)C = IC = C$$

Now we have to find an inverse of a matrix, we can formalize this problem as:

$$AX = I$$

Where $X$ is a matrix and not a vector, we have to find this matrix, if it exists. But we know that this $X$ is unique for this $A$ and $X$ is a square matrix of the same size as $A$.

Let's have an example for a $3 \times 3$ marix.

$$A = \begin{bmatrix} 1 & 5 & 3 \\ 2 & 9 & 4 \\ 5 & 7 & 1 \end{bmatrix}$$

We need to find its inverse(if it exists!).

$$AX = I \\ \begin{bmatrix} 1 & 5 & 3 \\ 2 & 9 & 4 \\ 5 & 7 & 1 \end{bmatrix} \ \begin{bmatrix}x_{11} & x_{12} & x_{13} \\ x_{21} & x_{22} & x_{23} \\ x_{31} & x_{32} & x_{33} \end{bmatrix} = \begin{bmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{bmatrix}$$

This above problem can be broken into three equations:

$$A\ \begin{bmatrix} x_{11} \\ x_{21} \\ x_{31} \end{bmatrix} = \begin{bmatrix} 1 \\ 0 \\ 0 \end{bmatrix},$$ 

$$A\ \begin{bmatrix} x_{12} \\ x_{22} \\ x_{32} \end{bmatrix} = \begin{bmatrix} 0 \\ 1 \\ 0 \end{bmatrix}$$

and 

$$A\ \begin{bmatrix} x_{13} \\ x_{23} \\ x_{33} \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \\ 1 \end{bmatrix}$$

Since the matrix $X$ has to be unique, each of these above equations should have just one unique solution, which means the nullspace should only have the zero vector and so the final solution should be only the particular solution.

$$\begin{align}\mathbf{x} &= \mathbf{x_p} + 0 \leftarrow \text{(zero vector of null space)}\\ \implies \mathbf{x} &= \mathbf{x_p}\end{align}$$

For just a unique solution, there should be no free columns, so the rank of the matrix should be equal to it's number of coulmns and since it is a square matrix, the rank is also equal to the number of rows, eliminating the possibility of zero rows in reduced form which can lead to no solution. 

**So a matrix is invertible, iff it is a square full ranked matrix.**

To check if a matrix is invertible, we simply check its rank, by reducing and counting the pivots. We can also reduce it to its reduced row echelon form and if that form ends of like an identity matrix, then it is an invertible matrix. 

Now we know which matrix is an invertible matrix. But what is the inverse of that matrix? And how to find it?


To find the inverse we simply have to solve the above equations and find the values of elements of matrix $X$.

Now we know we have just one unique solution which is also our particular solution to each of the above equations. While looking for the particular solution to a linear system, we used to set the free variables to be zero and the values of pivot variables came from $\mathbf{d}$.

Since in our form there are no free variables, the whole particular solution is equal to the $\mathbf{d}$, i.e:

$$\mathbf{x_p} = \mathbf{d}$$

So we just reduce our RHS and find the values.

Let's recall our equations:

$$\begin{bmatrix} 1 & 5 & 3 \\ 2 & 9 & 4 \\ 5 & 7 & 1 \end{bmatrix}\ \begin{bmatrix} x_{11} \\ x_{21} \\ x_{31} \end{bmatrix} = \begin{bmatrix} 1 \\ 0 \\ 0 \end{bmatrix},$$ 

$$\begin{bmatrix} 1 & 5 & 3 \\ 2 & 9 & 4 \\ 5 & 7 & 1 \end{bmatrix}\ \begin{bmatrix} x_{12} \\ x_{22} \\ x_{32} \end{bmatrix} = \begin{bmatrix} 0 \\ 1 \\ 0 \end{bmatrix}$$

and 

$$\begin{bmatrix} 1 & 5 & 3 \\ 2 & 9 & 4 \\ 5 & 7 & 1 \end{bmatrix}\ \begin{bmatrix} x_{13} \\ x_{23} \\ x_{33} \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \\ 1 \end{bmatrix}$$

Solving the first equation, we will augment the matrix and the RHS and then reduce, the resulting $\mathbf{d}$ will be the solution for the first column of the inverse matrix. 

$$\begin{bmatrix} 1 & 5 & 3 & \vdots & 1 \\ 2 & 9 & 4 & \vdots & 0\\ 5 & 7 & 1 & \vdots & 0\end{bmatrix}$$

$$\begin{align}\text{row}_2 \leftarrow \text{row}_2 - 2 \text{row}_1\\
\text{row}_3 \leftarrow \text{row}_3 - 5 \text{row}_1\\
\begin{bmatrix} 1 & 5 & 3 & \vdots & 1 \\ 0 & -1 & -2 & \vdots & -2\\ 0 & -18 & -14 & \vdots & -5\end{bmatrix}\end{align}$$

$$\begin{align}\text{row}_3 \leftarrow \text{row}_3 - 18 \text{row}_2\\
\text{row}_1 \leftarrow \text{row}_1 + 5 \text{row}_2\\
\text{row}_2 \leftarrow -1(\text{row}_2)\\
\begin{bmatrix} 1 & 0 & -7 & \vdots & -9 \\ 0 & 1 & 2 & \vdots & 2\\ 0 & 0 & 22 & \vdots & 31\end{bmatrix}\end{align}$$


$$\begin{align}\text{row}_1 \leftarrow \text{row}_1 + \frac{7}{22} \text{row}_3\\
\text{row}_2 \leftarrow \text{row}_2 - \frac{2}{22} \text{row}_3\\
\text{row}_3 \leftarrow \frac{\text{row}_3}{22}\\
\begin{bmatrix} 1 & 0 & 0 & \vdots & \frac{19}{22} \\ 0 & 1 & 0 & \vdots & \frac{-9}{11}\\ 0 & 0 & 1 & \vdots & \frac{31}{22}\end{bmatrix}\end{align}$$



And 

$$\mathbf{d} = \begin{bmatrix} \frac{19}{22} \\ \frac{-9}{11} \\ \frac{31}{22}\end{bmatrix} = \mathbf{x_p} = \mathbf{x}$$

This is the first column of the inverse matrix of $A$. We can do the same with other two equations, but instead of augmenting these RHS's one at a time and reducing, we can augment all of them and reduce all of the RHS at the same time.

We augment the matrix as:

$$\begin{bmatrix} A & I\end{bmatrix}$$

then reduce to:

$$\begin{bmatrix} I & X \end{bmatrix}$$

$X$ is the inverse of $A$.

It can also be seen as:

$$E\begin{bmatrix} A & I\end{bmatrix} = \begin{bmatrix} I & X \end{bmatrix}$$

Which means $EA = I$ and $EI= X$, So $E$ is the inverse of $A$ and $X= E$.

This method of finding the inverse is called the Gauss-Jordan method of finding the inverse.

Let's finish our current example and then look at one more example to conclude this topic.

Let's augment:

$$\begin{bmatrix} 1 & 5 & 3 & \vdots & 1 & 0 & 0 \\ 2 & 9 & 4 & \vdots & 0 & 1 & 0\\ 5 & 7 & 1 & \vdots & 0 & 0 & 1\end{bmatrix}$$

$$\begin{align}\text{row}_2 \leftarrow \text{row}_2 - 2 \text{row}_1\\
\text{row}_3 \leftarrow \text{row}_3 - 5 \text{row}_1\\
\begin{bmatrix} 1 & 5 & 3 & \vdots & 1 & 0 & 0 \\ 0 & -1 & -2 & \vdots & -2 & 1 & 0\\ 0 & -18 & -14 & \vdots & -5 & 0 & 1\end{bmatrix}\end{align}$$

$$\begin{align}\text{row}_3 \leftarrow \text{row}_3 - 18 \text{row}_2\\
\text{row}_1 \leftarrow \text{row}_1 + 5 \text{row}_2\\
\text{row}_2 \leftarrow -1(\text{row}_2)\\
\begin{bmatrix} 1 & 0 & -7 & \vdots & -9 & 5 & 0 \\ 0 & 1 & 2 & \vdots & 2 & -1 & 0\\ 0 & 0 & 22 & \vdots & 31 & -18 & 1\end{bmatrix}\end{align}$$

$$\begin{align}\text{row}_1 \leftarrow \text{row}_1 + \frac{7}{22} \text{row}_3\\
\text{row}_2 \leftarrow \text{row}_2 - \frac{2}{22} \text{row}_3\\
\text{row}_3 \leftarrow \frac{\text{row}_3}{22}\\
\begin{bmatrix} 1 & 0 & 0 & \vdots & \frac{19}{22} & \frac{-8}{11} & \frac{7}{22} \\ 0 & 1 & 0 & \vdots & \frac{-9}{11} & \frac{7}{11} & \frac{-1}{11}\\ 0 & 0 & 1 & \vdots & \frac{31}{22} & \frac{-9}{11} & \frac{1}{22}\end{bmatrix}\end{align}$$



So,

$$A^{-1} = \begin{bmatrix}  \frac{19}{22} & \frac{-8}{11} & \frac{7}{22} \\  \frac{-9}{11} & \frac{7}{11} & \frac{-1}{11}\\  \frac{31}{22} & \frac{-9}{11} & \frac{1}{22}\end{bmatrix}$$


```julia
A = [1 5 3; 2 9 4; 5 7 1]
inv_A = [19/22 -8/11 7/22; -9/11 7/11 -1/11; 31/22 -9/11 1/22]
A*inv_A
```




    3×3 Array{Float64,2}:
     1.0  -4.44089e-16  -5.55112e-17
     0.0   1.0          -5.55112e-17
     0.0  -7.77156e-16   1.0        




```julia
inv_A 
```




    3×3 Array{Float64,2}:
      0.863636  -0.727273   0.318182 
     -0.818182   0.636364  -0.0909091
      1.40909   -0.818182   0.0454545



We can also find the inverse in julia using:


```julia
inv(A)
```




    3×3 Array{Float64,2}:
      0.863636  -0.727273   0.318182 
     -0.818182   0.636364  -0.0909091
      1.40909   -0.818182   0.0454545



Let's look at one more example to clear this procedure.

Let's say our matrix,

$$A = \begin{bmatrix} 2 & 6 & 8 & 0 \\ 9 & 3 & 18 & 6 \\ 2 & 0 & 0 & 7 \\
11 & 9 & 26 & 6 \end{bmatrix}$$

To find the inverse, let's augment $A$ and $I$,

$$\begin{bmatrix} A & I \end{bmatrix} = \begin{bmatrix} 2 & 6 & 8 & 0 & \vdots & 1 & 0 & 0 & 0\\ 9 & 3 & 18 & 6 & \vdots & 0 & 1 & 0 & 0 \\ 2 & 0 & 0 & 7 & \vdots & 0 & 0 & 1 & 0 \\
11 & 9 & 26 & 6 & \vdots & 0 & 0 & 0 & 1 \end{bmatrix}$$

Elimiating,

$$E_1\begin{bmatrix} A & I \end{bmatrix} = \begin{bmatrix} 1 & 3 & 4 & 0 & \vdots & \frac{1}{2} & 0 & 0 & 0\\ 0 & -24 & -18 & 6 & \vdots & \frac{-9}{2} & 1 & 0 & 0 \\ 0 & -6 & -8 & 7 & \vdots & -1 & 0 & 1 & 0 \\ 0 & -24 & -18 & 6 & \vdots & \frac{-5}{2} & 0 & 0 & 1 \end{bmatrix}$$

$$E_2E_1\begin{bmatrix} A & I \end{bmatrix} = \begin{bmatrix} 1 & 0 & \frac{7}{4} & \frac{3}{4} & \vdots & \frac{-1}{16} & \frac{3}{24} & 0 & 0\\ 0 & 1 & \frac{3}{4} & \frac{-1}{4} & \vdots & \frac{9}{48} & \frac{-1}{24} & 0 & 0 \\ 0 & 0 & \frac{-7}{2} & \frac{11}{2} & \vdots & \frac{1}{8} & \frac{-1}{4} & 1 & 0 \\ 0 & 0 & 0 & 0 & \vdots & -1 & -1 & 0 & 1 \end{bmatrix}$$

We got a row of zeros for the matrix, which means this matrix is not full row ranked and so not full ranked at all. And so the inverse of this matrix is not possible.

I think this is a good stopping point. I would recommend practicing for everything that we have covered here. Next up we will try to explain Independence, basis and dimensions of subspaces.


{% include linalg/2.spaces/plot1.html %}