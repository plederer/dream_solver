# Hybrid Discontinuous Galerkin Method

## Boundary Conditions

We impose boundary conditions weakly by multiplying some boundary operator $B(\CVAR, \widehat{\CVAR})$ with facetwise test functions
\begin{align*}
\int_\Gamma B(\CVAR, \widehat{\CVAR}) \widehat{\vec{V}} = 0
\end{align*}

Some useful notation
\begin{align*}
\mat{A}^{+}(\CVAR) &:= \mat{A}_{\CVAR}^{\xi, +}(\CVAR) =  \mat{P}(\CVAR) \mat{\Lambda}^{+}(\CVAR) \mat{P}^{-1}(\CVAR) \\
\mat{A}^{-}(\CVAR) &:= \mat{A}_{\CVAR}^{\xi, -}(\CVAR) =  \mat{P}(\CVAR) \mat{\Lambda}^{-}(\CVAR) \mat{P}^{-1}(\CVAR)
\end{align*}

The eigenvalues $\mat{\Lambda}$ are split according to the imposed boundary e.g. a subsonic outflow requires only one incoming characteristic, hence
\begin{align*}
 \mat{\Lambda}^{+} &:=
\begin{bmatrix}
0 & 0 & 0 & 0 \\
0 & u_{\vec{\xi}} & 0 & 0 \\
0 & 0 & u_{\vec{\xi}} & 0 \\
0 & 0 & 0 & u_{\vec{\xi}} + c
\end{bmatrix}, \\
\mat{\Lambda}^{-} &:=
\begin{bmatrix}
u_{\vec{\xi}} - c & 0 & 0 & 0 \\
0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0
\end{bmatrix}.
\end{align*}

This can obvisouly been done in a dynamic fashion, if the type of boundary is not known at priori.

### **Farfield**
\begin{align*}
B(\CVAR, \widehat{\CVAR}) &:= \mat{A}^{+}(\widehat{\CVAR}) (\vec{U} - \widehat{\CVAR}) - \mat{A}^{-}(\widehat{\CVAR}) (\vec{U}_\infty - \widehat{\CVAR}) 
\end{align*}

### **NSCBC - Navier Stokes Characteristic Boundary Conditions**

Recall, Eulers equations in characteristic form

\begin{align*}
\pdt{\CHVAR} +  \underbrace{\Lambda  \frac{\partial \CHVAR}{\partial \xi}}_{\vec{\mathcal{L}}} + \mat{B}^{\eta}_{\CHVAR}\frac{\partial \CHVAR}{\partial \eta}  &= 0,
\end{align*}

In the NSCBC approach we model the incoming amplitudes in normal direction $\vec{\mathcal{L}}^{-} = \mat{\Lambda}^{-} \frac{\partial \CHVAR}{\partial \xi}$ and solve the partial differential equation in conservative form
\begin{align*}
\pdt{\widehat{\CVAR}} +  \mat{P}\vec{\mathcal{L}} + \mat{P} \mat{B}^{\eta}_{\CHVAR} \mat{P}^{-1} \frac{\partial \widehat{\CVAR}}{\partial \eta}  &= 0,
\end{align*}
at the boundary, instead of prescribing an outer state $\CVAR_\infty$.



::::{admonition} Subsonic Outflow

For a subsonic outflow exactly one characteristic, namely the acoustic wave, is entering the domain from
downstream, hence we have to model

\begin{align*}
\vec{\mathcal{L}}^{-}_{acou} &= (u_\xi - c) \left[\frac{\partial p}{\partial \xi} - \rho c \frac{\partial u_\xi}{\partial \xi} \right].
\end{align*}

<hr style="border:1px solid gray">

**1D approach**

Usually on a subsonic outflow, one wishes to impose some pressure $p_\infty$, while letting the velocity field untouched, thus an ansatz found in literature is to model the incoming acoustic wave as

\begin{align*}
\vec{\mathcal{L}}^{-}_{acou} &=  c (1 - \Ma^2)  \sigma \frac{p - p_\infty}{L}
\end{align*}
or
\begin{align*}
\vec{\mathcal{L}}^{-}_{acou} &=  (u_\xi - c)  \sigma \frac{p_\infty - p}{L}
\end{align*}

:::{hint}
If $\sigma = 0$ a perfectly non-reflecting outflow $\vec{\mathcal{L}}^{-}_{acou} = 0$ is achieved in theory. Nevertheless, a drift of the mean pressure has been observed.
:::

Neglecting tangential contributions, the partial differential equation to solve at the boundary, becomes
\begin{align*}
\pdt{\widehat{\CVAR}} +  \mat{P}\vec{\mathcal{L}}  &= 0,
\end{align*}

<hr style="border:1px solid gray">


**2D approach**

If tangential effects are not negligible it is possible to model the velocity gradient of the incoming
acoustic wave using the incompressibility assumption, namely
\begin{align*}
\frac{\partial u_\xi}{\partial \xi} + \frac{\partial u_\eta}{\partial \eta} = 0.
\end{align*}

The modelled incoming acoustic wave then becomes
\begin{align*}
\vec{\mathcal{L}}^{-}_{acou} &=  (u_\xi - c)  \sigma \frac{p_\infty - p}{L} - (1 - \Ma_\xi) \rho c^2 \frac{\partial u_\eta}{\partial \eta}
\end{align*}

:::{hint}
If we substitute back into the characteristic pde, we obtain for the incoming acoustic wave
\begin{align*}
\pdt{\CHVAR_{acou}} + (u_\xi - c)  \sigma \frac{p_\infty - p}{L} + u_\eta \left( \frac{\partial p}{\partial \eta} - \rho c \frac{\partial u_\xi}{\partial \eta} \right) + \Ma_\xi \rho c^2 \frac{\partial u_\eta}{\partial \eta} = 0.
\end{align*}

:::

::::
<!-- 
    \begin{align*}
    \pdt{\widehat{\CVAR}} +  \mat{P}\vec{\mathcal{L}} + \mat{P} \mat{B}^{\eta}_{\CHVAR} \mat{P}^{-1} \frac{\partial \widehat{\CVAR}}{\partial \eta}  &= 0,
    \end{align*} -->

