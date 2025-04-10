#import "@preview/peace-of-posters:0.5.0" as pop
#import "@preview/cetz:0.2.2": canvas, draw, tree, plot
#import "@preview/pinit:0.1.3": *


#show link: set text(blue)
#set page("a0", margin: 1cm)
#pop.set-poster-layout(pop.layout-a0)
#pop.set-theme(pop.uni-fr)
#set text(size: pop.layout-a0.at("body-size"))
#let box-spacing = 1.2em
#set par(justify: true) // 设置文本段落两端对齐
#set columns(gutter: box-spacing)
#set block(spacing: box-spacing)
#pop.update-poster-layout(spacing: box-spacing)

#pop.title-box(
  "Sensitivity Analysis of Green's Function Analytic Continuation",
  authors: [Kaiwen Jin, Jinguo Liu, Zecheng Gan],
  institutes: text(36pt)[
  $""^dagger$Advanced Materials Thrust, Function Hub, The Hong Kong University of Science and Technology (Guangzhou)
  ],
  image: image("amat-dark.png", width: 150%),
  title-size: 1.35em,
)

#columns(2,[

  #pop.column-box(heading: "Abstract")[
Numerical analytic continuation of Green's functions has been widely used in many areas, such as Condensed Matter Physics and Computational Materials Science. The performance of the analytic continuation methods is closed related to their sensitivity to the input data, which determines how much error is introduced by the noise in the input data. However, it is difficult to give an exact expression or estimate for the error. Automatic differentiation (AD) is a powerful tool for computing the sensitivity, while a detailed study of the numerical analytic continuation sensitivity analysis and AD is still missing.

Here, we first review analytic continuation of Green's functions and automatic differentiation. Then, we demonstrate a comparison between the error bounds obtained through sensitivity analysis of MaxEntropy Method using AD and the actual errors. Finally, I will present the automatic differentiation rules employed in our sensitivity analysis, along with other AD implementations we have developed. 
]


  #pop.column-box(heading: "Analytic Continuation of Green's Functions")[
  We consider a quantum system with Hamiltonian $H$, to get a correlation function:
  $
    &S_(A B)(t) = angle.l A(t) B(0) angle.r= 1/tr(e^(-beta H)) tr(e^(-beta H) A(t) B(0))\
  $
    Set $tau = i t$, we have: $G^0_(A B)(tau) = S_(A B)(- i tau)$, then we can apply Monte Carlo method to calculate $G^0_(A B)(tau)$ on $tau in [0,beta]$ and thus imagary data of $S_(A B)(z)$. Then we can get values of green function on imaginary axis by Fourier transform @gubernatis1991quantum:
$
  & G^0_(A B)( tau) = 1/beta sum_(n in ZZ) e^(-i omega_n tau) G_(A B)( i omega_n)\
$

Then we do analytic continuation to real axis to get the spectral density and extract dynamical information, corresponding to the real-axis data of $S_(A B)(t)$ that we initially require@gubernatis1991quantum:
$
  & A(w) = - 1/pi lim_(eta arrow 0^+) G_(A B)(w+i eta),quad G(z) = integral_RR A(w) / (z-w) d w, quad Im z >0
$
*Recommended Analytic Continuation Methods:*
#grid(
  columns: 3,
  gutter: 30pt,
  [
    #stack(
      spacing: 0.1pt,
      [#v(0pt) #image("maxent_smooth.svg", width: 330pt)],
      [#align(center)[(1.a) MaxEntropy, smooth]]
    )
  ],
  [
    #stack(
      spacing: 0.1pt,
      [#v(9pt)#image("SPX.png", width: 305pt)],
      [#align(center)[(2.a) SPX, smooth]]
    )
  ],
  [
    #stack(
      spacing: 0.1pt,
      [#v(9pt)#image("aaa_smooth.svg", width: 330pt)],
      [#align(center)[(3.a) AAA, smooth]]
    )
  ]
)
#grid(
  columns: 3,
  gutter: 30pt,
  [
    #stack(
      spacing: 0.1pt,
      [#v(0pt) #image("maxent_delta.svg", width: 330pt)],
      [#align(center)[(1.b) MaxEntropy, delta]]
    )
  ],
  [
    #stack(
      spacing: 0.1pt,
      [#v(9pt)#image("SPX.png", width: 305pt)],
      [#align(center)[(2.b) SPX, delta]]
    )
  ],
  [
    #stack(
      spacing: 0.1pt,
      [#v(9pt)#image("aaa_delta.svg", width: 330pt)],
      [#align(center)[(3.b) AAA, delta]]
    )
  ]
)


(1) MaxEntropy Method: fast, accurate and noise-resistant. It's the most common method in practice. But it can not deal with delta type: $A(w) = sum_(i=1)^n  gamma_i delta(w-p_i)$

(2) Stochastic Poles Expansion@huang_stochastic_2023: works well for both smooth and delta type spectrum without noise. But it's extremely slow and unstable to noise. 

(3) AAAa@nakatsukasa2018aaa: work for both smooth and delta type spectrum, and are fast for delta type. But they are extremely unstable to noise.

For more performance comparisons of analytic continuation methods, please refer to https://huangli712.github.io/projects/acflow/man/tricks.html
  ]


  // These properties will be given to the function which is responsible for creating the heading
  #let hba = pop.uni-fr.heading-box-args
  #hba.insert("stroke", (paint: gradient.linear(green, red, blue), thickness: 10pt))

  // and these are for the body.
  #let bba = pop.uni-fr.body-box-args
  #bba.insert("inset", 30pt)
  #bba.insert("stroke", (paint: gradient.linear(green, red, blue), thickness: 10pt))

  #pop.column-box(heading: "Automatic Differentiation", stretch-to-next: true)[
    #grid(
  columns: 2,
  gutter: 250pt,
  [
    #stack(
      spacing: 0.1pt,
      [#v(0pt) #image("ForwardAD.png", width: 300pt)],
      [#align(center)[Forward AD]]
    )
  ],
  [
    #stack(
      spacing: 8pt,
      [#v(9pt)#image("BackwardAD.png", width: 315pt)],
      [#align(center)[Backward AD]]
    )
  ]
)
*Forward AD* It's efficient when output are more than input, and can compute the gradients of some non-differentiable functions. 
$
  ln 😅 = 💧 ln 😄
$

*Backward AD* Inverse mode of Forward AD. It's efficient when input are more than output, and can bypass the multi-valued mapping part of the algorithm to calculate derivatives.
  ]


#colbreak()
// 



  #pop.column-box(heading: "Sensitivity Analysis Results ")[
 #text(weight: "bold")[Sensitivity of Green's Function MaxEntropy Analytic Continuation]


#grid(
  columns: 2,
  gutter: 65pt,
  [
    #stack(
      spacing: 1pt,
      [#image("error_bound.png", width: 530pt)],
      [#align(center)[#text(size: 22pt)[(c) Error bound with perturbation]]]
    )
  ],
  
  [
    #stack(
      spacing: 7pt,
      [#image("varybound.png", width: 530pt)],
      [#align(center)[#text(size: 22pt)[(d) Error bound varies with perturbation]]]
    )
  ]
)

(a) We superimposed little noise onto a function composed of two Gaussian peaks as the true spectrum $A_0(w)$. Using data on imaginary axis ${G(i w_n)}_(i=1)^N$ as input, we reconstructed the corresponding spectral function data ${A_i}_(i=1)^M$ as output. We use AD to get $(partial A)/(partial G)$ and $eta||(partial A_i)/(partial G)||_2$ provides an upper bound on the variation in $A_i$ under perturbations $eta$. The perturbation is introduced as follows:
$
  G_(#text[pert]) = G + eta * #text[rand] (N) .* exp(2 pi i #text[rand] (N))
$

(b) We define $L_2$ loss function as:
$
  & #text[Loss] (A) = ||A(w)-A_0(w)||_2 = sqrt(sum_(i=1)^M |A_i - A_(0i)|^2 d)\

$
The loss function is non-differentiable, otherwise its derivatives are always zeros. As perturbation $eta$ increases, $||nabla #text[Loss]||$ decreases rapidly. I am unable to explain this phenomenon. It seems to suggest that as the smoothness of the original spectrum decreases, the algorithm's sensitivity to the input drops rapidly.

  ]
  #pop.column-box(heading: "AD Rules")[
    AD rules we use in above sensitivity analysis and others we implement:

    *Backward AD rules*
      #table(
  columns: (auto, auto, auto,auto,auto),
  inset: 20pt,
  align: horizon,
  table.header(
    [Pfaffian],[LP],[SDP],[FFT],[NUFFT/INUFFT],[LU Decomp],[Schatten Norm],[GMRES],[Norm Matrix Func],[Cholesky Decomp],[QR],[SVD/RSVD],[Det],[Inverse],[Least Square],[Sym/Norm Eigen],[Linear System],[Matrix Multiply]
  ),)
  *Froward AD rules*
   #table(
  columns: (auto, auto,auto),
  inset: 20pt,
  align: horizon,
  table.header(
    [$L_1$ Norm],[$L_2$ Norm],[Zeros Solve]
  ),)

  *Backward AD rules to be implemented*
      #table(
    columns: (auto, auto, auto,auto),
    inset: 20pt,
    align: horizon,
    table.header(
      [Schur Decomp],[Jordan Decomp],[Matrix Function],[Rank-1 Decomp],[Matrix P Norm],[Generalized Inverse]
    )
   ) 
   #grid(
  columns: 2,
  gutter: 65pt,
  [
    We implement Backward AD rules in this GitHub repository:
    https://github.com/yuiyuiui/BackwardsLinalg.jl
  ],
  
  [
    #stack(
      spacing: 7pt,
      [#image("QRcode.png", width: 120pt)],
   
    )
  ]
)

  ]


  #pop.column-box(heading: "References", stretch-to-next: true)[ 
    // 调节Reference字体大小
    #text(size: 26pt)[ 
      #bibliography("bibliography.bib", title: none)
    ]
    
  ]
])

#pop.bottom-box()[
  #align(right, [
    #align(horizon, grid(
      columns: 5, 
      column-gutter: 30pt,
      h(50pt),

      [#image("email.png", width: 70pt)],
      "kjin327@connect.hkust-gz.edu.cn"
    ))
  ])
]
