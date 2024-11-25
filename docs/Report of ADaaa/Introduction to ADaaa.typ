#import "@preview/cetz:0.2.2"
#import "@preview/fletcher:0.4.5" as fletcher: node, edge
#import "@preview/touying:0.4.2": *
#import "@preview/touying-simpl-hkustgz:0.1.1" as hkustgz-theme

// cetz and fletcher bindings for touying
#let cetz-canvas = touying-reducer.with(reduce: cetz.canvas, cover: cetz.draw.hide.with(bounds: true))
#let fletcher-diagram = touying-reducer.with(reduce: fletcher.diagram, cover: fletcher.hide)

// Register university theme
// You can replace it with other themes and it can still work normally
#let s = hkustgz-theme.register()

// Set the numbering of section and subsection
#let s = (s.methods.numbering)(self: s, section: "1.", "1.1")

// Set the speaker notes configuration, you can show it by pympress
// #let s = (s.methods.show-notes-on-second-screen)(self: s, right)

// Global information configuration
#let s = (s.methods.info)(
  self: s,
  title: [Introduction to ADaaa],
  author: [Kaiwen Jin],
  date: datetime.today(),
  institution: [香港科技大学],
  others: none
)

// Pdfpc configuration
#let s = (s.methods.append-preamble)(self: s, pdfpc.config(
  duration-minutes: 30,
  start-time: datetime(hour: 14, minute: 00, second: 0),
  end-time: datetime(hour: 14, minute: 30, second: 0),
  last-minutes: 5,
  note-font-size: 12,
  disable-markdown: false,
  default-transition: (
    type: "push",
    duration-seconds: 2,
    angle: ltr,
    alignment: "vertical",
    direction: "inward",
  ),
))

// Extract methods
#let (init, slides, touying-outline, alert, speaker-note, tblock) = utils.methods(s)
#show: init.with(
  lang: "zh",
  font: ("Linux Libertine", "Source Han Sans SC", "Source Han Sans"),
)

#show strong: alert

// Extract slide functions
#let (slide, empty-slide, title-slide, outline-slide, new-section-slide, ending-slide) = utils.slides(s)
#show: slides.with()

= 1. Spectral density and Matsubara Green’s function

#tblock(title: [Spectral Density])[
  It has two kinds of forms. 
  
  $ 1. quad  A(x)= sum_(k=1)^N delta(x-x_i) $

  $ 2. quad A(x) in  cal(S), quad A(z) in bb(H(C)) $

  Now we nonly consider the second kind.
  
]

#tblock(title: [Green Function])[
  Define 1. Green function 

  $ G(z)=∫_bb(R) A(x)/(z-x)d x quad I m z>0, \ quad G(w)=lim_( eta  arrow.r 0^+) ∫A(x)/(w+i eta-x) d x = P.V.∫A(x)/(w-x)d x-i pi A(w) $

  $ A(w)=-  1/pi I m(G(w))  $ 

  Theorem 1. $ lim_( z arrow.r w) ∫A(x)/(z-x) d x=P.V.∫A(x)/(w-x)d x-i pi *s g n (eta)A(w) $

  Proposition 1.1. The green function $G(z)$ on $ I m z>0$ can be analytically continued to the whole complex plane $bb(C)$.

  It means that $G(z)$ has no pole on $bb(C)$.


 
]

= Touying 幻灯片动画

== 2. Complex Differentiation
Definition 2. For a function 

$ f: bb(C) arrow.r bb(C), quad f(x,y)=u(x,y)+i v(x,y), quad u,v in C^(infinity) $

We define that 

$ d f =(partial f)/(partial x)d x+(partial f)/(partial y)d y=(partial f)/(partial z)d z+(partial f)/(partial z^*)d z^* $

In above formula, 

$ d z= d x+ i d y, quad d z^*=d x - i d y $

$ (partial )/(partial z)=1/2((partial )/(partial x)-i (partial )/(partial y)), quad (partial )/(partial z^*)=1/2((partial )/(partial x)+ i (partial )/(partial y)) $




== 2. Complex Differentiation

Proposition 2.1 $ ((partial f)/(partial z))^* =  ((partial f^*)/(partial z^*)) $

Proposition 2.2 $ d g(f)=((partial g)/(partial f) (partial f)/(partial z) + (partial g)/(partial f^*)(partial f^*)/(partial z) )d z+ ((partial g)/(partial f) (partial f)/(partial z^*) + (partial g)/(partial f^*)(partial f^*)/(partial z^*) )d z^* $




== 3. AAA algorithm

It's a interesting algorithm because it's most important idea is not barycentric but deviding a set of points into 2 parts. One of them is insert points set and the second is checking points set.

The reason is you can't decide the weight just with all ponits as insert points.

Denote chosen points as $A$, and set of waiting points as $B$. Assume that 
$ A={z_1,..,z_n}, quad B={z_{n+1},..,z_m} $

Now consider:
$ L=((G(z_j)-G(z_k))/(z_j-z_k))_(j k) $

Then we get a sub matrix $L_n$ from it by getting the first $n$ columns and $n+1 , . , m$ rows. 

#image("1.png")

Now for 
$ G(z) approx (N_n(z))/(D_n(z)) $

$ N_n(z)= sum_(j=1)^n (w_j G(z_j))/(z-z_j), quad  D_n(z)=sum_(j=1)^n (w_j)/(z-z_j) $

We have 
$ (G D_n-N_n)(B)=L_n w $

$ arrow.double.r min_w ||(G D_n-L_n)(B)||_(L^2)=min_w ||L_n w||=min sigma(L_n) $

Then we can use svd to find such $min sigma(L_n)$ and related $w$.

Then we chose 
$ z_(n e w)=a r g m a x_(z in B) quad ||G(z)-(N_n(z))/(D_n(z))|| $

Add $z_(n e w)$ into $A$ and delete it from $B$ and continue iteration.

Get $G(z)$ and then we can reconstruct $A(w)$






== 4. Calculate $nabla$ Loss

Given $G_0(i w n)$ and $w n$, the way we calcuate Loss function is as following chart and the Loss function is defined as 

$ L o s s (G(i w n), w e i g h t)=||A(x)-A_0(x)||_2$


#image("2.png")

Finite difference (FD) works awful for calculating derivative of

$ L o s s(G,w) $ 

So we use AD to calculate $nabla L o s s$

But because $L_0$ is an ill-condition number matrix and the log of its condition number is approximately proportional to the number of $G(i w n)$, formulas of calculating compelx SVD performs disastrously. So we directly use FD and infact  

$ (w e i g h t(G_0+epsilon)-w e i g h t(G_0))/epsilon $

performs very stably as $epsilon$ changes.

Then with proposition 2.2 we have

Theorem 4.
$ nabla_G L o s s(G,w(G))=2((∂L)/(∂G))^*  = nabla_1 L + 2((∂L)/(∂w))^(dagger) * ((J w)/(J G))^* + 2((∂L)/(∂w))^T * (J w) /(J G^*) $  

Here $(J y)/(J x)$ means the jacobian matrix.



