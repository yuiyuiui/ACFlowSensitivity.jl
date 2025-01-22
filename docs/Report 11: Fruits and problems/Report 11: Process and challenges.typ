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
  title: [Add AD into analytic continuation algorithm of ACFlow],
  subtitle: [Current process and challenges],
  author: [Kaiwen Jin],
  date: datetime.today(),
  institution: [HKUST-GZ],
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

=

#tblock(title:[Challenges we meet])[
  1. AD on AAA is numerically instable and AAA may be undifferentiable; 

  2. Stochastic method works so bad. Should we give up or improve them? 

  3. Can we know poles of $A(w)$ in advance? Otherwise We have no effective method on discrete $A(w)$.
]

= 

#tblock(title: [Contents])[
  0. Conclusion.

  1. Continuous spectral density function ($A(w)$ for short)
      
    1.1 Features and improvement of Barycentric/AAA algorithm and challenges of it's AD.

    1.2 The implementation of the AD of chi2kink (a MaxEnt method).

    1.3 Awful performance of stochastic methods.

  2. Discrete $A(w)$

    2.1 Do we know poles of $A(w)$ in advance? And limitation of directly solve amplitudes.

    2.2 Effectiveness and limitation of Bary and chi2kink.

    2.3 Awful performence of statistic methods. 
    
  3. What should we do about stochastic methods?

]
=

#tblock(title:[0. Conclusion])[
  1. Accuracy and noise robustness: Chi2kink(MaxEnt)  >> math methods(AAA,Nanv) >> Stochastic method

  2. Speed : Math methods > Chi2kink >> Stochastic method

  3. Add AD:
  
   Chi2kink : well done

   AAA : essentially diffcult

   Stochastic Methods : unknown

  4. Challenges we meet : 
  
  (1) AD on AAA is numerically instable and AAA may be undifferentiable;

  (2) Stochastic method works so bad. Should we give up or improve them? 

  (3) Can we know poles of $A(w)$ in advance? Otherwise We have no effective method on discrete $A(w)$. Of course I see most algorithm assume we don't kmow poles in advance, but knowing poles in advance is the only effective method I can think out for discrete $A(w)$.

]

= 

#tblock(title: [1. Continuous $A(w)$])[
  1.1 Features and improvement of Barycentric/AAA algorithm and challenges of it's AD.

  Calling it barycentric algoorithm, what we use in fact is aaa algorithm. AAA and Nevanlinna are both purely mathematical algorithms. They are fast but extremely instable to noise. AAA for example:
  #image("1.png")
  #image("2.png")
  #image("3.png")

  ACflow's wrong use of svd in LinearAlgebra will cause the algorithm go wrong. I have fixed the bug.

  The main problem of it's AD is, in AAA we can't avoid applying svd on a Lowner matrix 
  $ C=(( G(i w_i)-G(i w_j))/(i w_i - i w_j))_(i  times  j) $
  and Cauchy matrix
  $ L=(1/(i w_i - i w_j))_(i times j) $
  Both of them are ill-conditioned matrices and will cause great numerical instability. Derivative formula of svd falls here. Some term of the formula shoule be zeros in theory but it end to be more than 1e5 in practical code running.

  Finite difference (FD for short) only works in rare specific cases. In most cases, AD and FD just blow up. And some FD results just suggest this method may be just undifferentiable.
  The following chart show the partial derivative of weights coefficient got in AAA with respect to $G$ we get:
  #image("4.png")

  1.2 The implementation of the AD of chi2kink (a MaxEnt method)
  It's fast and has extremely strong noisy robustness. But it only work for continous $A(w)$. The following chart is its performance under 1e-2 noise:
  #image("5.png")

  Now we have realized AD of chi2kind with combination of formula and Zygote. The running result is reasonale.
  #image("6.png")

  1.3 Awful performence of stochastic methods.

  AAA need second and Chi2kink need less than half a minute. But Sandvic algorithm need at least 15 minutes and its result is extremely awful:
  #image("7.png").

  According to ACFlow's document, other stochastic methods are equally slow or even much more. And accuracy of stochastic methods is worse than Barycentric/AAA. 

]

= 

#tblock(title: [2. Discrete $A(w)$])[
  2.1 Do we know poles of $A(w)$ in advance? And limitation of directly solve amplitudes.

  If we know accurate poles in advance, we can just solve
  $ (1/(i w_i - p_j))_(i times j)X=(G(i w_i))_(i times 1) $

  Limitation is that the cauchy matrix $(1/(i w_i - p_j))_(i times j)$ is an ill-conditioned matrix. But we have existing mathod to solve this question.

  The following discussion is based on that we don't know $w_n$ in advance.

  2.2 Effectiveness and limitation of Bary and chi2kink.

  Chi2kink, or all maximum entropy methods, is not suitable for discrete $A(w)$.

  Barycentric/AAA improved by me can find first several poles and amplitudes relatively accurately. But for bigger $w_n$ the result just blows up. So it can't be applied in practice.


  2.3 Awful performence of statistic methods.
  ACFlow doesn't give discrete example. I test the algorithm by example I write so I am not sure if I write an example not suitable. ACFlow documents claim these stochastic methods' accuracy is even worse than Barycentric/AAA. in my example, it is so:
  #image("8.png")

  It only find the first pole, which is same as original AAA (not improved by me).

  And it's still extremely slow.

]

#tblock(title: [What should we do about stochastic methods?])[
  With stochastic algoorithm's awful performence, do we need its AD?

  If we know poles in advance, it's tatolly different question. If not so and we have to add AD to stochastic algorithm, I have some ideas about future direction:

  1. Just add AD to stochastic algoorithm of ACFlow.

  2. Rewrite and improve ACFlow's code realization. If it work well (at least for discrete $A(w)$), we add AD.

  3. Versions of stochastic algorithms that ACFlow realize is relatively old. The Sandvic algoorithm it realizes was published in 2017 and a improved version was published in 2020. We can find new version and realize it to see if they work better.


]







