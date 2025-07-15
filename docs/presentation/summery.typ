#import "@preview/cetz:0.3.3"
#import "@preview/fletcher:0.5.5" as fletcher: node, edge
#import "@preview/touying:0.6.1": *
#import "./lib.typ": *

// cetz and fletcher bindings for touying
#let cetz-canvas = touying-reducer.with(reduce: cetz.canvas, cover: cetz.draw.hide.with(bounds: true))
#let fletcher-diagram = touying-reducer.with(reduce: fletcher.diagram, cover: fletcher.hide)

#show: hkustgz-theme.with(
  // Lang and font configuration
  lang: "en",
  font: (
    (
      name: "Libertinus Serif",
      covers: "latin-in-cjk",
    ),
    "Source Han Sans SC", "Source Han Sans",
  ),

  // Basic information
  config-info(
    title: [Summery of ACFlowSensitivity],
    subtitle: [],
    author: [Kaiwen Jin],
    date: datetime.today(),
    institution: [HKUST-GZ],
  ),

  // Pdfpc configuration
  // typst query --root . ./examples/main.typ --field value --one "<pdfpc-file>" > ./examples/main.pdfpc
  config-common(preamble: pdfpc.config(
    duration-minutes: 30,
    start-time: datetime(hour: 14, minute: 10, second: 0),
    end-time: datetime(hour: 14, minute: 40, second: 0),
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
  )),
)

#title-slide()


== Example

== Methods Summary
1. Mathematical method: Barycentric Rational Approximation (AAA + prony denoise), Nevanlinna
2. Maximal Entropy Method. According to parameters choosing: Historic Algorithm, Classic Algorithm, Chi2kink Algorithm, Bryan Algorithm (average).
3. Stochastic method.
Generate some spectrum with MC(SA) method
$
  & A(w) = sum_j gamma_j/(w-p_j)\
  & P(C arrow C') = exp(-alpha(chi^2(C') - chi^2(C)))
$

And average them.

== Methods Summary

(1) How choose the inverse temperature ($alpha, 1/theta$) of Simulated Annealing: choose a good $alpha$ (chi2kink, chi2min), take average

(2) How average measure spectrum: average all measured spectrums, average good spectrums, average spectrums with weights

(3) Sample what type of spectrums: $A(w)$ (better for delta type), $n(x)$ (better for smooth type)

== Methods Summary

#table(
  columns: (6cm, 6cm, 6cm, 6cm),
  align: (left, center, center, center),
  stroke: 0.5pt + black,
  [*Method*], [*Inverse temp*], [*Ave Spec*], [*Sample Obj*],
  [ssk(Sandvik)], [a good], [ave all], [A(w)],
  [sac(Beach)], [ave all], [weights], [n(x)],
  [som], [❌], [all good], [A(w)],
  [spx], [a good], [ave all], [A(w)],
)
== Methods Compare

#table(
  columns: (3.7cm, 3.7cm, 3.7cm, 3.7cm, 3.7cm, 3.7cm, 3.7cm),
  align: (left, center, center, center, center, center, center),
  stroke: 0.5pt + black,
  [*Method*], [*cont*], [*delta*], [*mixed*],[*noise robust*],[*Speed*],[*Accuracy
  (no noise)*],
  [barrat], [✅], [✅], [✅(maybe)], [weak], [fast], [high],
  [nac], [✅], [✅], [✅(maybe)], [weak], [fast], [high],
  [maxent
  (chi2kink)], [✅], [❌], [❌], [strong], [fast], [high],
  [ssk], [❌], [✅], [❌], [weak], [slow], [high],
  [sac], [✅], [✅], [❌(Difficult)], [weak], [slow], [low],
  [spx], [✅(Against)], [✅], [], [weak], [extremely slow], [low],
  [som], [✅], [✅], [❌(difficult)], [weak], [little slow], [low],
)


== Methods Compare(show ssk)
Why choose `ssk` in stochastic methods: It's the most accuracy method in stochastic methods.

#image("../ssk/ssk_cont.svg")
#image("../ssk/ssk_delta.svg")

== Methods Compare(sac vs ssk)
#image("../sac/sac_cont.svg")
#image("../sac/sac_delta_2p.svg")
#image("../sac/sac_delta_512p.svg")

== Methods Compare(som vs ssk)
#image("../som/som_cont.svg")
#image("../som/som_delta.svg")

== Methods Compare(spx vs ssk)
#image("../spx/spx_delta.svg")
Finally I choose `barrat`, `maxent(chi2kink)` and `ssk` to do sensitivity analysis.
== Analysis Results Show
(Show Tests)
 #image("../../error_plot/MaxEnt/eb-maxent-cont.svg")


#table(
  columns: (6cm, 6cm, 6cm),
  align: (left, center, center),
  stroke: 0.5pt + black,
  [*Method*], [*accuracy*], [*stability*],
  [barrat], [high], [✅],
  [maxent(chi2kink)], [moderate], [✅],
  [ssk], [low], [❌],
)