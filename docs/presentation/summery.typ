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
  title: [Summery of ACFlowSensitivity],
  subtitle: [],
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

== Denotation

Only consider fermionic system. 

The real Green's function is: $cal(G)$

The measure Green's function on the imag axis is: $G = [G_1,..,G_N]$

Denote the reconstructed Green's function as: $tilde(cal(G))$ . And denote $tilde(G) = [tilde(G)_1,..,tilde(G)_N] = [tilde(cal(G))(i w_1),..,tilde(cal(G))(i w_N)]$.

The real spectrum is: $cal(A)$

The date of the spectrum we calculate on the output mesh: $A = [A_1,..,A_M]$

Denote the reconstructed spectrum as: $tilde(cal(A))$.

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