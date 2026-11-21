# Writing this paper: craft, structure, and not sounding like a machine

Three parts: **structure** (what goes where, from Mensh and Kording),
**sentences** (how to make prose readable, from Gopen and Swan), and **voice**
(how to avoid the specific patterns that mark text as machine-generated).

Sources are listed at the end. Everything here is applied to this paper
specifically, with examples drawn from its own results.

---

## Part 1: structure

From Mensh and Kording, *Ten simple rules for structuring papers*, PLOS
Computational Biology (2017). Their ten rules, with what each means here.

### 1. Focus on a single central contribution, stated in the title

Papers that pursue several contributions are forgettable. This paper has eleven
gates and is at risk of reading as a list. **It needs one sentence.**

The candidate that fits the evidence:

> Reconstruction error under forward-operator mismatch can be bounded without
> ground truth and achieved by in-domain calibration, but cannot be anticipated
> blindly.

That is one claim with three clauses, not three claims. Everything else -- the
haze finding, the U-Net result, the retractions -- is support or limitation.

### 2. Write for a reader who does not know your work

Assume no knowledge of BioSR, of SIM reconstruction, or of why an assumed PSF is
a problem. Define the operator-mismatch setting before using the phrase.
Conformal prediction needs one paragraph of definition, not a citation.

### 3. Use context-content-conclusion at every scale

- **Document:** introduction sets context, results are content, discussion
  brings the conclusion.
- **Paragraph:** first sentence names the topic, the middle carries the new
  information, the last delivers the takeaway.
- **Abstract:** all three, in that order.

The failure mode this prevents is the paragraph that ends on its weakest
sentence, which is where readers stop.

### 4. Optimise logical flow: avoid zig-zag, use parallelism

Present related ideas consecutively. Do not introduce the haze finding, move to
the U-Net, then return to haze. Use identical syntax for parallel concepts: if
one gate is "Does X repair Y?", the next should be "Does Z repair Y?", not
"Whether Z has any effect on Y".

### 5. Tell a complete story in the abstract

More people read the abstract than the paper. Its shape:

- **Context**, narrowing from the field to the specific gap. Deconvolution
  assumes a PSF; when the assumption is wrong the output is confidently wrong;
  no existing reliability method addresses operator mismatch.
- **Content**, beginning "Here we". The bound, the grid, the real-measurement
  validation.
- **Conclusion**, answering the question posed and stating what it means for
  the field -- including that the deployable warning signal does not work.

Do not hide the negative in the abstract. It is a TMLR-appropriate result and
concealing it invites a reviewer to feel misled.

### 6. Say why the paper matters in the introduction

Establish the gap **and its cost** before the contribution. The cost here is
concrete: a deconvolved image that is partly invented, presented to a biologist
as data, with no signal that anything is wrong.

### 7. Results as a sequence of statements supported by figures

Each results subsection should be a claim, not a procedure. "The one-sided bound
holds on 48 of 48 cells" is a statement; "we computed d+ for each cell and
compared" is a method. Methods go in the methods section.

### 8. Discussion: how the gap was filled, the limits, the relevance

First paragraph summarises. Then the caveats, each linked to literature. The
`RESULTS.md` section "What is not established" is the raw material -- ten items,
and they should be *in* the paper, not waiting to be found.

### 9. Spend your time on title, abstract, figures and outline

These are read most. The five figures already exist and each carries one claim,
so the outline is nearly written: the figures **are** the argument's skeleton.

### 10. Get feedback; reduce, reuse, recycle

Give a colleague the abstract alone and ask what the paper claims. If the answer
is not the sentence in rule 1, the abstract is wrong.

---

## Part 2: sentences

From Gopen and Swan, *The Science of Scientific Writing*, American Scientist
(1990). The insight is that readers have fixed structural expectations, and
prose is hard to read when it violates them -- so clarity is a matter of
placement, not vocabulary.

### The two positions in a sentence

- **Topic position** (the beginning): where the reader looks for *what this
  sentence is about*, and for a link backward to what they already know.
- **Stress position** (the end): where the reader gives emphasis, and expects
  the new, important information.

The single most useful rule: **put old information in the topic position and new
information in the stress position.** Sentences then chain, each beginning where
the last ended.

Weak, because the new information is buried in the middle:

> A factor of 1.9 in spectral shape, which we measured across three structures,
> separates the simulated and real spectra.

Better, because "the simulated and real spectra" is old information and the
number lands in the stress position:

> The simulated and real spectra differ in shape by a factor of **1.9**.

### The other principles worth internalising

- **Follow a grammatical subject quickly with its verb.** Every word between
  them is read as an interruption. This is why long parenthetical qualifications
  after the subject are so tiring.
- **Articulate the action in the verb.** "We performed a comparison of" becomes
  "We compared". Nominalisations bury the action in a noun and force a weak verb
  to carry the sentence.
- **Provide context before new information.** A reader cannot evaluate a number
  they do not yet know the purpose of.
- **Make relative emphasis match structural emphasis.** If a clause sits in the
  stress position, the reader will treat it as the point. Check that it is.

### Applied to this paper

Numbers belong in the stress position, and the condition belongs in the topic
position. Not:

> At matched dose and iterations, +0.647 versus -0.251 was the correlation.

But:

> With dose and iteration count held fixed, the correlation changes sign:
> **+0.647 at 5% background against -0.251 at 31%.**

The condition first, because the reader must know what is being held fixed
before the numbers mean anything; the numbers last, because they are the point.

---

## Part 3: voice, and not sounding machine-generated

TMLR's own guidance states that submissions generated "with little or no human
involvement" are unlikely to meet its acceptance criteria. Beyond that policy
question there is a practical one: certain patterns now read as machine-written
to any regular reader, and they cost credibility whether or not a machine was
involved.

**The tell is cumulative.** No single word convicts. What marks text is the
repeated use of the same small set of moves. Empirical work tracking word
frequencies in biomedical and academic writing since 2023 shows sharp rises in a
specific vocabulary; the practical approach is to count patterns together rather
than police individual words.

### Vocabulary that now reads as machine-written

Avoid, or use at most once and deliberately: *delve, realm, landscape, journey,
tapestry, beacon, cacophony, transformative, underscore, harness, illuminate,
facilitate, bolster, pivotal, crucial, meticulous, robust* (as vague praise),
*nuanced, multifaceted, intricate, seamless, leverage* (as a verb), *showcase,
foster, testament, navigate* (figuratively), *unlock, unveil, shed light on*.

Note how many are metaphors doing no work. "This underscores the importance of"
means "this matters, because"; the first is longer and says less.

### Structural patterns that give it away

- **Transition words used as punctuation.** *Moreover, Furthermore, In addition,
  Additionally, It is worth noting that, It is important to note that.* Human
  writers use these occasionally. Machine text uses them at every paragraph
  boundary. If a paragraph opens with one, ask whether the logical relation is
  real -- usually the sentence is stronger without it.
- **"From X to Y" for describing range.** "From punctate structures to dense
  filaments" -- almost always replaceable with the concrete list.
- **Present participial phrases tacked on the end.** "The bound holds on all 48
  cells, **demonstrating** its robustness." This construction asserts the
  conclusion instead of earning it. Use a full clause: "The bound holds on all
  48 cells. It is therefore not an artefact of one setting."
- **Tricolon everywhere.** Three parallel items in every list, whether or not
  there are three things.
- **Symmetrical hedging.** "While X, it is also true that Y." Pick a position.
- **Section-closing summaries that repeat the section.** A paragraph that adds
  nothing but restates what precedes it.
- **Uniform paragraph and sentence length.** Human technical prose varies. A
  four-word sentence after three long ones is a legitimate emphasis device.
- **Confident vagueness.** "Significantly improves performance" with no number.
  This paper has numbers; use them.

### What to do instead

- **Be specific in preference to being emphatic.** "Considerably improves" is
  weaker than "+0.108". The number is the emphasis.
- **Let one voice choose the words.** Consistency of diction across sections is
  what makes prose read as authored.
- **State the awkward thing directly.** "No blind statistic worked on real data.
  We tried seven." That sentence is more credible than any amount of framing
  around it, and it is the kind of sentence machine drafts avoid.
- **Vary sentence length on purpose.** Put the important claim in a short
  sentence.
- **Keep the concrete noun as the subject.** "The residual inverts" beats "An
  inversion of the residual is observed".
- **Cut every sentence that only announces what you are about to do.** "In this
  section we present our results" earns nothing.

### A test before submission

Read the introduction aloud. Machine-written prose is fluent but rhythmically
flat -- every sentence lands the same way. If you cannot hear where the emphasis
falls, the reader will not find it either.

Then search the manuscript for the vocabulary list above and count. Five or six
hits in a short passage shifts a reader's judgement heavily, and once shifted it
colours their reading of the evidence too.

---

## Sources

- Mensh and Kording, *Ten simple rules for structuring papers*, PLOS
  Computational Biology 13(9), 2017.
  https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005619
- Gopen and Swan, *The Science of Scientific Writing*, American Scientist 78,
  1990. https://www.americanscientist.org/blog/the-long-view/the-science-of-scientific-writing
- TMLR acceptance criteria, on writing quality and on AI-generated submissions:
  https://jmlr.org/tmlr/acceptance-criteria.html
- Kobak et al., *Delving into "delved": the influence of LLMs on scientific
  writing*, on measured word-frequency shifts in academic and biomedical prose.
  https://www.medrxiv.org/content/10.1101/2024.05.31.24308296
