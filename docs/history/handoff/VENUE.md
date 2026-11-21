# Reading a venue properly, and where this paper goes

Two things here: a **method** for assessing a venue that does not rely on
reputation or memory, and the **result** of applying it to this paper in
September 2026.

---

## Part 1: how to read a journal's aims and scope

Most venue choices are made from folklore. The following takes about an hour per
candidate and replaces folklore with evidence.

### Step 1. Read the acceptance criteria, not the scope blurb

Scope statements are marketing: broad, welcoming, and nearly identical across
journals in a field. The **acceptance criteria** and **reviewer guidelines** are
the operative documents, because they are what a reviewer is actually asked.

Look specifically for:

- **What the criteria do not mention.** A venue that never uses the words
  "novel" or "significant" has made a deliberate choice, and it changes what
  kind of paper survives review.
- **The default when a reviewer is unsure.** Some venues instruct reviewers to
  assume in the author's favour. That is a strong signal about acceptance rate
  and about tolerance for unusual work.
- **Explicit paper types.** Reproducibility studies, negative results, position
  papers, dataset papers. If a type is named, it is welcome; if it is absent,
  you are asking a reviewer to make an exception.

### Step 2. Find five papers the venue published that resemble yours

Not five papers you admire -- five that are structurally like yours in **what
kind of claim they make**. The relevant similarity is rarely topical.

Ask of each:

- Is the central contribution a method, a theorem, a measurement, or a negative
  result? Match on this, not on subject matter.
- How long is it, and how many figures? That tells you the real length policy,
  which is often looser or tighter than the stated one.
- How much of it is limitations? A venue whose papers carry two paragraphs of
  caveats will not object to yours; one whose papers carry none will.
- Are the results positive? A venue with no published negatives is not a venue
  for a negative paper, whatever its scope says.

Search the venue's own archive rather than Google Scholar. DBLP and the
journal's volume listings give complete coverage; a search engine gives you the
well-cited subset, which is a biased sample of what gets in.

### Step 3. Price the venue honestly

Article processing charges, page charges beyond a limit, and open-access
options. Then check what your institution actually covers -- not what it says it
covers. An unfunded page charge that arrives after acceptance is a bad surprise.

### Step 4. Check the competitive clock

Search arXiv for the last eighteen months in your exact sub-area, sorted by
date. You are looking for two things: whether your contribution is still
distinct, and **how fast the area is moving**. A venue with a nine-month review
cycle is the wrong choice in an area publishing three preprints a year on your
specific question.

This step is the one most often skipped and the most consequential.

### Step 5. Decide what the paper is, then pick

Write one sentence of the form "this paper establishes X" without hedging. Then
ask which venue's criteria that sentence satisfies. If the sentence needs a
"but" to fit a venue, that venue is wrong -- you will spend review defending the
mismatch rather than the work.

### The failure mode to avoid

Choosing a venue by prestige and then reverse-engineering the paper to fit. It
produces a paper that overclaims in the introduction, hedges in the discussion,
and satisfies no reviewer. Pick the venue whose question your evidence already
answers.

---

## Part 2: applying it to this paper

Checked against sources in September 2026. Every claim below has a URL at the
end of this file.

### The competitive clock: read this first

Self-supervised conformal prediction for imaging is a **crowded 2025 line**, and
all of it comes from the group the project's own Gate 0 analysis flagged:

| work | approach | validated on | operator mismatch? |
|---|---|---|---|
| arXiv 2502.05127 (SSVM 2025) | SURE-based self-calibration | denoising, deblurring | no |
| arXiv 2502.19194 | **Poisson** URE self-calibration | denoising, deblurring | no |
| arXiv 2605.18655 (MaxEnt 2025) | equivariant bootstrapping | weak lensing | no |

Two consequences.

**The planned self-supervised extension is dead as conceived.** "Estimate or
control d+ without labels" is precisely what these three do, and one is in the
Poisson regime this project works in. Entering that race a year late without a
differentiator is the worst available use of the time.

**But the novelty gap is real and now located.** None of the three addresses
forward-operator mismatch; all validate on simulation or on FFHQ and fastMRI.
The distinctive claims are:

1. **operator mismatch** as the shift axis, not measurement noise;
2. **real measurements with paired ground truth**, where the literature uses
   simulation;
3. **characterising where the guarantee breaks** rather than proposing a method
   that works;
4. the **out-of-focus background** finding, which quantitatively explains a
   sim-to-real gap.

Frame the paper around 1 to 3. Do **not** frame it as "a conformal prediction
method for imaging" -- that is a crowded framing this work would lose in, and it
invites comparison on exactly the axis where it is weakest.

Post to arXiv immediately. Three competing papers appeared in 2025 and one is
Poisson-specific. This is no longer cheap insurance.

### TMLR: the recommendation

*Transactions on Machine Learning Research.* Its acceptance criteria are two
questions, quoted verbatim:

> "Are the claims made in the submission supported by accurate and convincing
> evidence?"

> "Would some individuals in TMLR's audience be interested in the findings of
> this paper, and does the paper communicate those findings clearly to its
> intended audience?"

Why it fits, point by point:

- **It deliberately avoids "significant", "impactful" and "novel" as grounds for
  rejection**, and states that "novelty of the studied method is not a necessary
  criterion for acceptance". This paper's novelty is narrow by its own
  admission; at a novelty-driven venue that is fatal, here it is irrelevant.
- **Reviewers unsure about audience interest are told to assume it is
  satisfied**, and papers are not to be rejected for lacking state-of-the-art
  results. Three of this paper's strongest results are failures.
- **It exists partly to carry work "not yet accepted in mainstream venues but
  may be important in the future"**, and emphasises technical correctness over
  subjective significance. Eleven documented retractions demonstrate exactly the
  standard the first criterion asks about -- at most venues they would read as
  instability.
- **Variable manuscript length.** Eleven gates, five figures and a wide grid
  will not compress into eight pages.
- **Free**, which matters while institutional open-access funding is unresolved.
- **It publishes conformal prediction on imaging inverse problems** and has
  conformal expertise on its editorial board, so the reviewer pool exists.
- Rolling submissions, double-blind, on OpenReview.

One caution from its own guidance: TMLR states that AI-generated submissions
"with little or no human involvement" are unlikely to meet its criteria. The
writing must be yours. See `WRITING.md`.

### SIAM Journal on Imaging Sciences: a legitimate second choice

An earlier assessment downgraded SIIMS on the assumption that it would find an
elementary bound too thin. **That assumption was wrong.** SIIMS publishes
applied conformal work -- *Posterior-Variance-Based Error Quantification for
Inverse Problems in Imaging* (doi 10.1137/23M1546129) uses posterior variance
together with conformal prediction to obtain coverage guarantees on pixelwise
error bounds without distributional assumptions. That is adjacent to this paper.

The real reservation is **disposition, not scope**. A mathematics-leaning venue
is a harder sell for a paper whose strongest results are failures, and this
bound is elementary by the project's own description. If the paper is reframed
around the bound as a theoretical contribution, SIIMS becomes the better fit;
as an empirical characterisation paper, TMLR is.

### IEEE Transactions on Computational Imaging: the fallback

Viable on scope. Charges roughly $175 per page beyond about nine pages, so
realistically $875 to $1,225 for a paper this size -- and the paper has grown.
Only worth it if institutional funding is confirmed.

### Rule out: optics and bio-imaging venues

Optics Express, Biomedical Optics Express and similar would be interested in the
haze finding and the real-measurement validation. But the paper's headline is a
method whose deployable warning signal does not work, and those venues want a
working instrument. The negatives that make this a good TMLR paper make it a bad
optics paper.

### Sources

- TMLR home and scope: https://jmlr.org/tmlr/
- TMLR acceptance criteria: https://jmlr.org/tmlr/acceptance-criteria.html
- arXiv 2502.05127, self-supervised conformal prediction for imaging
- arXiv 2502.19194, self-supervised conformal prediction for Poisson imaging
- arXiv 2605.18655, equivariant bootstrapping for image uncertainty
- SIIMS, Posterior-Variance-Based Error Quantification, doi 10.1137/23M1546129
- Hore and Barber, randomised localised conformal prediction, arXiv:2310.07850
- Qiao et al., Nature Methods 18, 194-202 (2021), the BioSR dataset
