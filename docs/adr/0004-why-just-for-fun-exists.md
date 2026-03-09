# ADR-0004: Include a "Just for Fun" Section in a Technical Documentation Site

**Status**: Accepted

**Date**: 2024-03-10

**Deciders**: Joshua N. Grant

**Tags**: documentation, information-architecture

---

## Context

A personal technical documentation site that presents only enterprise patterns and production best practices risks becoming indistinguishable from thousands of other engineering blogs. It signals "I can execute corporate patterns" but not "I explore ideas at the edges of what's technically possible."

The site author builds projects that don't fit neatly into "Best Practices" or "Tutorials" — things like:

- A Raspberry Pi sample library server with USB MIDI live audition
- Recursive cathedral generators from L-system grammars
- Metrics sonification via SuperCollider + OSC
- WebGL generative art from PostGIS raster data

These projects are technically rigorous, reproducible, and often more instructive about system design than a standard tutorial — but their primary motivation is curiosity, not production deployment.

## Decision

Create a **"Just for Fun" subsection** under Tutorials. Include creative, experimental, and exploratory projects that are:

- Technically reproducible (not just concept posts)
- Non-trivial in implementation (demonstrate real engineering)
- Built for intrinsic interest, not business requirements

Do not artificially restrict the section to specific technology areas. Let it grow organically.

## Options Considered

### Option 1: Exclude creative projects entirely

**Pros:**
- Site maintains a purely professional tone
- No reader confusion about what the site is "for"

**Cons:**
- Loses the most distinctive content on the site
- The best projects — the ones that demonstrate creative problem-solving — get no home
- Portfolio signals only "follows conventional patterns," not "thinks originally"

### Option 2: Separate blog/personal site for fun projects

**Pros:**
- Clean separation of concerns
- Professional docs site stays focused

**Cons:**
- Two sites to maintain; cross-traffic is lost
- The technical depth of these projects is documentation-worthy, not blog-post-worthy
- Discoverability drops: readers who arrive for best practices never discover the creative work

### Option 3: "Just for Fun" section within the documentation site (chosen)

**Pros:**
- Readers who arrive for serious content discover unexpected creative depth
- The same documentation framework serves both content types
- Cross-links between fun projects and related best practices reinforce conceptual connections (e.g., Pi sample server → Docker best practices)
- Signals that the author is a whole engineer, not just a pattern-follower

**Cons:**
- Some readers may find the tone inconsistent with the rest of the site
- Section needs its own voice: playful but technically precise, not corporate

## Rationale

The "Just for Fun" section is a differentiator, not a distraction. Every page in the section demonstrates real engineering — recursive L-systems, WebAudio API internals, MIDI protocol handling, SQLite query optimization. The playful framing makes these approachable; the technical depth makes them credible. Excluding this content would make the site less interesting and less representative of how the author actually works.

## Consequences

### Positive
- Site has a unique character that distinguishes it from generic engineering blogs
- Creative projects get the same documentation quality as production content
- Cross-links from fun projects into best practices and tutorials create unexpected discovery paths

### Negative
- Section requires active curation to maintain quality bar (reproducible, non-trivial)
- Voice calibration is harder: playful but not cringe; experimental but not hand-wavy

### Neutral / Trade-offs
- "Managing People in Software Development" is in this section despite being non-technical — it fits the spirit of "unexpected but useful" content that defines the section

## Related Documents

- [Just for Fun Overview](../tutorials/just-for-fun/index.md)
- [ADR-0003: Best Practices vs Tutorials](0003-why-best-practices-vs-tutorials.md)
- [Technical Documentation](../documentation.md) — site structure and navigation
