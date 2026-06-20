# Contribution [1]: [concat(a[n-1:n], a[n-2:n-1], ...., a[0:1]) to reverse #1409 — Enzyme-JAX]

**Contribution Number:** 1
**Student:** Amirjon Ulmasov  
**Issue:** [GitHub issue link](https://github.com/EnzymeAD/Enzyme-JAX/issues/1409)  
**Status:** Phase 3 Complete

---

## Why I Chose This Issue

I chose this issue because it connects directly to work I'm already 
doing. Through my undergraduate research with Professor Kelly Shaw at 
Williams College, I spend a lot of time optimizing CUDA kernels and 
profiling GPU performance. That work taught me to look for places where 
compute is being wasted and find ways to do less work for the same result.

This issue is that same idea applied at the compiler level. Right now, 
when code produces a tensor in reverse order, the compiler emits a bunch 
of individual slice and concat operations instead of just using a single 
reverse op. The fix is to teach the compiler to recognize that pattern 
and replace it with the simpler version. Fewer ops means less memory 
movement and faster execution on GPUs and other accelerators.

I picked Enzyme-JAX specifically because it's an active research project, 
the maintainers respond quickly, and it uses MLIR which is the same 
compiler infrastructure I see in GPU toolchains. This felt like a natural 
next step from my research and a good way to get real experience with 
compiler optimization in an ML context.

---

## Understanding the Issue

### Problem Description

When JAX reverses a tensor (like a[::-1]), the compiler outputs a bunch of 
individual slice operations followed by a concat instead of just using the 
built-in reverse op that already exists in StableHLO.

### Expected Behavior

The compiler should output a single stablehlo.reverse op. That's it.

### Current Behavior

Instead of one reverse op, you get N slice ops (one per row) plus a 
concatenate. For a 5-element tensor you get 5 slices + 1 concat = 6 ops 
where 1 would do.

### Affected Components

The canonicalization pass in Enzyme-JAX — the part of the compiler that 
looks for redundant patterns and simplifies them. Likely in src/enzyme_ad/jax/.

---

## Reproduction Process

### Environment Setup

Enzyme-JAX uses Bazel as its build system and requires bazel-6.5, clang++, 
python, python-virtualenv, and python3-dev. Rather than doing a full source 
build (which compiles LLVM/MLIR from scratch and takes hours), I used pip 
install enzyme-ad to get a working environment quickly, then cloned the fork 
to browse and modify the source. The relevant optimization pass is in 
src/enzyme_ad/jax/Passes/EnzymeHLOOpt.cpp.

### Steps to Reproduce

1. Clone fork and checkout fix-issue-1409
2. Build Enzyme-JAX following [build method from CI/README]
3. Create test file with the slice/concat MLIR pattern from the issue
4. Run through the compiler pipeline
5. Expected: stablehlo.reverse op emitted
6. Actual: individual slice + concatenate ops remain

### Reproduction Evidence

- **Commit showing reproduction:** [Link to commit in your fork](https://github.com/amirjon-1/Enzyme-JAX/tree/fix-issue-1409)
- **Screenshots/logs:** [If applicable]
- **My findings:** The pattern-matching logic for HLO optimizations lives in 
src/enzyme_ad/jax/Passes/EnzymeHLOOpt.cpp. This is where existing 
canonicalization rules are written — for example, folding redundant ops into 
simpler equivalents. There is currently no rule that matches the slice+concat 
reverse pattern, which confirms the issue. This is where the fix will go.

---

## Solution Approach

### Analysis

There's no pattern-matching rule that recognizes the slice+concat sequence 
as a reverse. The compiler just emits whatever JAX lowers to without checking 
if there's a simpler equivalent.

### Proposed Solution

Add a canonicalization rule that detects when slices cover a full axis in 
reverse order and are concatenated back together, then replaces the whole 
thing with a single stablehlo.reverse.

### Implementation Plan

Using UMPIRE framework (adapted):

**Understand:** The compiler emits N stablehlo.slice + stablehlo.concatenate ops when 
the operands form a complete reverse along one axis. A single stablehlo.reverse 
is equivalent and cheaper.

**Match:** Look for existing canonicalization patterns in the codebase — likely in
src/enzyme_ad/jax/ or similar — where patterns of ops get folded into simpler ops.

**Plan:** [Step-by-step implementation plan]
1. Add a canonicalization pattern (C++ rewrite rule) that detects the slice/concat 
   signature: consecutive unit-stride slices covering the full axis in reverse order, 
   all concatenated along that axis
2. Replace the matched pattern with stablehlo.reverse on the appropriate dimension
3. Add a lit test (.mlir file) that verifies the transformation fires correctly

**Implement:** [Link to your branch/commits as you work](https://github.com/amirjon-1/Enzyme-JAX/tree/fix-issue-1409)

**Review:** Check CONTRIBUTING.md for commit message format and test conventions.

**Evaluate:** The lit test should show CHECK lines confirming stablehlo.reverse appears 
and the slice/concat sequence does not.


---

## Testing Strategy

### Unit Tests

- [x] Test case 1: 3D tensor reverse along dim 0 (5×3×10, matches exact pattern from issue)
- [x] Test case 2: 1D tensor simple reverse
- [x] Test case 3: Negative case — forward-ordered slices do NOT fold (no false positives)

### Integration Tests

- [ ] Integration scenario 1
- [ ] Integration scenario 2

### Manual Testing

Ran bazel run //:enzymexlamlir-opt -- --enzyme-hlo-opt /tmp/test_reverse.mlir after 
implementing the fix. Output confirmed: 5 slice ops + 1 concatenate collapsed into 
a single stablehlo.reverse op along dim 0.

---

## Implementation Notes

### Week [1] Progress

Implemented the ConcatSlicesToReverse canonicalization pattern in 
src/enzyme_ad/jax/Passes/EnzymeHLOOpt.cpp. The pattern anchors on 
stablehlo::ConcatenateOp and fires when all operands are unit-stride 
SliceOps from the same source tensor, covering the full extent of the 
concat dimension in reverse order, with all other dimensions fully 
covered. On match, emits a single stablehlo::ReverseOp. Registered the 
pattern alongside ConcatFuse and ConcatToBroadcast in the patterns.add 
block. Also wrote a lit test with 3 cases (positive 3D, positive 1D, 
negative wrong-order). All three pass.

### Week [Y] Progress

[Continue documenting as you work]

### Code Changes

- **Files modified:** src/enzyme_ad/jax/Passes/EnzymeHLOOpt.cpp, 
  test/lit_tests/concat_slices_to_reverse.mlir
- **Key commits:** https://github.com/amirjon-1/Enzyme-JAX/tree/fix-issue-1409
- **Approach decisions:** Anchored the pattern on ConcatenateOp rather than 
  SliceOp since that's the output op and matches how similar patterns like 
  ConcatFuse are structured in the codebase.

---

## Pull Request

**PR Link:** [GitHub PR URL when submitted]

**PR Description:** [Draft or final PR description - much of the content above can be adapted]

**Maintainer Feedback:**
- [Date]: [Summary of feedback received]
- [Date]: [How you addressed it]

**Status:** [Awaiting review / Iterating / Approved / Merged]

---

## Learnings & Reflections

### Technical Skills Gained

[What you learned technically]

### Challenges Overcome

[What was hard and how you solved it]

### What I'd Do Differently Next Time

[Reflection on your process]

---

## Resources Used

- [Link to helpful documentation]
- [Tutorial or Stack Overflow post that helped]
- [GitHub issues or discussions that helped]
