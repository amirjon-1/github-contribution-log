# Contribution [1]: [concat(a[n-1:n], a[n-2:n-1], ...., a[0:1]) to reverse #1409 — Enzyme-JAX]

**Contribution Number:** 1
**Student:** Amirjon Ulmasov  
**Issue:** [GitHub issue link](https://github.com/EnzymeAD/Enzyme-JAX/issues/1409)  
**Status:** Phase 4 Complete

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

**PR Link:** [https://github.com/EnzymeAD/Enzyme-JAX/pull/2589](https://github.com/EnzymeAD/Enzyme-JAX/pull/2589)

**PR Description:** Added ConcatSlicesToReverse canonicalization pattern that 
detects N unit-stride slices of the same tensor concatenated in reverse order 
and replaces the sequence with a single stablehlo.reverse op. Includes 3 lit 
tests. Fixes #1409.

**Maintainer Feedback:**
- [Date]: [Summary of feedback received]
- [Date]: [How you addressed it]

**Status:** Merged

---

## Learnings & Reflections

### Technical Skills Gained
First time writing a C++ MLIR canonicalization pattern. Learned how 
pattern-matching works in the MLIR rewrite framework, how to inspect op 
attributes like slice indices, and how to use replaceOpWithNewOp to emit 
a simpler op. Also got hands-on experience with Bazel builds and lit tests.

### Challenges Overcome
Merge conflict during rebase — someone had added new patterns to the same 
block in EnzymeHLOOpt.cpp. Resolved by keeping both sets of patterns and 
inserting ConcatSlicesToReverse in the right place.

### What I'd Do Differently Next Time
Rebase on upstream main before starting to avoid merge conflicts at PR time.

---

## Resources Used

- [Issue](https://github.com/EnzymeAD/Enzyme-JAX/issues/1409)



# Contribution [2]: slice(gather(x, ind)) -> gather(x, slice(ind)) #1924

**Contribution Number:** 2 
**Student:** Amirjon Ulmasov 
**Issue:** [GitHub issue link](https://github.com/EnzymeAD/Enzyme-JAX/issues/1924) 
**Status:** Phase 4 In Progress

---

## Why I Chose This Issue

After completing my first contribution to Enzyme-JAX (#1409), I wanted to 
stay in the same codebase and keep building on what I learned. This issue 
is the same class of problem — a compiler pattern that emits more ops than 
necessary. Instead of slicing the result of a gather, you can slice the 
indices first and gather fewer elements, which is cheaper.

I'm now familiar with EnzymeHLOOpt.cpp, the canonicalization pattern 
structure, the lit test format, and the Bazel build system, so I can 
move faster on this second contribution. The issue was filed by the same 
maintainer (avik-pal) who opened #1409, which means it's the same style 
of well-scoped, clearly described optimization request.

---

## Understanding the Issue

### Problem Description
The compiler emits a gather over the full index tensor followed by a slice 
of the result, when it could instead slice the indices first and gather 
fewer elements — avoiding unnecessary computation.

### Expected Behavior

slice(gather(x, ind)) should fold into gather(x, slice(ind)), processing 
only the elements actually needed.

### Current Behavior

The optimizer emits the full gather first, then slices the result. For 
large index tensors this wastes memory and compute.

### Affected Components

src/enzyme_ad/jax/Passes/EnzymeHLOOpt.cpp — same canonicalization pass 
as contribution 1.

---

## Reproduction Process

### Environment Setup

Same environment as contribution 1 — Bazel build already working on macOS. 
No additional setup needed.

### Steps to Reproduce

1. Checkout fix-issue-1924 branch
2. Create test .mlir file with slice(gather) pattern from the issue
3. Run: bazel run //:enzymexlamlir-opt -- --enzyme-hlo-opt /tmp/test_gather.mlir
4. Expected: gather on sliced indices
5. Actual: full gather followed by slice

### Reproduction Evidence

- **Commit showing reproduction:** https://github.com/amirjon-1/Enzyme-JAX/tree/fix-issue-1924
- **My findings:**  No pattern in EnzymeHLOOpt.cpp currently matches 
  slice(gather) — confirmed by running the optimizer and observing 
  the pattern is not folded.

---

## Solution Approach

### Analysis

No canonicalization rule exists that recognizes a slice of a gather result 
as equivalent to a gather on sliced indices. The maintainer also noted the 
gather must have only one user before applying the transformation.

### Proposed Solution

Add a pattern anchored on stablehlo::SliceOp that checks if its operand 
is a single-use GatherOp, then rewrites to a gather on the sliced indices.

### Implementation Plan

Using UMPIRE framework (adapted):

**Understand:** The compiler gathers all elements then slices, when it could 
slice indices first and gather fewer elements.

**Match:** ConcatSlicesToReverse from contribution 1 follows the same structure 
— anchor on the output op, check the input op, rewrite.

**Plan:** 
1. Add SliceOfGather pattern in EnzymeHLOOpt.cpp anchored on SliceOp
2. Check that the SliceOp's operand is a single-use GatherOp
3. Slice the gather's index tensor instead
4. Emit a new GatherOp on the sliced indices
5. Register in patterns.add, TransformOps.td, and EnzymeXLA.cpp
6. Add lit tests


**Review:** Follow same clang-format and commit conventions as PR #2589.

**Evaluate:** Lit test confirms gather on sliced indices is emitted, 
full gather+slice sequence is not.

---

## Testing Strategy

- [x] Test case 1: Single-use gather — slice(gather(x, ind)) folds into gather(x, slice(ind))
- [x] Test case 2: Negative case — gather with multiple users does NOT fold

### Manual Testing

Ran bazel test //test/lit_tests:slicegather.mlir.test after implementing 
the fix. All tests passed — pattern correctly folds single-use 
slice(gather) and correctly skips multi-user gathers.

---

## Implementation Notes

### Week 1 Progress

Implemented the SliceOfGather canonicalization pattern in 
src/enzyme_ad/jax/Passes/EnzymeHLOOpt.cpp. The pattern anchors on 
stablehlo::SliceOp and checks that its operand is a single-use GatherOp 
(per maintainer guidance from wsmoses). Maps the slice on the gather's 
batch output dims onto the corresponding start_indices dims, emits a new 
SliceOp on the index tensor, then emits a new GatherOp on the sliced 
indices preserving all gather attributes. Registered in patterns.add, 
TransformOps.td, and EnzymeXLA.cpp following the same steps as #2589. 
Both lit tests pass.

### Code Changes

- **Files modified:** src/enzyme_ad/jax/Passes/EnzymeHLOOpt.cpp,
  src/enzyme_ad/jax/TransformOps/TransformOps.td,
  src/enzyme_ad/jax/Integrations/c/EnzymeXLA.cpp,
  test/lit_tests/slicegather.mlir
- **Key commits:** https://github.com/amirjon-1/Enzyme-JAX/tree/fix-issue-1924
- **Approach decisions:** Anchored on SliceOp rather than GatherOp since 
  the slice is the output op. Single-user guard added per maintainer request 
  to avoid redundant computation.

---

## Pull Request

**PR Link:** https://github.com/EnzymeAD/Enzyme-JAX/pull/2650

**PR Description:** Added SliceOfGather canonicalization pattern that detects 
when a slice of a single-use gather result can be rewritten as a gather on 
sliced indices, avoiding redundant computation on the full index tensor. 
Includes single-user guard per maintainer guidance. Fixes #1924.

**Maintainer Feedback:**
- July 2025: vimarsh6739 requested three changes: convert while loop to if 
  for indexVectorDim advancement, add static shape checks for indices and 
  gather types, and noted batching dim slicing as a potential follow-up. 
  Applied all three changes — converted while to if, added 
  hasStaticShape() guards, and added a TODO comment for the batching dim 
  follow-up.
- July 2025: Updated affine_for_mem.mlir golden output after SliceOfGather 
  pattern began firing on that test's IR, producing correct but smaller 
  tensors (8x16 → 4x16).

**Status:** Iterating

---

## Learnings & Reflections

### Technical Skills Gained
Built on contribution 1 by tackling a more complex pattern — gather ops 
have significantly more attributes to reason about (dimension_numbers, 
slice_sizes, index_vector_dim) compared to simple slice/concat. Learned 
how to map output-space slices back onto the index tensor space and 
preserve gather semantics correctly.

### Challenges Overcome
The gather op's dimension_numbers attributes made the rewrite more involved 
than contribution 1. Had to carefully map the slice on batch output dims 
back to the corresponding start_indices dims without breaking the gather 
semantics.

### What I'd Do Differently Next Time
None

---

## Resources Used

- [Issue](https://github.com/EnzymeAD/Enzyme-JAX/issues/1924)
- [PR #2589](https://github.com/EnzymeAD/Enzyme-JAX/pull/2589) - my old pr used as a guide.
