// Copyright 2026 Intel Corporation
// Licensed under the Apache License, Version 2.0

// Tests for fold_details.js. Run: node .github/scripts/test_fold_details.js

const assert = require("assert");
const { foldDetails } = require("./fold_details.js");

let failures = 0;

function check(name, fn) {
  try {
    fn();
    console.log(`ok   ${name}`);
  } catch (error) {
    failures += 1;
    console.error(`FAIL ${name}\n     ${error.message}`);
  }
}

check("folds the body behind a summary and keeps the verdict visible", () => {
  const out = foldDetails("VERDICT: two issues.\n\n## Summary\nbody");
  assert.strictEqual(
    out,
    "VERDICT: two issues.\n\n<details><summary>Details</summary>\n\n## Summary\nbody\n\n</details>",
  );
});

check("keeps the blank line between the verdict and the fold", () => {
  // A single newline would let GitHub render the verdict and the summary as one
  // paragraph, so the separator is load-bearing rather than cosmetic.
  const out = foldDetails("VERDICT: x.\nbody");
  assert.ok(out.startsWith("VERDICT: x.\n\n<details>"));
});

check("tolerates leading blank lines before the verdict", () => {
  const out = foldDetails("\n\nVERDICT: x.\n\nbody");
  assert.ok(out.startsWith("VERDICT: x."));
  assert.ok(out.includes("<details>"));
});

check("accepts a bold verdict", () => {
  // Models reliably bold a leading label, and a plain /^VERDICT:/ would reject it.
  const out = foldDetails("**VERDICT:** x.\n\nbody");
  assert.ok(out.includes("<details>"));
});

check("returns the verdict alone when there is no body", () => {
  assert.strictEqual(foldDetails("VERDICT: all good.\n\n"), "VERDICT: all good.");
});

check("returns the text unchanged when there is no verdict", () => {
  const input = "## PR Review: #1\n\nbody";
  assert.strictEqual(foldDetails(input), input);
});

check("reports a missing verdict to the caller", () => {
  const seen = [];
  foldDetails("## PR Review: #1", (m) => seen.push(m));
  assert.strictEqual(seen.length, 1);
});

check("does not report when the verdict is present", () => {
  const seen = [];
  foldDetails("VERDICT: x.\n\nbody", (m) => seen.push(m));
  assert.strictEqual(seen.length, 0);
});

if (failures) {
  console.error(`\n${failures} test(s) failed`);
  process.exit(1);
}
console.log("\nall tests passed");
