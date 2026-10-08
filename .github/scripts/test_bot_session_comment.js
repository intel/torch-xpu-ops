// Copyright 2026 Intel Corporation
// Licensed under the Apache License, Version 2.0

// node .github/scripts/test_bot_session_comment.js
const assert = require('assert');
const { backfill, PLACEHOLDER } = require('./bot_session_comment.js');

const url = 'https://example/run/1';
const staged = '### Agent fix run\n\n| Stage | Status |\n|---|---|\n| Triage | done |';

// Agent wrote its summary: untouched, whatever its markers (the #5272 overwrite case).
assert.strictEqual(backfill(`${staged}\n\n<!-- agent:summary -->\nfixed`, 'closing', url), null);

// Agent never touched the placeholder: closing message replaces it.
const fresh = backfill(`${PLACEHOLDER} Follow progress in the run.`, 'closing', url);
assert(fresh.startsWith('<!-- agent:session -->') && fresh.includes('closing') && fresh.includes(url));
assert(!fresh.includes(PLACEHOLDER));

// Agent stopped mid-pipeline: stage blocks kept, closing appended as the summary.
const partial = backfill(staged, 'closing', url);
assert(partial.startsWith(staged) && partial.endsWith('closing'));
assert(partial.includes('<!-- agent:summary -->'));

// No closing message (agent errored): leave the comment as it is.
assert.strictEqual(backfill(staged, '', url), null);

// Oversized: the closing message is cut, the stage blocks and footer are not.
const big = backfill(staged, 'x'.repeat(70000), url);
assert(big.length <= 65000 && big.startsWith(staged) && big.includes('truncated'));
const bigFresh = backfill(PLACEHOLDER, 'x'.repeat(70000), url);
assert(bigFresh.length <= 65000 && bigFresh.endsWith('closing summary only._'));

// The start step runs before checkout, so it cannot require PLACEHOLDER; keep its text in sync.
const yml = require('fs').readFileSync(require('path').join(__dirname, '../workflows/bot.yml'), 'utf8');
assert(yml.includes(`body: \`${PLACEHOLDER} Follow progress`), 'bot.yml start comment drifted from PLACEHOLDER');

console.log('ok');
