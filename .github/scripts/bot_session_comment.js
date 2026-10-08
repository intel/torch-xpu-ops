// Copyright 2026 Intel Corporation
// Licensed under the Apache License, Version 2.0

// What the fix job's session comment should become once the agent has exited.
// Returns the new body, or null to leave the comment alone.
//
// The agent owns the comment and its format varies run to run, so the stage
// blocks it wrote are kept whenever they exist; its closing message only fills
// a gap: the whole comment if it never touched the placeholder, the summary if
// it stopped before writing one.

const PLACEHOLDER = 'Starting automated fix on this issue.';
const SUMMARY = '<!-- agent:summary -->';
const LIMIT = 65000; // GitHub's comment cap is 65536

function backfill(body, closing, runUrl) {
  body = body || '';
  if (body.includes(SUMMARY) || !closing) return null;
  if (body.startsWith(PLACEHOLDER)) {
    const footer = `\n\n---\n_The [\`fix\` job](${runUrl}) did not update this comment itself, so this is the agent's closing summary only._`;
    return fit(`<!-- agent:session -->\n\n`, closing, footer);
  }
  const head = `${body}\n\n${SUMMARY}\n\n**Summary** _(the agent's closing message; the run stopped before writing its own)_\n\n`;
  return fit(head, closing, '');
}

// Truncate the closing message, never the stage blocks in front of it.
function fit(head, closing, tail) {
  const note = '\n\n_(truncated -- full output in the job artifacts)_';
  const room = LIMIT - head.length - tail.length;
  if (closing.length <= room) return head + closing + tail;
  return head + closing.slice(0, Math.max(0, room - note.length)) + note + tail;
}

module.exports = { backfill, PLACEHOLDER };
