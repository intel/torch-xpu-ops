// Who may run `@torchxpubot fix`.
//
// `fix` drives an autonomous agent on a GPU runner, which is too much to hand
// to the MEMBER author_association this gate used to accept: MEMBER is every
// one of intel's ~2000 org members, ~1950 of whom have no role in this repo.
//
// SEED is the standing list. Everything past it is granted by comment on the
// access issue, so getting `fix` access does not need a PR:
//
//     @torchxpubot allow @login      <- APPROVERS only, on the access issue
//     @torchxpubot deny @login
//
// Later comments win, so a grant is revoked the same way it was made, and the
// issue's comment history is the audit trail.

const SEED = [
  // PyTorch core team
  'Chao1Han', 'fengyuan14', 'majing921201', 'minmingzhu', 'NeoZhangJianyu',
  'tye1', 'xiaolil1', 'YzyParry', 'zhangxiaoli73',
  // PyTorch upstream team
  'chunhuanMeng', 'CuiYifeng', 'etaf', 'guangyey', 'hoshibara', 'jianyizh',
  'laifenxiawucha', 'lchen2331', 'liangan1', 'LuFinch', 'msnliu',
  'NaOHCC', // Qin, Hang -- joining the intel org, not a member yet
  'orrangetabby17', 'Stonepia', 'weishi-deng', 'xiaowangintel', 'xuhancn',
  'yucai-pro', 'ZhaoqiongZ',
  // maintainers
  'EikanWang', 'gujinghui', 'riverliuintel',
  // CI / validation
  'chuanqi129', 'daisyden', 'mengfei25',
  // the CI DISABLED queue posts its triggers as this account
  'torchxpubot',
];

// Kept to admin/maintain/owner accounts on purpose: if every allowed account
// could grant, one grant would let the list grow to anyone.
const APPROVERS = ['chuanqi129', 'EikanWang', 'gujinghui', 'riverliuintel', 'tye1'];

// A GitHub login is alphanumerics and single inner hyphens, up to 39 chars.
// Anchored at a line start, so an approver quoting someone else's grant with a
// `>` prefix does not re-cast it.
const GRANT_RE = /^@torchxpubot[ \t]+(allow|deny)[ \t]+@?([A-Za-z\d](?:-?[A-Za-z\d]){0,38})\b/im;

const low = (s) => String(s).toLowerCase();

// comments: the access issue's comments, oldest first.
function allowedAccounts(comments) {
  const approvers = new Set(APPROVERS.map(low));
  const allowed = new Set(SEED.map(low));
  for (const c of comments) {
    if (!approvers.has(low(c.user.login))) continue;
    const m = (c.body || '').match(GRANT_RE);
    if (!m) continue;
    if (low(m[1]) === 'allow') {
      allowed.add(low(m[2]));
    } else {
      allowed.delete(low(m[2]));
    }
  }
  return allowed;
}

const canApprove = (login) => APPROVERS.map(low).includes(low(login));

module.exports = { SEED, APPROVERS, allowedAccounts, canApprove };

if (require.main === module && process.argv[2] === '--self-test') {
  const assert = require('assert');
  const c = (login, body) => ({ user: { login }, body });

  // The seed list stands on its own, and the accounts it names are real:
  // duplicates would hide a typo behind a name that already works.
  assert.strictEqual(new Set(SEED.map(low)).size, SEED.length, 'SEED has a duplicate');
  assert(APPROVERS.every((a) => SEED.map(low).includes(low(a))), 'approver missing from SEED');
  assert(allowedAccounts([]).has('guangyey'));
  assert(!allowedAccounts([]).has('nobody'));

  // A grant from an approver lands, whatever case either login is written in.
  // The fixture login is deliberately not in SEED, so these assert the grant
  // and not the seeding.
  assert(!SEED.map(low).includes('an-applicant'), 'fixture login must not be seeded');
  assert(allowedAccounts([c('EikanWang', '@torchxpubot allow @an-applicant')]).has('an-applicant'));
  assert(allowedAccounts([c('eikanwang', '@torchxpubot allow an-applicant')]).has('an-applicant'));

  // From anyone else it does not.
  assert(!allowedAccounts([c('guangyey', '@torchxpubot allow @an-applicant')]).has('an-applicant'));
  assert(!allowedAccounts([c('drive-by', '@torchxpubot allow @drive-by')]).has('drive-by'));

  // Last verdict wins, both ways, and deny reaches the seed list too.
  const grant = c('tye1', '@torchxpubot allow @an-applicant');
  const revoke = c('tye1', '@torchxpubot deny @an-applicant');
  assert(!allowedAccounts([grant, revoke]).has('an-applicant'));
  assert(allowedAccounts([grant, revoke, grant]).has('an-applicant'));
  assert(!allowedAccounts([c('chuanqi129', '@torchxpubot deny @mengfei25')]).has('mengfei25'));

  // A quoted grant is text, not a grant: GitHub's reply button prefixes `>`.
  assert(!allowedAccounts([c('tye1', '> @torchxpubot allow @an-applicant\n\nwho is this?')]).has('an-applicant'));

  // Prose mentioning the syntax, and the commands this file does not own.
  assert(!allowedAccounts([c('tye1', 'ask an approver to `@torchxpubot allow @you`')]).has('you'));
  assert(allowedAccounts([c('tye1', '@torchxpubot fix')]).size === SEED.length);

  assert(canApprove('EIKANWANG') && !canApprove('guangyey'));
  console.log(`ok: ${SEED.length} seeded, ${APPROVERS.length} approvers`);
}
