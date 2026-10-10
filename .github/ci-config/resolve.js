// Shared resolver for the CI defaults in defaults.json.
// Loaded through actions/github-script by .github/actions/ci-config and
// .github/actions/setup-python. Set BASE_ENV to point at a different config file.
const fs = require('fs');
const path = require('path');

const DEFAULT_CONFIG = path.join(__dirname, 'defaults.json');

function resolve({ core, overrides = {}, config = '' }) {
  const file = config || process.env.BASE_ENV || DEFAULT_CONFIG;
  if (!fs.existsSync(file)) {
    throw new Error(`CI config not found: ${file}`);
  }

  const resolved = JSON.parse(fs.readFileSync(file, 'utf8'));
  for (const [key, value] of Object.entries(overrides)) {
    if (value) {
      resolved[key] = value;
    }
  }

  for (const [key, value] of Object.entries(resolved)) {
    if (!value) {
      throw new Error(`"${key}" is empty in ${file}`);
    }
    core.setOutput(key, value);
  }
  core.setOutput('json', JSON.stringify(resolved));
  // exported so later steps of the calling job can reuse the resolved version
  core.exportVariable('PYTHON_VERSION', resolved.python);
  core.info(`Resolved CI defaults from ${file}: ${JSON.stringify(resolved)}`);

  return resolved;
}

module.exports = { resolve };
