// Shared resolver for the CI defaults in defaults.yaml.
// Loaded through actions/github-script by .github/actions/ci-config and
// .github/actions/setup-python. Set BASE_ENV to point at a different config file.
const fs = require('fs');
const path = require('path');

const DEFAULT_CONFIG = path.join(__dirname, 'defaults.yaml');

// Two-level `key: value` YAML: scalars, or one level of nested scalar maps.
function parseYaml(text, file) {
  const out = {};
  let section = null;
  text.split(/\r?\n/).forEach((raw, i) => {
    const line = raw.replace(/(^|\s+)#.*$/, '').trimEnd();
    if (!line.trim()) return;
    const m = line.trim().match(/^([A-Za-z_][\w-]*):\s*(.*)$/);
    if (!m) throw new Error(`${file}:${i + 1}: expected "key: value"`);
    const value = m[2].replace(/^(['"])(.*)\1$/, '$2');
    if (/^\s/.test(line)) {
      if (!section) throw new Error(`${file}:${i + 1}: nested key outside a section`);
      out[section][m[1]] = value;
    } else if (value === '') {
      section = m[1];
      out[section] = {};
    } else {
      section = null;
      out[m[1]] = value;
    }
  });
  return out;
}

function resolve({ core, overrides = {}, config = '' }) {
  const file = config || process.env.BASE_ENV || DEFAULT_CONFIG;
  if (!fs.existsSync(file)) {
    throw new Error(`CI config not found: ${file}`);
  }

  const resolved = parseYaml(fs.readFileSync(file, 'utf8'), file);
  for (const [key, value] of Object.entries(overrides)) {
    if (value) {
      resolved.common[key] = value;
    }
  }

  // Outputs stay flat (python, op_ut_linux, ...) so callers keep their existing names.
  const seen = new Set();
  for (const section of Object.values(resolved)) {
    for (const [key, value] of Object.entries(section)) {
      if (!value) {
        throw new Error(`"${key}" is empty in ${file}`);
      }
      if (seen.has(key)) {
        throw new Error(`"${key}" is defined in more than one section of ${file}`);
      }
      seen.add(key);
      core.setOutput(key, value);
    }
  }
  core.setOutput('json', JSON.stringify(resolved));
  // exported so later steps of the calling job can reuse the resolved version
  core.exportVariable('PYTHON_VERSION', resolved.common.python);
  core.info(`Resolved CI defaults from ${file}: ${JSON.stringify(resolved)}`);

  return resolved;
}

module.exports = { resolve };
