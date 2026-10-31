/* Parse all local frontend sources using the exact vendored Babel compiler. */
const fs = require('node:fs');
const path = require('node:path');
const root = path.resolve(__dirname, '..');
const babel = require(path.join(root, 'webapp/vendor/babel.min.js'));
const files = fs.readdirSync(path.join(root, 'webapp')).filter(name => /\.(jsx|js)$/.test(name));
for (const name of files) {
  babel.transform(fs.readFileSync(path.join(root, 'webapp', name), 'utf8'), {
    filename: name, presets: ['react'], sourceType: 'module', ast: false, code: false,
  });
}
console.log(`Frontend syntax OK: ${files.length} files; Babel ${babel.version}`);
